# System Architecture

How a question travels from the browser to a validated answer, and why the
pieces are arranged this way.

## Layers

```text
Streamlit UI            app/main.py            widgets, rendering, error display
  |
Session memory          app/chatbot/memory.py  per-browser-session chat history
  |
Chat service            app/services/          validation, error translation, logging
  |
LCEL chain              app/chatbot/chain.py   routing, fan-out, structured output
  |
Prompts                 app/chatbot/prompts.py router + three specialist personas
  |
Groq API                langchain-groq         llama-3.1-8b / llama-3.3-70b
```

Each layer only knows the one below it. The UI never imports LangChain, and the
chain never imports Streamlit, so the chain can be exercised in tests with no
browser and no network.

## LCEL pipeline execution

The chain is a single `Runnable` composed with the `|` operator, built once by
`build_chat_chain()` and cached by `get_chat_chain()`:

```text
RunnableLambda(_coerce_input)
  -> RunnableParallel(query, history, category)
  -> RunnablePassthrough.assign(response=RunnableBranch(...))
  -> RunnableLambda(_apply_routed_category)
```

Because the whole pipeline is one `Runnable`, it inherits `invoke`, `batch`,
`ainvoke` and streaming without any extra code, and the graph can be inspected
with `chain.get_graph()`.

**Stage 1 — `_coerce_input`.** Accepts a bare string, a `ChatRequest`, or a dict
carrying `query` and `history`, and normalises it to a dict. The query is
validated here, so a blank message fails before a single token is bought.

**Stage 2 — `RunnableParallel`.** Classifies the query while passing `query` and
`history` through untouched.

**Stage 3 — `RunnableBranch`.** Selects one specialist prompt and calls the
answering model bound to the response schema.

**Stage 4 — `_apply_routed_category`.** Returns the `ChatBotResponse` with the
router's category applied.

## RunnableBranch decision matrix

The router is a small, fast model at temperature 0 asked for exactly one word.
Its reply is matched leniently against the known categories:

| Router reply                   | Matched category | Specialist prompt              |
| ------------------------------ | ---------------- | ------------------------------ |
| `Programming`                  | Programming      | Senior software engineer       |
| `Math`                         | Math             | Patient mathematics tutor      |
| `General`                      | General          | General assistant              |
| `  math  ` / `MATH`            | Math             | Patient mathematics tutor      |
| `The category is Programming.` | Programming      | Senior software engineer       |
| `Astrology`, `""`, anything    | General          | General assistant (with a log) |

Two properties matter here. First, the fallback is total: an unrecognised reply
degrades to the General specialist and logs a warning rather than raising, so a
chatty router cannot take down a request. Second, the branch is exhaustive by
construction, since `RunnableBranch` requires a default.

Ambiguous follow-ups are the known weak spot. Asked to "write it as a Python
function" directly after a calculus exchange, the router will sometimes answer
Math. Measured across a set of eight follow-up cases, the small router, a
"classify only the new message" variant, and the 70B model all scored 7 of 8,
failing on different cases. The cheapest configuration was kept. A mis-route
only changes which persona answers; the response is still schema-valid.

## RunnableParallel concurrent execution

`RunnableParallel` runs its branches concurrently in a thread pool (or as
coroutines under `ainvoke`) and gathers them into one dict:

```text
          { "query": "...", "history": [...] }
                        |
        +---------------+---------------+
        |               |               |
   itemgetter      itemgetter      ROUTER_PROMPT | router_llm
    ("query")      ("history")      | StrOutputParser | _parse_category
        |               |               |
        +---------------+---------------+
                        |
     { "query": ..., "history": [...], "category": "Math" }
```

Only the classification branch does real work; the other two are pass-throughs
that keep the original input available to the branch downstream. This shape is
deliberate. Summary, confidence and keywords all describe the answer, so they
cannot be computed *beside* the answer without describing something that does
not exist yet. Generating them in the same structured call as the answer costs
one request instead of three and makes it impossible for the summary to
disagree with the answer it summarises.

`RunnablePassthrough.assign` in stage 3 is the same idea: it runs the branch and
merges the result into the existing dict, so the router's category is still in
scope when the final step needs it.

## Pydantic data contract boundary

Two models bracket the chain, and nothing crosses either boundary unvalidated.

**Inbound — `ChatRequest`.** Strips whitespace before checking length, so a
message of only spaces is rejected as empty. Caps length at 4000 characters and
forbids unknown fields.

**Outbound — `ChatBotResponse`.** Bound to the model through
`with_structured_output(ChatBotResponse)`. This model is doing double duty: it
is both the runtime validator and the prompt. Pydantic serialises it to a JSON
schema that is sent to the LLM as a tool definition, which is why every field
carries a description written as an instruction, and why the class docstring is
addressed to the model rather than to developers.

| Field        | Constraint             | Enforced by                       |
| ------------ | ---------------------- | --------------------------------- |
| `answer`     | non-empty              | `min_length=1` after strip        |
| `summary`    | non-empty              | `min_length=1` after strip        |
| `confidence` | 0.0 – 1.0 inclusive    | `ge` / `le`, surfaced in `schema` |
| `category`   | non-empty              | `min_length=1`, router overrides  |
| `keywords`   | 1–10, deduplicated     | `field_validator`, case-folded    |

If the model returns something that does not fit, Pydantic raises inside the
chain and `ChatService` converts it into a readable message rather than a
traceback.

## Error translation

`ChatService` is the only place that knows about Groq's exception hierarchy. It
maps each failure to a message safe to render, logs the technical detail, and
raises `ChatServiceError`:

| Failure                            | What the user sees                      |
| ---------------------------------- | --------------------------------------- |
| `AuthenticationError`              | Check `GROQ_API_KEY` in your `.env`     |
| `PermissionDeniedError`            | Key cannot use the configured model     |
| `RateLimitError`                   | Rate limited, wait and retry            |
| `APIConnectionError` / timeout     | Could not reach Groq, check the network |
| other `APIStatusError`             | Groq returned HTTP `<status>`           |
| `ValidationError` from the chain   | Reply did not match the expected format |
| anything else                      | Generic message, full traceback logged  |

The UI stores failures as turns in session state, so an error stays on screen
after the next rerun instead of vanishing.

## Session model

Streamlit reruns the entire script on every interaction, so all per-user state
lives in `st.session_state`, keyed per browser session. Each turn keeps the
whole `ChatBotResponse`, letting the metadata panel redraw without re-calling
the model. Only the last ten messages are replayed to the LLM, and error turns
are excluded because they are UI text, not something the model said.

## Docker containerisation model

```text
docker compose up --build
        |
        +-- builds python:3.10-slim image
        |     requirements.txt copied first  -> cached dependency layer
        |     application code copied second -> rebuilt on every edit
        |     runs as non-root uid 1000
        |
        +-- injects .env through env_file (secrets never enter the image)
        +-- publishes 8501 -> http://localhost:8501
        +-- health check polls /_stcore/health every 30s
```

The port is set once, as `STREAMLIT_SERVER_PORT`. Streamlit reads it natively,
the health check interpolates it, and Compose publishes it, so the three cannot
disagree. `.dockerignore` keeps `.env`, virtualenvs and caches out of the build
context; the container receives configuration only through `env_file`.

The Compose file bind-mounts the project directory for local development, which
means code edits reload without a rebuild. Remove that volume to run strictly
what was baked into the image.
