# LangChain Multi-Branch & Parallel Chatbot

[![CI](https://github.com/engrsakib/langchain-branch-parallel-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/engrsakib/langchain-branch-parallel-chatbot/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-F55036?logo=groq&logoColor=white)](https://console.groq.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.60-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-E92063?logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Pytest](https://img.shields.io/badge/Pytest-passing-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://docs.docker.com/compose/)
[![License](https://img.shields.io/badge/License-Apache%202.0-D22128?logo=apache&logoColor=white)](LICENSE)

A production-grade Streamlit chatbot built on **LangChain Expression Language
(LCEL)**, **Pydantic v2** and **Groq**. Every question is classified, routed to a
subject-matter specialist, and answered as a schema-validated object rather than
a wall of text.

---

## Overview

General-purpose chatbots answer a debugging question and a calculus question in
the same voice, and hand back prose that an application then has to parse with
regular expressions and hope.

This project fixes both problems:

- **A router picks the right expert.** A small, fast model classifies each
  question as Programming, Math or General, and a `RunnableBranch` dispatches to
  a prompt written for that discipline. A debugging question gets a senior
  engineer; an integral gets a tutor who shows the working.
- **The answer arrives as data.** Every reply is a validated `ChatBotResponse`
  carrying an answer, a summary, a confidence score, a category and keywords.
  Downstream code reads typed fields, so the UI can render a confidence bar and
  keyword tags without parsing anything.

The result is a chatbot whose output is safe to build on: routing decisions are
logged, confidence is explicit, and malformed model output is rejected at the
boundary instead of leaking into the interface.

---

## Key technical innovations

### Dynamic routing with `RunnableBranch`

A dedicated classifier runs on `llama-3.1-8b-instant` at temperature 0 and is
asked for exactly one word. Its answer is matched leniently, so a chatty reply
such as `"The category is Programming."` still routes correctly, and an
unrecognised answer degrades to the General specialist with a logged warning
rather than raising. The branch is exhaustive by construction.

### Concurrent fan-out with `RunnableParallel`

The routing stage runs classification while passing the query and conversation
history through untouched, gathering all three into one dict for the branch that
follows. `RunnablePassthrough.assign` then merges the specialist's answer back
into that dict, which is how the router's category survives to the final step and
overrides whatever category the answering model reported.

### Schema enforcement with Pydantic

`ChatBotResponse` is both the runtime validator and part of the prompt. Bound to
the model through `with_structured_output`, Pydantic serialises it into the JSON
schema the LLM receives as a tool definition, so every field description is
written as an instruction to the model. Constraints are real: confidence outside
0.0–1.0 is rejected, blank answers are rejected, and keywords are deduplicated
case-insensitively.

Full technical detail lives in [`docs/architecture.md`](docs/architecture.md).

---

## Project structure

```text
langchain-branch-parallel-chatbot/
│
├── app/
│   ├── core/
│   │   ├── config.py          # Pydantic Settings loaded from .env
│   │   └── logger.py          # Centralised console logging
│   ├── schemas/
│   │   ├── request.py         # ChatRequest: inbound validation
│   │   └── response.py        # ChatBotResponse: the structured output contract
│   ├── chatbot/
│   │   ├── prompts.py         # Router prompt + three specialist personas
│   │   ├── chain.py           # The LCEL pipeline
│   │   └── memory.py          # Per-session history in st.session_state
│   ├── services/
│   │   └── chat_service.py    # Orchestration and error translation
│   ├── utils/
│   │   └── helpers.py         # Sanitising, formatting, session-state repair
│   └── main.py                # Streamlit entrypoint
│
├── tests/                     # pytest suite, no network access required
├── docs/architecture.md       # System architecture
├── .github/workflows/ci.yml   # Lint, format check and tests
├── Dockerfile                 # Container image
├── docker-compose.yml         # One-command local stack
└── requirements.txt           # Pinned dependencies
```

---

## Getting started

### Prerequisites

- Python 3.10 or newer
- A Groq API key from [console.groq.com](https://console.groq.com)
- Docker Desktop, if you want the container route

### Local setup

**1. Clone the repository**

```bash
git clone https://github.com/engrsakib/langchain-branch-parallel-chatbot.git
cd langchain-branch-parallel-chatbot
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your environment**

```bash
# macOS / Linux
cp .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

Open `.env` and set `GROQ_API_KEY` to your own key. Every other variable has a
working default.

**5. Run the app**

```bash
streamlit run app/main.py
```

Streamlit opens <http://localhost:8501> automatically. If configuration is
missing the app still starts and tells you which variable is wrong, rather than
crashing with a traceback.

---

## Docker

### Auto-run with Compose

From the project root, with your `.env` in place:

```bash
docker compose up --build
```

That single command builds the image, injects `.env`, starts Streamlit bound to
`0.0.0.0` inside the container, and publishes it locally. Open:

**<http://localhost:8501>**

Stop the stack with `Ctrl+C`, or from another terminal:

```bash
docker compose down
```

### Useful commands

```bash
docker compose up --build -d          # start detached
docker compose logs -f chatbot-app    # follow logs
docker compose ps                     # show status, including health
docker compose down -v                # stop and remove volumes
```

### How the container is wired

- **Health check** — `curl` polls `/_stcore/health` every 30 seconds after a
  25-second grace period, so `docker compose ps` reports real readiness rather
  than merely "running".
- **Host binding** — the server binds `0.0.0.0` inside the container, which is
  what makes it reachable from your host browser.
- **Secrets** — `.env` is injected at runtime through `env_file` and excluded
  from the build context by `.dockerignore`, so keys are never baked into the
  image.
- **Non-root** — the image runs as uid 1000.
- **Layer caching** — dependencies are installed before application code is
  copied, so editing Python files does not reinstall the dependency tree.

> **Port note:** the published port comes from `DOCKER_CHATBOT_PORT_EXTERNAL`,
> which defaults to `8501`. If you set it to something else in `.env`, browse to
> that port instead.

---

## Environment variables

| Variable                       | Required | Default                       | Purpose                                                    |
| ------------------------------ | -------- | ----------------------------- | ---------------------------------------------------------- |
| `GROQ_API_KEY`                 | Yes      | –                             | Groq API key. Held as a `SecretStr` and never rendered.     |
| `GROQ_MODEL`                   | No       | `llama-3.3-70b-versatile`     | Answering model. Must support tool calling.                 |
| `GROQ_ROUTER_MODEL`            | No       | `llama-3.1-8b-instant`        | Small model used only for classification.                   |
| `GROQ_TEMPERATURE`             | No       | `0.3`                         | Answer temperature, 0.0–2.0. Routing is always 0.           |
| `APP_NAME`                     | No       | `langchain-branch-parallel-chatbot` | Shown in the sidebar and browser tab.                 |
| `APP_ENV`                      | No       | `development`                 | One of `development`, `staging`, `production`.              |
| `LOG_LEVEL`                    | No       | `INFO`                        | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`.          |
| `DOCKER_CHATBOT_PORT_INTERNAL` | No       | `8501`                        | Port Streamlit binds inside the container.                  |
| `DOCKER_CHATBOT_PORT_EXTERNAL` | No       | `8501`                        | Port published on your machine.                             |

Invalid values fail fast: an unknown `LOG_LEVEL` or a non-numeric
`GROQ_TEMPERATURE` is reported at startup with the offending variable named.

---

## Testing

The suite runs entirely offline. Models are replaced with recording fakes, so no
Groq key and no network are needed.

```bash
pytest                    # everything
pytest -v                 # verbose
pytest tests/test_chains.py
pytest --cov=app          # coverage, requires pytest-cov
```

What is covered:

| File               | Focus                                                              |
| ------------------ | ------------------------------------------------------------------ |
| `test_schemas.py`  | Request stripping and limits, confidence bounds, keyword dedup      |
| `test_chains.py`   | Routing matrix, parallel stage, structured output, `get_chat_chain` |
| `test_helpers.py`  | Input sanitising, badge formatting, session-state repair            |

Style is enforced with [Ruff](https://docs.astral.sh/ruff/):

```bash
ruff check .
ruff format --check .
```

---

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push and
pull request targeting `main` or `master`. It installs the pinned dependencies
on Python 3.10, enforces lint and formatting with Ruff, then runs the test
suite.

---

## License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE).
