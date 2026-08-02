
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from app.chatbot import memory  # noqa: E402
from app.core.config import Settings, get_settings  # noqa: E402
from app.core.logger import get_logger  # noqa: E402
from app.schemas.response import ChatBotResponse  # noqa: E402
from app.services.chat_service import ChatService, ChatServiceError  # noqa: E402
from app.utils.helpers import format_confidence, format_keyword_badges  # noqa: E402

logger = get_logger(__name__)

APP_TITLE = "LangChain Branch & Parallel Chatbot"
APP_TAGLINE = (
    "Every question is routed to a Programming, Math, or General specialist, "
    "then returned as a validated structured answer."
)
EXAMPLE_QUESTIONS = [
    "Why does my for loop skip the last item?",
    "What is the derivative of x^3 * ln(x)?",
    "Solve (1 + i)z = (2 + 2i, 4) for the complex vector z.",
    "A 2 kg block slides down a frictionless 30 degree incline. What is its acceleration?",
    "Balance the combustion of propane and explain each step.",
    "Who wrote the novel Dune?",
]
CATEGORY_COLOURS = {"Programming": "violet", "Math": "blue", "General": "green"}

PENDING_PROMPT_KEY = "pending_prompt"


@st.cache_resource(show_spinner=False)
def load_service() -> ChatService:
    return ChatService()


def _load_settings() -> tuple[Settings | None, ValidationError | None]:
    try:
        return get_settings(), None
    except ValidationError as exc:
        return None, exc


def _render_config_error(error: ValidationError) -> None:
    st.title(APP_TITLE)
    st.error("The app is not configured yet, so it cannot start.")
    missing = ", ".join(str(err["loc"][0]) for err in error.errors()) or "unknown"
    st.markdown(
        f"Invalid or missing settings: **{missing}**.\n\n"
        "Copy `.env.example` to `.env` and fill in your Groq API key, "
        "available from [console.groq.com](https://console.groq.com)."
    )


def _detail(label: str, value: str) -> None:
    label_col, value_col = st.columns([0.9, 1.1], vertical_alignment="center")
    label_col.caption(label)
    value_col.markdown(f"`{value}`")


def _render_sidebar(settings: Settings) -> None:
    with st.sidebar:
        st.subheader("App settings")
        _detail("Name", settings.APP_NAME)
        _detail("Answer model", settings.GROQ_MODEL)
        _detail("Router model", settings.GROQ_ROUTER_MODEL)
        _detail("Temperature", f"{settings.GROQ_TEMPERATURE:g}")

        st.divider()

        st.subheader("Environment")
        _detail("Mode", settings.APP_ENV)
        _detail("Log level", settings.LOG_LEVEL)
        _detail("Groq API key", "set" if settings.GROQ_API_KEY.get_secret_value() else "missing")
        _detail("Messages", str(len(memory.get_history())))

        st.divider()

        if st.button("Clear chat", width="stretch", disabled=not memory.has_history()):
            memory.clear_history()
            st.rerun()


def _render_metadata(response: ChatBotResponse) -> None:
    with st.expander("Response details", expanded=False):
        category_col, confidence_col = st.columns([1, 2], vertical_alignment="center")
        with category_col:
            st.caption("Category")
            st.badge(response.category, color=CATEGORY_COLOURS.get(response.category, "gray"))
        with confidence_col:
            st.caption("Confidence")
            st.progress(response.confidence, text=format_confidence(response.confidence))

        st.caption("Summary")
        st.info(response.summary)

        st.caption("Keywords")
        st.markdown(format_keyword_badges(response.keywords))


def _render_turn(turn: memory.ChatTurn) -> None:
    with st.chat_message(turn.role):
        if turn.is_error:
            st.error(turn.content)
            return
        st.markdown(turn.content)
        if turn.response is not None:
            _render_metadata(turn.response)


def _render_history() -> None:
    for turn in memory.get_history():
        _render_turn(turn)


def _render_empty_state() -> str | None:
    st.caption("Try asking:")
    columns = st.columns(2)
    clicked: str | None = None
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        column = columns[index % len(columns)]
        if column.button(question, key=f"example-{index}", width="stretch", type="tertiary"):
            clicked = question
    return clicked


def _handle_prompt(prompt: str) -> None:
    history = memory.get_langchain_history()
    memory.add_user_message(prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = load_service().respond(prompt, history=history)
        except ChatServiceError as exc:
            memory.add_error_message(str(exc))
            st.error(str(exc))
            return
        except Exception as exc:
            logger.exception("Unhandled error while answering from the UI")
            message = f"Something went wrong: {exc}"
            memory.add_error_message(message)
            st.error(message)
            return

        memory.add_assistant_message(response)
        st.markdown(response.answer)
        _render_metadata(response)


def main() -> None:
    settings, config_error = _load_settings()

    st.set_page_config(
        page_title=settings.APP_NAME if settings else APP_TITLE,
        layout="centered",
        initial_sidebar_state="expanded",
    )

    if config_error is not None or settings is None:
        _render_config_error(config_error)
        return

    memory.init_history()

    st.title(APP_TITLE)
    st.caption(APP_TAGLINE)

    _render_sidebar(settings)

    pending = st.session_state.pop(PENDING_PROMPT_KEY, None)

    if not memory.has_history() and not pending:
        example = _render_empty_state()
        if example:
            st.session_state[PENDING_PROMPT_KEY] = example
            st.rerun()

    _render_history()

    prompt = st.chat_input("Ask about code, maths, science, or anything else") or pending
    if prompt:
        _handle_prompt(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
