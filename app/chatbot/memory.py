"""Per-session chat history, stored in Streamlit's session state (FR-6).

Streamlit reruns the whole script on every interaction, so anything that must
survive between turns lives in ``st.session_state``. Each turn keeps the full
``ChatBotResponse`` alongside the text so the metadata panel can be redrawn on
later reruns without calling the model again.
"""

from dataclasses import dataclass
from typing import Literal

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.schemas.response import ChatBotResponse

MESSAGES_KEY = "chat_messages"

# Only the most recent turns are replayed to the model, to bound prompt size.
MAX_HISTORY_MESSAGES = 10

Role = Literal["user", "assistant"]


@dataclass
class ChatTurn:
    """One rendered message: a user question, an answer, or a failure notice."""

    role: Role
    content: str
    response: ChatBotResponse | None = None
    is_error: bool = False


def init_history() -> None:
    """Create the history list on first run of a session."""
    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []


def get_history() -> list[ChatTurn]:
    """Return every turn in the current session, oldest first."""
    init_history()
    return st.session_state[MESSAGES_KEY]


def has_history() -> bool:
    return bool(get_history())


def _append(turn: ChatTurn) -> ChatTurn:
    get_history().append(turn)
    return turn


def add_user_message(content: str) -> ChatTurn:
    return _append(ChatTurn(role="user", content=content))


def add_assistant_message(response: ChatBotResponse) -> ChatTurn:
    return _append(ChatTurn(role="assistant", content=response.answer, response=response))


def add_error_message(message: str) -> ChatTurn:
    """Record a failure as a turn so it survives the next rerun."""
    return _append(ChatTurn(role="assistant", content=message, is_error=True))


def get_langchain_history(limit: int = MAX_HISTORY_MESSAGES) -> list[BaseMessage]:
    """Convert recent turns into LangChain messages for the prompt placeholder.

    Failed turns are skipped: an error notice is UI text, not something the
    model said, and replaying it would only confuse the next answer.
    """
    messages: list[BaseMessage] = []
    for turn in get_history():
        if turn.is_error:
            continue
        if turn.role == "user":
            messages.append(HumanMessage(content=turn.content))
        else:
            messages.append(AIMessage(content=turn.content))
    return messages[-limit:] if limit else messages


def clear_history() -> None:
    st.session_state[MESSAGES_KEY] = []
