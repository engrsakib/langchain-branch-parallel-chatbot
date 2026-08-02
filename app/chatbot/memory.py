from dataclasses import dataclass
from typing import Literal

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.schemas.response import ChatBotResponse
from app.utils.helpers import state_list

MESSAGES_KEY = "chat_messages"


MAX_HISTORY_MESSAGES = 10

Role = Literal["user", "assistant"]


@dataclass
class ChatTurn:
    role: Role
    content: str
    response: ChatBotResponse | None = None
    is_error: bool = False


def init_history() -> None:
    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []


def get_history() -> list[ChatTurn]:
    init_history()
    return state_list(st.session_state, MESSAGES_KEY, ChatTurn)


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
    return _append(ChatTurn(role="assistant", content=message, is_error=True))


def get_langchain_history(limit: int = MAX_HISTORY_MESSAGES) -> list[BaseMessage]:
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
