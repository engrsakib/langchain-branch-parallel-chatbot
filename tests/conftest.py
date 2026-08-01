"""Shared fixtures and fake models.

No test in this suite may reach the network, so every chain is built with
injected fakes and settings come from environment variables rather than a
developer's local .env file.
"""

import pytest
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableLambda

from app.chatbot.chain import get_chat_chain
from app.core.config import Settings, get_settings
from app.schemas.response import ChatBotResponse


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    """Give every test the same settings, independent of any local .env."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_MODEL", "test-answer-model")
    monkeypatch.setenv("GROQ_ROUTER_MODEL", "test-router-model")
    monkeypatch.setenv("GROQ_TEMPERATURE", "0.1")
    monkeypatch.setenv("APP_ENV", "development")
    get_settings.cache_clear()
    get_chat_chain.cache_clear()
    yield
    get_settings.cache_clear()
    get_chat_chain.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def sample_response() -> ChatBotResponse:
    return ChatBotResponse(
        answer="Use reversed(items) to walk a list backwards.",
        summary="reversed() returns a lazy backwards iterator.",
        confidence=0.8,
        category="Programming",
        keywords=["python", "reversed"],
    )


class RecordingRouter:
    """A stand-in chat model that always returns the same category word."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[ChatPromptValue] = []

    def as_runnable(self) -> RunnableLambda:
        return RunnableLambda(self._invoke)

    def _invoke(self, prompt_value: ChatPromptValue):
        from langchain_core.messages import AIMessage

        self.calls.append(prompt_value)
        return AIMessage(content=self.reply)


class RecordingAnswerLLM:
    """Duck-types the only method the chain calls on the answering model."""

    def __init__(self, response: ChatBotResponse) -> None:
        self.response = response
        self.calls: list[ChatPromptValue] = []
        self.structured_schema: type | None = None

    def with_structured_output(self, schema, **_kwargs) -> RunnableLambda:
        self.structured_schema = schema
        return RunnableLambda(self._invoke)

    def _invoke(self, prompt_value: ChatPromptValue) -> ChatBotResponse:
        self.calls.append(prompt_value)
        return self.response

    @property
    def system_prompts(self) -> list[str]:
        return [str(call.to_messages()[0].content) for call in self.calls]


@pytest.fixture
def recording_router():
    return RecordingRouter


@pytest.fixture
def recording_answer_llm(sample_response):
    def _factory(response: ChatBotResponse | None = None) -> RecordingAnswerLLM:
        return RecordingAnswerLLM(response or sample_response)

    return _factory
