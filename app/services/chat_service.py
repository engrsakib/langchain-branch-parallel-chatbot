"""Orchestration layer between the UI and the LCEL chain.

The UI should never see a LangChain or Groq exception. Everything raised out of
this module is a ``ChatServiceError`` whose message is safe to render verbatim,
with the underlying failure logged for whoever is reading the logs.
"""

from collections.abc import Sequence

from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    PermissionDeniedError,
    RateLimitError,
)
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import ValidationError

from app.chatbot.chain import get_chat_chain
from app.core.logger import get_logger
from app.schemas.request import ChatRequest
from app.schemas.response import ChatBotResponse

logger = get_logger(__name__)


class ChatServiceError(RuntimeError):
    """A failure with a message intended for the person using the app."""


class ChatService:
    """Runs one user turn through the chain and normalises the failure modes."""

    def __init__(self, chain: Runnable | None = None) -> None:
        self._chain = chain if chain is not None else get_chat_chain()

    def respond(
        self,
        query: str | ChatRequest,
        history: Sequence[BaseMessage] | None = None,
    ) -> ChatBotResponse:
        """Answer one question, raising ChatServiceError if anything goes wrong."""
        request = self._validate(query)
        payload = {"query": request.query, "history": list(history or [])}

        try:
            response = self._chain.invoke(payload)
        except AuthenticationError as exc:
            logger.error("Groq rejected the API key: %s", exc)
            raise ChatServiceError(
                "The Groq API key was rejected. Check GROQ_API_KEY in your .env file."
            ) from exc
        except PermissionDeniedError as exc:
            logger.error("Groq denied access to the configured model: %s", exc)
            raise ChatServiceError(
                "This API key is not allowed to use the configured model. "
                "Check GROQ_MODEL and your Groq account's model access."
            ) from exc
        except RateLimitError as exc:
            logger.warning("Groq rate limit hit: %s", exc)
            raise ChatServiceError(
                "Groq is rate limiting this key. Wait a moment and try again."
            ) from exc
        except (APIConnectionError, APITimeoutError) as exc:
            logger.warning("Could not reach Groq: %s", exc)
            raise ChatServiceError(
                "Could not reach Groq. Check your network connection and try again."
            ) from exc
        except APIStatusError as exc:
            logger.error("Groq returned %s: %s", exc.status_code, exc)
            raise ChatServiceError(
                f"Groq returned an error (HTTP {exc.status_code}). Please try again."
            ) from exc
        except ValidationError as exc:
            # The model answered, but its structured output did not fit the schema.
            logger.warning("Model output failed ChatBotResponse validation: %s", exc)
            raise ChatServiceError(
                "The model's reply did not match the expected format. Please try again."
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected failure while answering a query")
            raise ChatServiceError(f"Something went wrong: {exc}") from exc

        logger.info(
            "Answered a %s query with confidence %.2f", response.category, response.confidence
        )
        return response

    @staticmethod
    def _validate(query: str | ChatRequest) -> ChatRequest:
        if isinstance(query, ChatRequest):
            return query
        try:
            return ChatRequest(query=query)
        except ValidationError as exc:
            logger.debug("Rejected an invalid query: %s", exc)
            raise ChatServiceError("Please enter a question before sending.") from exc
