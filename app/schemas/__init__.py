"""Pydantic schemas for chatbot input and structured output."""

from app.schemas.request import ChatRequest
from app.schemas.response import ChatBotResponse

__all__ = ["ChatRequest", "ChatBotResponse"]
