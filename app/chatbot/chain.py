
from functools import lru_cache
from operator import itemgetter
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_groq import ChatGroq

from app.chatbot.prompts import (
    CATEGORIES,
    DEFAULT_CATEGORY,
    MATH,
    PROGRAMMING,
    PROMPT_BY_CATEGORY,
    ROUTER_PROMPT,
    Category,
)
from app.core.config import Settings, get_settings
from app.core.logger import get_logger
from app.schemas.request import ChatRequest
from app.schemas.response import ChatBotResponse

logger = get_logger(__name__)

                                                                               
                                  
ChainInput = str | ChatRequest | dict[str, Any]


def _coerce_input(payload: ChainInput) -> dict[str, Any]:
    if isinstance(payload, ChatRequest):
        return {"query": payload.query, "history": []}
    if isinstance(payload, str):
        return {"query": ChatRequest(query=payload).query, "history": []}
    if isinstance(payload, dict):
        request = ChatRequest(query=payload.get("query", ""))
        return {"query": request.query, "history": list(payload.get("history") or [])}
    raise TypeError(f"Unsupported chain input: {type(payload).__name__}")


def _parse_category(raw_category: str) -> Category:
    text = raw_category.casefold()
    for category in CATEGORIES:
        if category.casefold() in text:
            logger.debug("Routed query to %s", category)
            return category

    logger.warning(
        "Router returned %r, which matches no known category; using %s",
        raw_category.strip(),
        DEFAULT_CATEGORY,
    )
    return DEFAULT_CATEGORY


def _apply_routed_category(payload: dict[str, Any]) -> ChatBotResponse:
    response: ChatBotResponse = payload["response"]
    return response.model_copy(update={"category": payload["category"]})


def _build_llm(settings: Settings, *, model: str, temperature: float) -> ChatGroq:
    return ChatGroq(
        model=model,
        api_key=settings.GROQ_API_KEY,
        temperature=temperature,
    )


def _build_router(router_llm: BaseChatModel) -> Runnable[dict[str, Any], Category]:
    return ROUTER_PROMPT | router_llm | StrOutputParser() | RunnableLambda(_parse_category)


def _build_answer_branch(answer_llm: BaseChatModel) -> Runnable[dict[str, Any], ChatBotResponse]:
    structured_llm = answer_llm.with_structured_output(ChatBotResponse)
    specialists = {
        category: prompt | structured_llm for category, prompt in PROMPT_BY_CATEGORY.items()
    }
    return RunnableBranch(
        (lambda payload: payload["category"] == PROGRAMMING, specialists[PROGRAMMING]),
        (lambda payload: payload["category"] == MATH, specialists[MATH]),
        specialists[DEFAULT_CATEGORY],
    )


def build_chat_chain(
    settings: Settings | None = None,
    *,
    answer_llm: BaseChatModel | None = None,
    router_llm: BaseChatModel | None = None,
) -> Runnable[ChainInput, ChatBotResponse]:
    settings = settings or get_settings()
    answer_llm = answer_llm or _build_llm(
        settings, model=settings.GROQ_MODEL, temperature=settings.GROQ_TEMPERATURE
    )
    router_llm = router_llm or _build_llm(
        settings, model=settings.GROQ_ROUTER_MODEL, temperature=0.0
    )

    routing_stage = RunnableParallel(
        query=itemgetter("query"),
        history=itemgetter("history"),
        category=_build_router(router_llm),
    )

    return (
        RunnableLambda(_coerce_input)
        | routing_stage
        | RunnablePassthrough.assign(response=_build_answer_branch(answer_llm))
        | RunnableLambda(_apply_routed_category)
    ).with_config(run_name="chat_chain")


@lru_cache(maxsize=1)
def get_chat_chain() -> Runnable[ChainInput, ChatBotResponse]:
    settings = get_settings()
    logger.info(
        "Building chat chain (answer model=%s, router model=%s)",
        settings.GROQ_MODEL,
        settings.GROQ_ROUTER_MODEL,
    )
    return build_chat_chain(settings)
