
import asyncio
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableParallel
from pydantic import ValidationError

from app.chatbot.chain import (
    _coerce_input,
    _parse_category,
    build_chat_chain,
    get_chat_chain,
)
from app.chatbot.prompts import GENERAL, MATH, PROGRAMMING
from app.schemas.request import ChatRequest
from app.schemas.response import ChatBotResponse

PERSONA_MARKERS = {
    PROGRAMMING: "software engineer",
    MATH: "mathematics tutor",
    GENERAL: "general assistant",
}


@pytest.fixture
def make_chain(settings, recording_router, recording_answer_llm):

    def _make(router_reply: str = GENERAL, response: ChatBotResponse | None = None):
        router = recording_router(router_reply)
        answer_llm = recording_answer_llm(response)
        chain = build_chat_chain(settings, answer_llm=answer_llm, router_llm=router.as_runnable())
        return chain, router, answer_llm

    return _make


class TestCategoryParsing:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Programming", PROGRAMMING),
            ("Math", MATH),
            ("General", GENERAL),
            ("  math  ", MATH),
            ("PROGRAMMING", PROGRAMMING),
            ("The category is Programming.", PROGRAMMING),
            ("Astrology", GENERAL),
            ("", GENERAL),
        ],
    )
    def test_maps_router_replies_onto_known_categories(self, raw, expected):
        assert _parse_category(raw) == expected


class TestInputCoercion:
    def test_accepts_a_plain_string(self):
        assert _coerce_input("  hi  ") == {"query": "hi", "history": []}

    def test_accepts_a_chat_request(self):
        assert _coerce_input(ChatRequest(query="hi"))["query"] == "hi"

    def test_accepts_a_dict_with_history(self):
        history = [HumanMessage(content="earlier")]
        coerced = _coerce_input({"query": "hi", "history": history})
        assert coerced == {"query": "hi", "history": history}

    def test_defaults_missing_history_to_empty(self):
        assert _coerce_input({"query": "hi"})["history"] == []

    @pytest.mark.parametrize("payload", ["", "   ", {"query": ""}, {}])
    def test_rejects_a_blank_query(self, payload):
        with pytest.raises(ValidationError):
            _coerce_input(payload)

    @pytest.mark.parametrize("payload", [12345, None, ["hi"]])
    def test_rejects_unsupported_payloads(self, payload):
        with pytest.raises(TypeError):
            _coerce_input(payload)


class TestRunnableBranchRouting:
    @pytest.mark.parametrize("category", [PROGRAMMING, MATH, GENERAL])
    def test_each_category_reaches_its_specialist(self, make_chain, category):
        chain, _, answer_llm = make_chain(router_reply=category)

        result = chain.invoke("a question")

        assert result.category == category
        assert PERSONA_MARKERS[category] in answer_llm.system_prompts[0]

    def test_unknown_category_falls_back_to_the_general_specialist(self, make_chain):
        chain, _, answer_llm = make_chain(router_reply="Astrology")

        result = chain.invoke("a question")

        assert result.category == GENERAL
        assert PERSONA_MARKERS[GENERAL] in answer_llm.system_prompts[0]

    def test_the_specialist_prompt_pins_the_routed_category(self, make_chain):
        chain, _, answer_llm = make_chain(router_reply=MATH)

        chain.invoke("2 + 2")

        assert f"category: exactly {MATH}" in answer_llm.system_prompts[0]

    def test_only_one_specialist_runs_per_query(self, make_chain):
        chain, router, answer_llm = make_chain(router_reply=PROGRAMMING)

        chain.invoke("a question")

        assert len(router.calls) == 1
        assert len(answer_llm.calls) == 1


class TestRunnableParallelStage:
    def test_the_routing_stage_is_a_runnable_parallel(self, make_chain):
        chain, _, _ = make_chain()

        parallel_steps = [step for step in chain.steps if isinstance(step, RunnableParallel)]

        assert parallel_steps, "no RunnableParallel found in the chain"
        assert set(parallel_steps[0].steps__) == {"query", "history", "category"}

    def test_query_and_history_survive_the_parallel_stage(self, make_chain):
        chain, router, answer_llm = make_chain(router_reply=MATH)
        history = [HumanMessage(content="earlier turn"), AIMessage(content="earlier reply")]

        chain.invoke({"query": "the new question", "history": history})

        specialist_messages = answer_llm.calls[0].to_messages()
        assert [type(message).__name__ for message in specialist_messages] == [
            "SystemMessage",
            "HumanMessage",
            "AIMessage",
            "HumanMessage",
        ]
        assert specialist_messages[-1].content == "the new question"

    def test_history_also_reaches_the_router(self, make_chain):
        chain, router, _ = make_chain(router_reply=MATH)
        history = [HumanMessage(content="earlier turn"), AIMessage(content="earlier reply")]

        chain.invoke({"query": "follow-up", "history": history})

        assert len(router.calls[0].to_messages()) == 4

    def test_batch_invocation_answers_every_input(self, make_chain):
        chain, _, answer_llm = make_chain(router_reply=MATH)

        results = chain.batch(["1 + 1", "2 + 2"])

        assert [result.category for result in results] == [MATH, MATH]
        assert len(answer_llm.calls) == 2

    def test_async_invocation_returns_the_same_shape(self, make_chain):
        chain, _, _ = make_chain(router_reply=GENERAL)

        result = asyncio.run(chain.ainvoke("hello"))

        assert isinstance(result, ChatBotResponse)


class TestStructuredOutput:
    def test_the_chain_returns_a_validated_model(self, make_chain):
        chain, _, _ = make_chain(router_reply=PROGRAMMING)

        result = chain.invoke("a question")

        assert isinstance(result, ChatBotResponse)

    def test_the_response_schema_is_bound_to_the_model(self, make_chain):
        _, _, answer_llm = make_chain()
        assert answer_llm.structured_schema is ChatBotResponse

    def test_the_router_category_overrides_the_model_answer(self, make_chain, sample_response):
                                                                   
        confused = sample_response.model_copy(update={"category": "General"})
        chain, _, _ = make_chain(router_reply=MATH, response=confused)

        assert chain.invoke("2 + 2").category == MATH

    def test_the_rest_of_the_response_is_untouched(self, make_chain, sample_response):
        chain, _, _ = make_chain(router_reply=MATH, response=sample_response)

        result = chain.invoke("2 + 2")

        assert result.answer == sample_response.answer
        assert result.summary == sample_response.summary
        assert result.confidence == sample_response.confidence
        assert result.keywords == sample_response.keywords


class TestGetChatChain:

    def test_builds_both_models_from_settings(self, settings):
        with patch("app.chatbot.chain.ChatGroq") as chat_groq:
            chat_groq.return_value = MagicMock()
            get_chat_chain()

        keywords = [call.kwargs for call in chat_groq.call_args_list]
        assert [kwargs["model"] for kwargs in keywords] == [
            settings.GROQ_MODEL,
            settings.GROQ_ROUTER_MODEL,
        ]

    def test_routes_at_temperature_zero(self, settings):
        with patch("app.chatbot.chain.ChatGroq") as chat_groq:
            chat_groq.return_value = MagicMock()
            get_chat_chain()

        answer_kwargs, router_kwargs = (call.kwargs for call in chat_groq.call_args_list)
        assert answer_kwargs["temperature"] == settings.GROQ_TEMPERATURE
        assert router_kwargs["temperature"] == 0.0

    def test_passes_the_api_key_through(self):
        with patch("app.chatbot.chain.ChatGroq") as chat_groq:
            chat_groq.return_value = MagicMock()
            get_chat_chain()

        api_key = chat_groq.call_args_list[0].kwargs["api_key"]
        assert api_key.get_secret_value() == "test-key"

    def test_returns_a_runnable(self):
        with patch("app.chatbot.chain.ChatGroq") as chat_groq:
            chat_groq.return_value = MagicMock()
            chain = get_chat_chain()

        assert isinstance(chain, Runnable)

    def test_is_cached_so_models_are_built_once(self):
        with patch("app.chatbot.chain.ChatGroq") as chat_groq:
            chat_groq.return_value = MagicMock()
            first = get_chat_chain()
            second = get_chat_chain()

        assert first is second
        assert chat_groq.call_count == 2                                      
