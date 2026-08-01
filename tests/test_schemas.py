"""Tests for the Pydantic data contract (FR-3)."""

import pytest
from pydantic import ValidationError

from app.schemas.request import MAX_QUERY_LENGTH, ChatRequest
from app.schemas.response import ChatBotResponse

VALID_RESPONSE = {
    "answer": "42",
    "summary": "The answer is 42.",
    "confidence": 0.5,
    "category": "General",
    "keywords": ["douglas adams"],
}


def build_response(**overrides) -> ChatBotResponse:
    return ChatBotResponse(**{**VALID_RESPONSE, **overrides})


class TestChatRequest:
    def test_strips_surrounding_whitespace(self):
        assert ChatRequest(query="  hello  ").query == "hello"

    @pytest.mark.parametrize("query", ["", "   ", "\n\t "])
    def test_rejects_blank_queries(self, query):
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(query=query)
        assert exc_info.value.errors()[0]["type"] == "string_too_short"

    def test_accepts_a_query_at_the_limit(self):
        query = "x" * MAX_QUERY_LENGTH
        assert len(ChatRequest(query=query).query) == MAX_QUERY_LENGTH

    def test_rejects_a_query_one_character_over_the_limit(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(query="x" * (MAX_QUERY_LENGTH + 1))
        assert exc_info.value.errors()[0]["type"] == "string_too_long"

    def test_padding_is_stripped_before_the_limit_is_applied(self):
        query = " " * 50 + "x" * MAX_QUERY_LENGTH + " " * 50
        assert len(ChatRequest(query=query).query) == MAX_QUERY_LENGTH

    def test_rejects_unknown_fields(self):
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(query="hi", user_id=7)
        assert exc_info.value.errors()[0]["type"] == "extra_forbidden"

    def test_rejects_a_missing_query(self):
        with pytest.raises(ValidationError):
            ChatRequest()


class TestChatBotResponseConfidence:
    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_accepts_the_full_inclusive_range(self, confidence):
        assert build_response(confidence=confidence).confidence == confidence

    @pytest.mark.parametrize(
        ("confidence", "error_type"),
        [
            (-0.01, "greater_than_equal"),
            (1.01, "less_than_equal"),
            (100, "less_than_equal"),
        ],
    )
    def test_rejects_values_outside_the_range(self, confidence, error_type):
        with pytest.raises(ValidationError) as exc_info:
            build_response(confidence=confidence)
        assert exc_info.value.errors()[0]["type"] == error_type

    def test_coerces_an_integer_to_a_float(self):
        response = build_response(confidence=1)
        assert isinstance(response.confidence, float)
        assert response.confidence == 1.0

    def test_rejects_a_non_numeric_confidence(self):
        with pytest.raises(ValidationError) as exc_info:
            build_response(confidence="very sure")
        assert exc_info.value.errors()[0]["type"] == "float_parsing"


class TestChatBotResponseKeywords:
    def test_deduplicates_ignoring_case(self):
        response = build_response(keywords=["Python", "python", "PYTHON", "lists"])
        assert response.keywords == ["Python", "lists"]

    def test_keeps_the_first_spelling_and_the_model_ordering(self):
        response = build_response(keywords=["Recursion", "base case", "RECURSION"])
        assert response.keywords == ["Recursion", "base case"]

    def test_strips_and_drops_blank_entries(self):
        response = build_response(keywords=["  spacing  ", "", "   ", "tabs"])
        assert response.keywords == ["spacing", "tabs"]

    def test_rejects_an_empty_list(self):
        with pytest.raises(ValidationError) as exc_info:
            build_response(keywords=[])
        assert exc_info.value.errors()[0]["type"] == "too_short"

    def test_rejects_a_list_that_is_empty_after_cleaning(self):
        with pytest.raises(ValidationError) as exc_info:
            build_response(keywords=["", "   "])
        assert "at least one non-empty entry" in str(exc_info.value)

    def test_rejects_more_keywords_than_the_cap(self):
        with pytest.raises(ValidationError) as exc_info:
            build_response(keywords=[f"kw{index}" for index in range(11)])
        assert exc_info.value.errors()[0]["type"] == "too_long"


class TestChatBotResponseText:
    @pytest.mark.parametrize("field", ["answer", "summary", "category"])
    def test_rejects_blank_text_fields(self, field):
        with pytest.raises(ValidationError) as exc_info:
            build_response(**{field: "   "})
        assert exc_info.value.errors()[0]["type"] == "string_too_short"

    @pytest.mark.parametrize("field", ["answer", "summary", "category", "confidence", "keywords"])
    def test_every_field_is_required(self, field):
        payload = {key: value for key, value in VALID_RESPONSE.items() if key != field}
        with pytest.raises(ValidationError) as exc_info:
            ChatBotResponse(**payload)
        assert exc_info.value.errors()[0]["type"] == "missing"

    def test_strips_whitespace_from_text_fields(self):
        response = build_response(answer="  padded  ", category="  Math  ")
        assert (response.answer, response.category) == ("padded", "Math")


class TestStructuredOutputSchema:
    """The JSON schema is the prompt the LLM sees, so its content is load-bearing."""

    def test_every_field_carries_a_description(self):
        properties = ChatBotResponse.model_json_schema()["properties"]
        missing = [name for name, spec in properties.items() if not spec.get("description")]
        assert missing == []

    def test_all_fields_are_required(self):
        schema = ChatBotResponse.model_json_schema()
        assert set(schema["required"]) == set(VALID_RESPONSE)

    def test_confidence_bounds_reach_the_schema(self):
        confidence = ChatBotResponse.model_json_schema()["properties"]["confidence"]
        assert (confidence["minimum"], confidence["maximum"]) == (0.0, 1.0)

    def test_class_docstring_is_addressed_to_the_model(self):
        # This description is sent to the LLM, so it must not contain notes to developers.
        description = ChatBotResponse.model_json_schema()["description"]
        assert "developer" not in description.lower()
