"""Structured output schema produced by the LLM (FR-3)."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatBotResponse(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    answer: str = Field(
        ...,
        min_length=1,
        description="The complete answer to the user's question, written for the user.",
    )
    summary: str = Field(
        ...,
        min_length=1,
        description="A concise summary of the answer, one to two sentences long.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "How confident you are in the answer, as a number between 0.0 and 1.0, "
            "where 0.0 is a pure guess and 1.0 is certain."
        ),
    )
    category: str = Field(
        ...,
        min_length=1,
        description=(
            "The topic category of the question, such as 'Programming', 'Math', or 'General'."
        ),
    )
    keywords: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Three to six keywords describing the topic of the question.",
    )

    @field_validator("keywords")
    @classmethod
    def _drop_blank_and_duplicate_keywords(cls, keywords: list[str]) -> list[str]:
        """Deduplicate case-insensitively, preserving the order the model chose."""
        seen: set[str] = set()
        cleaned: list[str] = []
        for keyword in keywords:
            stripped = keyword.strip()
            if not stripped or stripped.casefold() in seen:
                continue
            seen.add(stripped.casefold())
            cleaned.append(stripped)
        if not cleaned:
            raise ValueError("keywords must contain at least one non-empty entry")
        return cleaned
