from pydantic import BaseModel, ConfigDict, Field

MAX_QUERY_LENGTH = 4000


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="The user's question. Surrounding whitespace is stripped, "
        "and a blank or whitespace-only query is rejected.",
    )
