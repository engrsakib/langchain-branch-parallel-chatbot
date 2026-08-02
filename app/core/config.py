from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    GROQ_API_KEY: SecretStr = Field(
        ...,
        description="Groq API key; wrapped in SecretStr so it is masked in reprs and logs.",
    )
    GROQ_MODEL: str = Field(
        "llama-3.3-70b-versatile",
        description="Groq model that answers; must support tool calling for structured output.",
    )
    GROQ_ROUTER_MODEL: str = Field(
        "llama-3.1-8b-instant",
        description="Smaller, cheaper Groq model used for the one-word routing decision.",
    )
    GROQ_TEMPERATURE: float = Field(0.3, ge=0.0, le=2.0)

    APP_NAME: str = "langchain-branch-parallel-chatbot"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: LogLevel = "INFO"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
