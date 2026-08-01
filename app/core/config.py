"""Application configuration loaded from the environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root, i.e. the directory holding .env (app/core/config.py -> app/core -> app -> root).
BASE_DIR = Path(__file__).resolve().parents[2]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Environment-backed settings, validated once at load time."""

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
    APP_NAME: str = "langchain-branch-parallel-chatbot"
    APP_ENV: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: LogLevel = "INFO"

    @property
    def is_production(self) -> bool:
        return self.APP_ENV == "production"


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance.

    Settings are not instantiated at import time so that a missing GROQ_API_KEY
    fails where it is used rather than breaking every import of this module.
    """
    return Settings()
