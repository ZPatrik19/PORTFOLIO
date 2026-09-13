from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Runtime configuration kept explicit for reproducible portfolio experiments."""

    mode: str = os.getenv("AGENT_MODE", "offline").lower()
    methodology: str = os.getenv("AGENT_METHODOLOGY", "auto").lower()
    language: str = os.getenv("APP_LANGUAGE", "en").lower()
    model: str = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    max_agent_steps: int = int(os.getenv("MAX_AGENT_STEPS", "8"))


def get_settings() -> Settings:
    return Settings()
