"""Chat-model factory: always uses Groq (free tier)."""
from __future__ import annotations

import os

from config.settings import Settings
from src.errors import ConfigError

# Own variable name, so an old LLM_MODEL value in secrets can't bring back a retired model.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def _groq_key(settings: Settings) -> str:
    key = os.getenv("GROQ_API_KEY") or getattr(settings, "groq_api_key", "")
    if not key:
        # Your Groq key may still be stored under the OpenAI name from the earlier workaround.
        old = os.getenv("OPENAI_API_KEY", "") or getattr(settings, "openai_api_key", "")
        if old.startswith("gsk_"):
            key = old
    return key


def get_llm(settings: Settings):
    """Return a Groq chat model."""
    key = _groq_key(settings)
    if not key:
        raise ConfigError("GROQ_API_KEY is not set. Add it in Streamlit Secrets.")
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ConfigError(f"'langchain-groq' is not installed: {exc}") from exc
    return ChatGroq(model=GROQ_MODEL, temperature=settings.llm_temperature, api_key=key)
