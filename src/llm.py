"""Configurable chat-model factory (OpenAI, Anthropic or local Ollama)."""
from __future__ import annotations

from config.settings import Settings
from src.errors import ConfigError


def get_llm(settings: Settings):
    """Return a LangChain chat model for ``settings.llm_provider``."""
    provider = settings.llm_provider
    if not settings.has_api_key(provider):
        env_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider, "")
        raise ConfigError(f"{env_name} is not set. Add it to your .env file or choose another provider in Settings.")

    try:
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=settings.llm_model, temperature=settings.llm_temperature, api_key=settings.openai_api_key)
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(model=settings.llm_model, temperature=settings.llm_temperature, api_key=settings.anthropic_api_key)
        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=settings.llm_model, temperature=settings.llm_temperature, base_url=settings.ollama_base_url)
    except ImportError as exc:
        raise ConfigError(f"The LangChain integration for '{provider}' is not installed: {exc}") from exc
    raise ConfigError(f"Unsupported LLM provider: {provider!r}")
