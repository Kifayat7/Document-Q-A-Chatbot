"""Configurable embedding model factory (Hugging Face local or OpenAI)."""
from __future__ import annotations

from config.settings import Settings
from src.errors import ConfigError


def get_embeddings(settings: Settings):
    """Return a LangChain ``Embeddings`` object for the configured provider.

    * ``huggingface`` - runs locally via sentence-transformers (no API key).
    * ``openai``      - needs ``OPENAI_API_KEY``.
    """
    if settings.embedding_provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ConfigError(
                "Install 'langchain-huggingface' and 'sentence-transformers' to use local embeddings."
            ) from exc
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is not set, but EMBEDDING_PROVIDER=openai.")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)

    raise ConfigError(f"Unsupported embedding provider: {settings.embedding_provider!r}")
