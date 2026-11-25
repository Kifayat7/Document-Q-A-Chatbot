"""Embedding model factory: always uses free local Hugging Face embeddings."""
from __future__ import annotations

from functools import lru_cache

from config.settings import Settings
from src.errors import ConfigError

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def _load_model():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as exc:
        raise ConfigError(
            "Install 'langchain-huggingface' and 'sentence-transformers' to use local embeddings."
        ) from exc
    return HuggingFaceEmbeddings(
        model_name=HF_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_embeddings(settings: Settings):
    """Return the Hugging Face embeddings (loaded once, then reused)."""
    return _load_model()
