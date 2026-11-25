"""Central, environment-driven configuration for DocuMind.

All secrets come from environment variables (or a local ``.env`` file that is
git-ignored). Nothing in this module is ever rendered in the UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.errors import ConfigError

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

SUPPORTED_LLM_PROVIDERS = ("openai", "anthropic", "ollama", "groq")
SUPPORTED_EMBEDDING_PROVIDERS = ("huggingface", "openai")

DEFAULT_LLM_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-5-5",
    "ollama": "llama3.1",
    "groq": "openai/gpt-oss-120b",
}
DEFAULT_EMBEDDING_MODELS = {
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
    "openai": "text-embedding-3-small",
}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be an integer, got {raw!r}.") from exc
    if value < minimum:
        raise ConfigError(f"Environment variable {name} must be >= {minimum}.")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"Environment variable {name} must be a number, got {raw!r}.") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Immutable application settings. Use ``dataclasses.replace`` to override."""

    # Paths
    data_dir: Path
    chroma_dir: Path
    # LLM
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    # Embeddings (fixed per deployment: changing them requires re-indexing)
    embedding_provider: str = "huggingface"
    embedding_model: str = DEFAULT_EMBEDDING_MODELS["huggingface"]
    # Credentials / endpoints (never displayed)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    # Ingestion
    max_file_mb: int = 10
    chunk_size: int = 1000
    chunk_overlap: int = 200
    enable_ocr: bool = False
    # Retrieval
    top_k: int = 4
    min_relevance: float = 0.0
    history_turns: int = 3
    # Behaviour
    persist_history: bool = True

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "chat_history.db"

    @property
    def max_file_bytes(self) -> int:
        return self.max_file_mb * 1024 * 1024

    def has_api_key(self, provider: str) -> bool:
        """Whether the given provider is usable (key present / no key needed)."""
        if provider == "openai":
            return bool(self.openai_api_key)
        if provider == "anthropic":
            return bool(self.anthropic_api_key)
        if provider == "groq":
            return bool(self.groq_api_key)
        return provider == "ollama"

    def ensure_dirs(self) -> None:
        """Create the data directories if they do not exist."""
        for path in (self.upload_dir, self.processed_dir, self.chroma_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Read settings from the environment and validate them."""
    llm_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    emb_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface").strip().lower()
    if llm_provider not in SUPPORTED_LLM_PROVIDERS:
        raise ConfigError(f"LLM_PROVIDER must be one of {SUPPORTED_LLM_PROVIDERS}, got {llm_provider!r}.")
    if emb_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise ConfigError(
            f"EMBEDDING_PROVIDER must be one of {SUPPORTED_EMBEDDING_PROVIDERS}, got {emb_provider!r}."
        )

    chunk_size = _env_int("CHUNK_SIZE", 1000, minimum=100)
    chunk_overlap = _env_int("CHUNK_OVERLAP", 200, minimum=0)
    if chunk_overlap >= chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    settings = Settings(
        data_dir=Path(os.getenv("DOCUMIND_DATA_DIR") or BASE_DIR / "data"),
        chroma_dir=Path(os.getenv("CHROMA_DIR") or BASE_DIR / "vectorstore" / "chroma_db"),
        llm_provider=llm_provider,
        llm_model=os.getenv("LLM_MODEL") or DEFAULT_LLM_MODELS[llm_provider],
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.0),
        embedding_provider=emb_provider,
        embedding_model=os.getenv("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODELS[emb_provider],
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        max_file_mb=_env_int("MAX_FILE_MB", 10),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        enable_ocr=_env_bool("ENABLE_OCR", False),
        top_k=_env_int("TOP_K", 4),
        min_relevance=_env_float("MIN_RELEVANCE", 0.0),
        history_turns=_env_int("HISTORY_TURNS", 3, minimum=0),
        persist_history=_env_bool("PERSIST_HISTORY", True),
    )
    settings.ensure_dirs()
    return settings
