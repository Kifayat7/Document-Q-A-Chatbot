"""Session state and service wiring for the Streamlit UI.

This is the only place where Streamlit and the backend (``src``/``database``)
meet; page modules call these helpers instead of constructing services.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from pathlib import Path

import streamlit as st

from config.settings import Settings, load_settings
from database.chat_history import ChatHistory
from src.document_manager import DocumentManager

logger = logging.getLogger(__name__)
ASSETS = Path(__file__).resolve().parent.parent / "assets"
PAGES = ["Home", "My Documents", "Chat History", "Settings", "About"]


@st.cache_resource
def _base_settings() -> Settings:
    return load_settings()


@st.cache_resource
def _history_db() -> ChatHistory:
    return ChatHistory(_base_settings().db_path)


@st.cache_resource(show_spinner="Loading the embedding model (first run may download it)…")
def _embeddings():
    from src.embeddings import get_embeddings

    return get_embeddings(_base_settings())


def get_settings() -> Settings:
    """Environment settings merged with this session's Settings-page overrides."""
    return replace(_base_settings(), **st.session_state.get("overrides", {}))


def init_session() -> None:
    """Initialise per-session defaults (idempotent)."""
    ss = st.session_state
    ss.setdefault("page", "Home")
    ss.setdefault("messages", [])
    ss.setdefault("conv_id", None)
    ss.setdefault("uploader_key", 0)
    ss.setdefault("upload_report", [])
    ss.setdefault("session_questions", 0)
    ss.setdefault("overrides", {})
    ss.setdefault("session_uuid", uuid.uuid4().hex)


def get_user_id() -> str:
    """Identify the current user.

    If Streamlit login (OIDC) is configured and the visitor is signed in, the
    verified e-mail is used so data survives across sessions. Otherwise each
    browser session gets its own random id, so sessions never see each other's data.
    """
    try:
        user = st.user
        if getattr(user, "is_logged_in", False) and user.get("email"):
            return f"auth:{user.get('email')}"
    except Exception:  # login not configured / older Streamlit
        pass
    return f"anon:{st.session_state['session_uuid']}"


def user_label() -> str:
    try:
        if getattr(st.user, "is_logged_in", False):
            return str(st.user.get("name") or st.user.get("email") or "User")
    except Exception:
        pass
    return "Guest"


def get_history() -> ChatHistory:
    return _history_db()


def get_vector_store():
    """This user's Chroma collection (created lazily; loads embeddings on first use)."""
    ss = st.session_state
    if "vector_store" not in ss:
        from src.vector_store import VectorStore

        ss["vector_store"] = VectorStore(_base_settings(), _embeddings(), get_user_id())
    return ss["vector_store"]


def get_document_manager() -> DocumentManager:
    ss = st.session_state
    if "doc_manager" not in ss:
        ss["doc_manager"] = DocumentManager(_base_settings(), get_user_id(), get_vector_store)
    return ss["doc_manager"]


def questions_asked() -> int:
    """Total questions: from the database when history is persisted, else this session."""
    if get_settings().persist_history:
        return get_history().count_questions(get_user_id())
    return st.session_state["session_questions"]


def new_conversation() -> None:
    st.session_state.update(messages=[], conv_id=None)


def go_to(page: str) -> None:
    st.session_state["page"] = page


def selected_doc_ids(valid_ids: list[str]) -> list[str]:
    """Selected document ids, silently dropping ones that were deleted."""
    selected = st.session_state.get("selected_docs")
    if selected is None:  # first visit: everything selected
        selected = list(valid_ids)
    selected = [d for d in selected if d in valid_ids]
    st.session_state["selected_docs"] = selected
    return selected
