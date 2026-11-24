"""Chat card: conversation, document selector, citations and chat controls."""
from __future__ import annotations

import html
import logging
from datetime import datetime

import streamlit as st

from app.state import (
    ASSETS,
    get_history,
    get_settings,
    get_user_id,
    get_vector_store,
    new_conversation,
    selected_doc_ids,
)
from src.errors import ConfigError, DocuMindError
from src.llm import get_llm
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)
BOT_AVATAR = str(ASSETS / "bot_avatar.png")


def render_chat(dm) -> None:
    """Render the full chat card for the Home page."""
    docs = dm.list_documents()
    names = {d.doc_id: d.filename for d in docs}
    ids = list(names)
    selected = selected_doc_ids(ids)

    with st.container(key="card_chat"):
        pill = (
            f"<span class='pill-ok'>📄 {len(docs)} Document{'s' if len(docs) != 1 else ''} Loaded</span>"
            if docs
            else "<span class='pill-warn'>No documents yet</span>"
        )
        st.markdown(f"<div class='card-title'><span class='ic'>💬</span>Chat with Your Documents {pill}</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([1, 1, 2])
        c1.button("New chat", icon=":material/add:", key="new_chat", on_click=new_conversation)
        c2.button("Clear conversation", icon=":material/delete_sweep:", key="clear_chat", on_click=_clear, disabled=not st.session_state["messages"])

        if docs:
            with st.expander(f"Searching {len(selected)} of {len(docs)} documents", expanded=False):
                if "ms_selected" not in st.session_state:
                    st.session_state["ms_selected"] = list(selected)
                st.session_state["ms_selected"] = [d for d in st.session_state["ms_selected"] if d in ids]
                chosen = st.multiselect("Documents to search", ids, key="ms_selected", format_func=lambda i: names[i])
                st.session_state["selected_docs"] = chosen
                selected = chosen

        box = st.container(height=430, border=False, key="chat_box")
        with box:
            if not st.session_state["messages"]:
                _empty_state(bool(docs))
            for msg in st.session_state["messages"]:
                _render_message(msg)

        prompt = st.chat_input("Ask a question about your documents…", disabled=not docs, key="chat_input")
        if prompt:
            _handle_prompt(prompt, box, selected)


def _empty_state(has_docs: bool) -> None:
    if has_docs:
        text = "<b>Ask your first question</b>Try “What is this document about?” or “Summarise the key points.”"
        icon = "💡"
    else:
        text = "<b>Upload a PDF to get started</b>Add one or more PDFs on the left, then press “Process”."
        icon = "📄"
    st.markdown(f"<div class='empty'><div class='big'>{icon}</div>{text}</div>", unsafe_allow_html=True)


def _render_message(msg: dict) -> None:
    avatar = BOT_AVATAR if msg["role"] == "assistant" else "user"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("ts"):
            st.markdown(f"<div class='msg-time'>{html.escape(msg['ts'])}</div>", unsafe_allow_html=True)
        if msg["role"] == "assistant" and msg.get("sources"):
            _render_sources(msg["sources"])


def _render_sources(sources: list[dict]) -> None:
    """Show filename + page for each source, with expandable excerpts."""
    seen, rows = set(), []
    for s in sources:
        key = (s["filename"], s["page"])
        if key not in seen:
            seen.add(key)
            rows.append(f"<div class='r'>📄 <span class='f'>{html.escape(s['filename'])}</span> • Page {int(s['page'])}</div>")
    st.markdown(f"<div class='source-box'><div class='t'>Sources:</div>{''.join(rows)}</div>", unsafe_allow_html=True)
    with st.expander("View source excerpts"):
        for s in sources:
            st.markdown(f"**[{s['ref']}] {s['filename']} — page {s['page']}**  (relevance {s['score']:.2f})")
            st.text(s["excerpt"])


def _handle_prompt(prompt: str, box, selected: list[str]) -> None:
    """Run the RAG pipeline for one question and persist the exchange."""
    ss, settings, user_id = st.session_state, get_settings(), get_user_id()
    ts = datetime.now().strftime("%I:%M %p").lstrip("0")
    user_msg = {"role": "user", "content": prompt, "ts": ts}

    with box:
        _render_message(user_msg)
        if not selected:
            st.warning("Select at least one document to search (see the selector above).")
            return
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            try:
                with st.spinner("Searching your documents…"):
                    pipeline = RAGPipeline(
                        get_vector_store(),
                        get_llm(settings),
                        top_k=settings.top_k,
                        min_relevance=settings.min_relevance,
                        history_turns=settings.history_turns,
                    )
                    answer = pipeline.ask(prompt, selected, history=ss["messages"])
            except ConfigError as exc:
                st.error(f"Configuration problem: {exc}")
                return
            except DocuMindError as exc:
                st.error(str(exc))
                return
            except Exception:
                logger.exception("Question answering failed")
                st.error("Something went wrong while generating the answer. Check your API key, model name and network, then try again.")
                return

    bot_msg = {"role": "assistant", "content": answer.text, "sources": [s.to_dict() for s in answer.sources], "ts": ts}
    ss["messages"] += [user_msg, bot_msg]
    ss["session_questions"] += 1
    _persist(user_id, settings, prompt, bot_msg)
    st.rerun()  # refresh stats and render the saved messages


def _persist(user_id: str, settings, prompt: str, bot_msg: dict) -> None:
    """Save the exchange when history persistence is enabled."""
    if not settings.persist_history:
        return
    try:
        history, ss = get_history(), st.session_state
        if ss["conv_id"] is None:
            ss["conv_id"] = history.create_conversation(user_id, prompt[:60])
        history.add_message(user_id, ss["conv_id"], "user", prompt)
        history.add_message(user_id, ss["conv_id"], "assistant", bot_msg["content"], bot_msg["sources"])
    except Exception:
        logger.exception("Could not save chat history")
        st.toast("Could not save this conversation to history.", icon="⚠️")


def _clear() -> None:
    """Clear the on-screen conversation and its saved messages."""
    ss = st.session_state
    if ss.get("conv_id") and get_settings().persist_history:
        try:
            get_history().clear_messages(get_user_id(), ss["conv_id"])
        except Exception:
            logger.exception("Could not clear saved messages")
    ss["messages"] = []
