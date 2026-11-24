"""Home dashboard: hero, stats, upload card, feature list and chat."""
import html
import logging

import streamlit as st

from app import chat_interface
from app.state import get_document_manager, get_settings, questions_asked, user_label
from src.errors import DocuMindError, DuplicateDocumentError

logger = logging.getLogger(__name__)

_FEATURES = [
    ("💬", "#dbeafe", "Ask Questions", "Get accurate answers from your document"),
    ("🛡️", "#dcfce7", "Source Citations", "See references with page numbers"),
    ("📚", "#fef3c7", "Multi-Document Support", "Upload and query multiple PDFs"),
    ("⚡", "#ede9fe", "Powered by AI", "Built with LangChain & ChromaDB"),
]
_BOTTOM = [
    ("✅", "#dcfce7", "Private sessions", "Documents are isolated per session"),
    ("⚡", "#fef3c7", "Fast & Accurate", "Answers grounded in your PDFs"),
    ("🔒", "#d1fae5", "Offline-ready", "Run local models with Ollama"),
]


def render_home() -> None:
    dm = get_document_manager()
    _hero()
    _stats_row(dm)
    left, right = st.columns([5, 7], gap="large")
    with left:
        _upload_card(dm)
        _features_card()
    with right:
        chat_interface.render_chat(dm)
    cols = st.columns(3, gap="medium")
    for col, (icon, bg, title, sub) in zip(cols, _BOTTOM):
        col.markdown(
            f"<div class='bottom'><div class='fi' style='background:{bg}'>{icon}</div>"
            f"<div><b>{title}</b><span>{sub}</span></div></div>",
            unsafe_allow_html=True,
        )


def _hero() -> None:
    st.markdown(
        f"""<div class='hero'>
        <div class='who'>👤 Welcome, {html.escape(user_label())}</div>
        <span class='badge'>✨ RAG-Powered • Smart • Accurate</span>
        <h1>Document <span class='grad'>Q&amp;A</span> Chatbot</h1>
        <p>Upload your PDF and ask anything about its content!</p></div>""",
        unsafe_allow_html=True,
    )


def _stats_row(dm) -> None:
    stats = dm.stats()
    status = "Ready" if stats["documents"] else "No documents"
    items = [
        (stats["documents"], "Documents uploaded"),
        (stats["pages"], "Pages processed"),
        (questions_asked(), "Questions asked"),
        (status, "Current status"),
    ]
    for col, (value, label) in zip(st.columns(4, gap="medium"), items):
        col.markdown(f"<div class='stat'><div class='n'>{value}</div><div class='l'>{label}</div></div>", unsafe_allow_html=True)
    st.write("")


def _upload_card(dm) -> None:
    settings = get_settings()
    with st.container(key="card_upload"):
        st.markdown("<div class='card-title'><span class='ic'>⬆</span>Upload Your PDF</div>", unsafe_allow_html=True)
        st.markdown("<div class='card-sub'>Drag &amp; drop your PDF files here or click to browse</div>", unsafe_allow_html=True)
        files = st.file_uploader(
            "PDF files",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state['uploader_key']}",
            label_visibility="collapsed",
        )
        st.caption(f"Supports multiple PDFs • Max size: {settings.max_file_mb}MB each")
        if files and st.button(f"Process {len(files)} PDF{'s' if len(files) != 1 else ''}", type="primary", icon=":material/bolt:", key="process_btn"):
            _process(dm, files)
        _report()


def _process(dm, files) -> None:
    """Ingest each uploaded file, collecting a per-file status report."""
    report, new_ids = [], []
    bar = st.progress(0.0, text="Starting…")
    for i, f in enumerate(files):
        bar.progress(i / len(files), text=f"Processing {f.name}…")
        try:
            rec = dm.add_document(f.name, f.getvalue())
            new_ids.append(rec.doc_id)
            note = f"{rec.pages} pages • {rec.chunks} chunks" + (" • OCR" if rec.used_ocr else "")
            report.append(("ready", f.name, note))
        except DuplicateDocumentError as exc:
            report.append(("skipped", f.name, str(exc)))
        except DocuMindError as exc:
            report.append(("failed", f.name, str(exc)))
        except Exception:
            logger.exception("Unexpected error while processing %s", f.name)
            report.append(("failed", f.name, "Unexpected error while processing this file. Check the server logs."))
    bar.empty()
    if new_ids:
        selected = st.session_state.get("selected_docs") or []
        st.session_state["selected_docs"] = selected + new_ids
        st.session_state.pop("ms_selected", None)  # let the selector re-initialise
    st.session_state["upload_report"] = report
    st.session_state["uploader_key"] += 1  # clears the uploader so files are not re-processed
    st.rerun()


def _report() -> None:
    report = st.session_state.get("upload_report") or []
    labels = {"ready": "✓ Ready", "failed": "✗ Failed", "skipped": "• Skipped"}
    for status, name, note in report:
        st.markdown(
            f"<div class='upl-row'><span class='nm'>📄 {html.escape(name)}</span>"
            f"<span class='st {status}'>{labels[status]}</span></div>"
            f"<div class='card-sub' style='margin:.1rem .4rem .2rem'>{html.escape(note)}</div>",
            unsafe_allow_html=True,
        )


def _features_card() -> None:
    st.write("")
    with st.container(key="card_features"):
        st.markdown("<div class='card-title'>⭐ Key Features</div>", unsafe_allow_html=True)
        for icon, bg, title, sub in _FEATURES:
            st.markdown(
                f"<div class='feat'><div class='fi' style='background:{bg}'>{icon}</div><div><b>{title}</b><span>{sub}</span></div></div>",
                unsafe_allow_html=True,
            )
