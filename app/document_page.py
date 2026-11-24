"""My Documents: list, select-for-chat and delete."""
import html
import logging

import streamlit as st

from app.state import get_document_manager, go_to, questions_asked

logger = logging.getLogger(__name__)


def render_documents() -> None:
    st.markdown("## 📁 My Documents")
    dm = get_document_manager()
    docs = dm.list_documents()
    stats = dm.stats()

    for col, (value, label) in zip(
        st.columns(4, gap="medium"),
        [(stats["documents"], "Documents"), (stats["pages"], "Pages processed"), (stats["chunks"], "Text chunks"), (questions_asked(), "Questions asked")],
    ):
        col.markdown(f"<div class='stat'><div class='n'>{value}</div><div class='l'>{label}</div></div>", unsafe_allow_html=True)
    st.write("")

    with st.container(key="card_docs"):
        if not docs:
            st.markdown(
                "<div class='empty'><div class='big'>📄</div><b>No documents yet</b>Upload PDFs from the Home page to see them here.</div>",
                unsafe_allow_html=True,
            )
            st.button("Go to Home", icon=":material/home:", on_click=go_to, args=("Home",), key="docs_go_home")
            return

        for doc in docs:
            c_name, c_meta, c_chat, c_del = st.columns([5, 4, 2, 2], vertical_alignment="center")
            c_name.markdown(f"**📄 {html.escape(doc.filename)}**  \n<span class='card-sub'>Uploaded {doc.uploaded_at[:10]}</span>", unsafe_allow_html=True)
            c_meta.markdown(
                f"<span class='st ready'>✓ {doc.status.capitalize()}</span> &nbsp; {doc.pages} pages • {doc.chunks} chunks • {doc.size_label}",
                unsafe_allow_html=True,
            )
            if c_chat.button("Chat", icon=":material/chat:", key=f"chat_{doc.doc_id}"):
                st.session_state["selected_docs"] = [doc.doc_id]
                st.session_state.pop("ms_selected", None)
                go_to("Home")
                st.rerun()
            if c_del.button("Delete", icon=":material/delete:", key=f"del_{doc.doc_id}"):
                try:
                    with st.spinner("Deleting…"):
                        dm.delete_document(doc.doc_id)
                    st.session_state["selected_docs"] = [d for d in st.session_state.get("selected_docs", []) if d != doc.doc_id]
                    st.session_state.pop("ms_selected", None)
                    st.toast(f"Deleted {doc.filename}", icon="🗑️")
                    st.rerun()
                except Exception:
                    logger.exception("Delete failed")
                    st.error("Could not delete this document. Check the server logs.")
