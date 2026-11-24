"""DocuMind entry point: ``streamlit run app.py``."""
import logging

import streamlit as st

st.set_page_config(page_title="DocuMind - Document Q&A", page_icon="🧠", layout="wide")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from app.about_page import render_about
from app.document_page import render_documents
from app.history_page import render_history
from app.home import render_home
from app.settings_page import render_settings
from app.sidebar import render_sidebar
from app.state import ASSETS, init_session
from src.errors import DocuMindError

st.markdown(f"<style>{(ASSETS / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

PAGES = {
    "Home": render_home,
    "My Documents": render_documents,
    "Chat History": render_history,
    "Settings": render_settings,
    "About": render_about,
}

try:
    init_session()
    render_sidebar()
    PAGES[st.session_state["page"]]()
except DocuMindError as exc:  # e.g. invalid .env values
    st.error(str(exc))
except Exception:
    logging.getLogger("documind").exception("Unhandled error")
    st.error("Something unexpected went wrong. Please refresh the page; details are in the server log.")
