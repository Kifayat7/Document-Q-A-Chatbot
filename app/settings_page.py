"""Settings: model choice, retrieval options, history and data deletion."""
import logging

import streamlit as st

from config.settings import DEFAULT_LLM_MODELS, SUPPORTED_LLM_PROVIDERS
from app.state import get_document_manager, get_history, get_settings, get_user_id

logger = logging.getLogger(__name__)


def render_settings() -> None:
    st.markdown("## ⚙️ Settings")
    cfg = get_settings()
    overrides = dict(st.session_state["overrides"])

    with st.container(key="card_settings_llm"):
        st.markdown("### Language model")
        provider = st.selectbox("Provider", SUPPORTED_LLM_PROVIDERS, index=SUPPORTED_LLM_PROVIDERS.index(cfg.llm_provider))
        # Reset the model name to the provider default when the provider changes.
        model_default = cfg.llm_model if provider == cfg.llm_provider else DEFAULT_LLM_MODELS[provider]
        model = st.text_input("Model name", value=model_default)
        temperature = st.slider("Temperature", 0.0, 1.0, float(cfg.llm_temperature), 0.05, help="0 keeps answers closest to the documents.")
        top_k = st.slider("Passages retrieved per question", 1, 12, cfg.top_k)
        if cfg.has_api_key(provider):
            st.success(f"{provider.capitalize()} is ready to use (credentials loaded from the server environment).", icon="✅")
        else:
            st.warning(f"No API key found for {provider}. Add it to your .env file and restart the app.", icon="⚠️")
        st.caption(f"Embeddings: {cfg.embedding_provider} / {cfg.embedding_model} (set in .env; changing them requires re-uploading documents).")

    with st.container(key="card_settings_hist"):
        st.markdown("### Conversations")
        persist = st.toggle("Save conversations to history", value=cfg.persist_history)

    if st.button("Save settings", type="primary", icon=":material/save:", key="save_settings"):
        overrides.update(llm_provider=provider, llm_model=model.strip() or DEFAULT_LLM_MODELS[provider],
                         llm_temperature=temperature, top_k=top_k, persist_history=persist)
        st.session_state["overrides"] = overrides
        st.toast("Settings saved for this session.", icon="✅")
        st.rerun()

    with st.container(key="card_settings_danger"):
        st.markdown("### Your data")
        st.caption("Permanently deletes your uploaded PDFs, their vectors and your saved conversations.")
        confirm = st.checkbox("I understand this cannot be undone", key="confirm_wipe")
        if st.button("Delete all my data", icon=":material/delete_forever:", disabled=not confirm, key="wipe"):
            try:
                get_document_manager().delete_all()
                get_history().delete_all(get_user_id())
                st.session_state.update(messages=[], conv_id=None, selected_docs=None, upload_report=[])
                st.session_state.pop("ms_selected", None)
                st.toast("All your data was deleted.", icon="🗑️")
                st.rerun()
            except Exception:
                logger.exception("Data wipe failed")
                st.error("Could not delete everything. Check the server logs.")
