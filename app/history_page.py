"""Chat History: reopen or delete saved conversations."""
import html
import logging

import streamlit as st

from app.state import get_history, get_settings, get_user_id, go_to

logger = logging.getLogger(__name__)


def _open(conv_id: str) -> None:
    messages = get_history().get_messages(get_user_id(), conv_id)
    st.session_state["messages"] = [{**m, "ts": ""} for m in messages]
    st.session_state["conv_id"] = conv_id
    go_to("Home")


def render_history() -> None:
    st.markdown("## 🕘 Chat History")
    settings, user_id, history = get_settings(), get_user_id(), get_history()
    if not settings.persist_history:
        st.info("Saving is turned off, so new conversations are not stored. Turn it on in Settings.")

    convs = history.list_conversations(user_id)
    with st.container(key="card_history"):
        if not convs:
            st.markdown(
                "<div class='empty'><div class='big'>🕘</div><b>No saved conversations</b>Your conversations will appear here after you ask a question.</div>",
                unsafe_allow_html=True,
            )
            return
        for conv in convs:
            c_title, c_open, c_del = st.columns([7, 2, 2], vertical_alignment="center")
            c_title.markdown(
                f"**{html.escape(conv['title'])}**  \n<span class='card-sub'>{conv['updated_at'][:16].replace('T', ' ')} UTC • {conv['message_count']} messages</span>",
                unsafe_allow_html=True,
            )
            c_open.button("Open", icon=":material/open_in_new:", key=f"open_{conv['id']}", on_click=_open, args=(conv["id"],))
            if c_del.button("Delete", icon=":material/delete:", key=f"hdel_{conv['id']}"):
                history.delete_conversation(user_id, conv["id"])
                if st.session_state.get("conv_id") == conv["id"]:
                    st.session_state.update(conv_id=None, messages=[])
                st.rerun()
        st.divider()
        if st.button("Delete all history", icon=":material/delete_forever:", key="hist_delete_all"):
            history.delete_all(user_id)
            st.session_state.update(conv_id=None, messages=[])
            st.rerun()
