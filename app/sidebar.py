"""Dark-navy sidebar with navigation."""
import streamlit as st

from app.state import ASSETS, PAGES, go_to

_ICONS = {
    "Home": ":material/home:",
    "My Documents": ":material/folder_open:",
    "Chat History": ":material/history:",
    "Settings": ":material/settings:",
    "About": ":material/info:",
}


def render_sidebar() -> None:
    with st.sidebar:
        st.image(str(ASSETS / "logo.png"), width=150)
        st.markdown(
            "<div class='brand'><h1>DocuMind</h1><p>Your AI Document<br>Assistant</p></div>",
            unsafe_allow_html=True,
        )
        for name in PAGES:
            active = st.session_state["page"] == name
            st.button(
                name,
                key=f"nav_{name}",
                icon=_ICONS[name],
                type="primary" if active else "secondary",
                on_click=go_to,
                args=(name,),
            )
        st.markdown(
            "<div class='powered'>⚡<br>Powered by<br><b class='lc'>LangChain</b> + <b class='ch'>ChromaDB</b></div>"
            "<div class='tagline'>Ask • Explore • Learn</div>",
            unsafe_allow_html=True,
        )
