"""About page."""
import streamlit as st


def render_about() -> None:
    st.markdown("## ℹ️ About DocuMind")
    with st.container(key="card_about"):
        st.markdown(
            """
**DocuMind** answers questions about your own PDFs using Retrieval-Augmented Generation (RAG).
Answers are grounded in the passages it retrieves, and every answer links back to the file and page it came from.

**How it works**

1. Your PDF is read page by page and split into overlapping chunks.
2. Each chunk is turned into an embedding and stored in ChromaDB, in a collection private to your session.
3. When you ask a question, the most similar chunks are retrieved from the documents you selected.
4. The language model answers using only those chunks, or says it could not find the answer.

**Built with** Streamlit, LangChain, ChromaDB, Hugging Face sentence-transformers and a configurable LLM (OpenAI, Anthropic or local Ollama).

**Good to know**

- Scanned PDFs need OCR (optional, off by default).
- Answers can still be wrong: check the cited pages for anything important.
- With a hosted LLM, the retrieved passages are sent to that provider. Use Ollama to keep everything on your machine.
            """
        )
