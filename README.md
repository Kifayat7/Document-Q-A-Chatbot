# DocuMind – Document Q&A Chatbot

Upload one or more PDFs and ask questions about them. Answers come from the documents (RAG) and link back to the **file and page** they were found on.

**Pipeline:** PDF upload → text extraction (`PyPDFLoader`) → chunking (`RecursiveCharacterTextSplitter`) → embeddings → ChromaDB → question → semantic retrieval → LLM → answer with citations.

## Features
- Multiple PDFs, drag & drop, per-file status (ready / failed / skipped duplicate), size/type/signature validation
- Filename + page stored on every chunk; citations show file, page and an excerpt
- Select which documents to search; multi-document answers cite every relevant PDF
- "Not found in the documents" behaviour instead of guessing
- My Documents (stats, chat-with-this-doc, delete), Chat History (reopen/delete), Settings, About
- Configurable LLM (OpenAI, Anthropic, local Ollama) and embeddings (local Hugging Face or OpenAI)
- Per-session isolation: one Chroma collection, upload folder, registry and history scope per user
- Optional OCR for scanned PDFs (off by default)

## Project layout
```
app.py                 Streamlit entry point (routing + CSS)
app/                   UI pages (state.py wires UI to backend)
src/                   loader, splitter, embeddings, vector store, retriever, prompts, LLM, RAG pipeline, document manager
database/chat_history.py   SQLite chat history
config/settings.py     environment-driven settings
tests/                 unit + integration tests
data/samples/          two original sample PDFs (+ generator script)
assets/                logo, avatar, style.css
```

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then add OPENAI_API_KEY (or set LLM_PROVIDER=ollama)
streamlit run app.py
```
The first upload downloads the embedding model (~90 MB) once. Then upload `data/samples/*.pdf` and ask e.g. *"What is machine learning?"* or *"What is link building?"*.

**Fully local:** `LLM_PROVIDER=ollama`, `LLM_MODEL=llama3.1` (after `ollama pull llama3.1`), `EMBEDDING_PROVIDER=huggingface`.

## Run the tests
```bash
python -m pytest -v        # or: python -m unittest discover -s tests -t . -v
```
Tests that need LangChain/Chroma use fake embeddings and a fake LLM (no network, no API key). Without those packages installed they are skipped rather than failing.

## Push to GitHub
```bash
git init && git add . && git status    # confirm .env, data/uploads, vectorstore are NOT listed
git commit -m "Initial commit: DocuMind"
git branch -M main
git remote add origin https://github.com/<you>/DocuMind.git
git push -u origin main
```
If you ever committed `.env` by mistake, **rotate the key immediately**; deleting the file does not remove it from history.

## Deploy on Streamlit Community Cloud
1. Push the repo to GitHub, then create an app at share.streamlit.io pointing at `app.py`.
2. In *Advanced settings → Secrets* add, for example:
   ```toml
   OPENAI_API_KEY = "sk-..."
   LLM_PROVIDER = "openai"
   ```
   (Top-level secrets are also exposed as environment variables, which is what `config/settings.py` reads.)
3. Things to check on a hosted deploy:
   - The disk is **ephemeral**: uploads, vectors and history are lost when the app restarts.
   - Local `sentence-transformers` makes the build heavy; consider `EMBEDDING_PROVIDER=openai` on small instances.
   - ChromaDB needs a recent SQLite; if the platform's SQLite is too old, add `pysqlite3-binary` and the usual `sys.modules["sqlite3"]` swap at the top of `app.py`.
   - Ollama is not available on Community Cloud.

## Security and privacy notes
- API keys are read from the environment only and are never rendered in the UI.
- Uploads are checked for extension, size and the `%PDF-` signature; filenames are sanitised and files are stored under random ids.
- Without login, each browser session gets a random id and sees only its own data. **A page refresh starts a new session**, leaving the old session's files on disk until deleted. For real multi-user use, enable Streamlit login (OIDC, `st.login`): a signed-in user is keyed by verified e-mail and data persists. Also add a cleanup job for abandoned sessions.
- Hosted LLM providers receive the retrieved passages. Use Ollama to keep everything local.

## Known limitations
- Scanned PDFs need `ENABLE_OCR=true`, `pytesseract`, `pypdfium2` and the Tesseract binary.
- Changing the embedding model requires re-uploading documents (collections are keyed by embedding model).
- Follow-up questions use recent turns only to understand references; retrieval searches with the latest question as typed.
- Answers can be wrong. Check the cited pages for anything important.
