"""Split page-level documents into overlapping chunks, keeping metadata."""
from __future__ import annotations

from src.errors import ValidationError


def split_pages(pages: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """Split page documents into chunks with ``RecursiveCharacterTextSplitter``.

    Each chunk inherits ``filename``, ``doc_id`` and ``page`` from its page and
    gets a running ``chunk`` index (unique per document) used to build ids.
    """
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValidationError("chunk_overlap must be >= 0 and smaller than chunk_size.")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = index
    return chunks
