"""Semantic retrieval with multi-document coverage.

This module has no LangChain dependency: it only needs an object with a
``search(query, k, doc_ids)`` method (see ``src.vector_store.VectorStore``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk plus everything needed to cite it."""

    text: str
    filename: str
    page: int
    doc_id: str
    score: float  # relevance, higher is better


class SearchableStore(Protocol):
    def search(self, query: str, k: int = 4, doc_ids: list[str] | None = None) -> list[tuple[Any, float]]: ...


def _to_chunk(doc: Any, score: float) -> RetrievedChunk:
    meta = doc.metadata or {}
    return RetrievedChunk(
        text=doc.page_content,
        filename=str(meta.get("filename", "unknown")),
        page=int(meta.get("page", 0) or 0),
        doc_id=str(meta.get("doc_id", "")),
        score=float(score),
    )


def retrieve(
    store: SearchableStore,
    query: str,
    *,
    k: int = 4,
    doc_ids: list[str] | None = None,
    min_score: float = 0.0,
    diversify_margin: float = 0.15,
    max_docs_for_diversify: int = 8,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks for ``query``.

    When several documents are selected, a document that missed the global
    top-k still contributes its single best chunk if that chunk scores within
    ``diversify_margin`` of the best overall hit. This lets one question cite
    every relevant PDF without padding the answer with irrelevant ones.
    """
    hits = [_to_chunk(d, s) for d, s in store.search(query, k=k, doc_ids=doc_ids)]
    hits = [h for h in hits if h.score >= min_score]
    hits.sort(key=lambda h: h.score, reverse=True)

    if hits and doc_ids and 1 < len(doc_ids) <= max_docs_for_diversify:
        best = hits[0].score
        present = {h.doc_id for h in hits}
        for doc_id in doc_ids:
            if doc_id in present:
                continue
            extra = store.search(query, k=1, doc_ids=[doc_id])
            if not extra:
                continue
            candidate = _to_chunk(*extra[0])
            if candidate.score >= max(min_score, best - diversify_margin):
                hits.append(candidate)
        hits.sort(key=lambda h: h.score, reverse=True)
    return hits
