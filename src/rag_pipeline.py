"""Retrieval-Augmented Generation: retrieve -> prompt -> LLM -> cited answer."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.errors import ValidationError
from src.prompt_template import NOT_FOUND_MESSAGE, build_messages
from src.retriever import RetrievedChunk, retrieve

_CITATION = re.compile(r"\[(\d+)\]")
_EXCERPT_CHARS = 450


@dataclass
class Source:
    """A citation shown under an answer."""

    ref: int  # matches the [n] marker in the answer
    filename: str
    page: int
    excerpt: str
    score: float
    doc_id: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Answer:
    text: str
    sources: list[Source] = field(default_factory=list)
    found: bool = True


def _coerce_text(response) -> str:
    """LLM responses may be strings, messages, or lists of content blocks."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = [p if isinstance(p, str) else p.get("text", "") for p in content if isinstance(p, (str, dict))]
        content = "".join(parts)
    return str(content).strip()


def _excerpt(text: str) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= _EXCERPT_CHARS else flat[: _EXCERPT_CHARS - 1].rstrip() + "…"


class RAGPipeline:
    """Answers questions from a user's selected documents."""

    def __init__(self, store, llm, *, top_k: int = 4, min_relevance: float = 0.0, history_turns: int = 3):
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.history_turns = history_turns

    def ask(self, question: str, doc_ids: list[str], history: list[dict] | None = None) -> Answer:
        """Answer ``question`` using only chunks from ``doc_ids``."""
        if not question or not question.strip():
            raise ValidationError("Please type a question.")
        if not doc_ids:
            raise ValidationError("Select at least one document to search.")

        chunks = retrieve(self.store, question, k=self.top_k, doc_ids=doc_ids, min_score=self.min_relevance)
        if not chunks:
            return Answer(NOT_FOUND_MESSAGE, [], found=False)

        messages = build_messages(question, chunks, history, max_turns=self.history_turns)
        text = _coerce_text(self.llm.invoke(messages))
        if not text or NOT_FOUND_MESSAGE.lower() in text.lower():
            return Answer(NOT_FOUND_MESSAGE, [], found=False)
        return Answer(text, self._sources_for(text, chunks), found=True)

    @staticmethod
    def _sources_for(answer: str, chunks: list[RetrievedChunk]) -> list[Source]:
        """Prefer the passages the model actually cited; otherwise show all retrieved."""
        cited = {int(n) for n in _CITATION.findall(answer) if 1 <= int(n) <= len(chunks)}
        indices = sorted(cited) if cited else range(1, len(chunks) + 1)
        return [
            Source(
                ref=i,
                filename=chunks[i - 1].filename,
                page=chunks[i - 1].page,
                excerpt=_excerpt(chunks[i - 1].text),
                score=round(chunks[i - 1].score, 3),
                doc_id=chunks[i - 1].doc_id,
            )
            for i in indices
        ]
