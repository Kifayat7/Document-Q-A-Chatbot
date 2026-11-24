"""Retriever tests (pure Python: no LangChain/Chroma needed)."""
import unittest
from dataclasses import dataclass, field

from src.retriever import retrieve


@dataclass
class FakeDoc:
    page_content: str
    metadata: dict = field(default_factory=dict)


def doc(text, doc_id, filename, page):
    return FakeDoc(text, {"doc_id": doc_id, "filename": filename, "page": page})


class FakeStore:
    """Returns canned (doc, score) pairs, honouring k and doc_ids like the real store."""

    def __init__(self, items):
        self.items = items
        self.calls = []

    def search(self, query, k=4, doc_ids=None):
        self.calls.append((k, doc_ids))
        pool = [(d, s) for d, s in self.items if doc_ids is None or d.metadata["doc_id"] in doc_ids]
        return sorted(pool, key=lambda p: p[1], reverse=True)[:k]


class RetrieverTests(unittest.TestCase):
    def test_returns_sorted_chunks_with_citation_metadata(self):
        store = FakeStore([(doc("a", "d1", "ai.pdf", 3), 0.5), (doc("b", "d1", "ai.pdf", 5), 0.9)])
        hits = retrieve(store, "q", k=4, doc_ids=["d1"])
        self.assertEqual([h.page for h in hits], [5, 3])
        self.assertEqual(hits[0].filename, "ai.pdf")
        self.assertEqual(hits[0].doc_id, "d1")

    def test_min_score_filters_weak_hits(self):
        store = FakeStore([(doc("a", "d1", "x.pdf", 1), 0.1), (doc("b", "d1", "x.pdf", 2), 0.8)])
        hits = retrieve(store, "q", doc_ids=["d1"], min_score=0.5)
        self.assertEqual([h.page for h in hits], [2])

    def test_empty_doc_selection_returns_nothing(self):
        store = FakeStore([(doc("a", "d1", "x.pdf", 1), 0.9)])
        self.assertEqual(retrieve(store, "q", doc_ids=[]), [])

    def test_only_selected_documents_are_searched(self):
        store = FakeStore([(doc("a", "d1", "x.pdf", 1), 0.9), (doc("b", "d2", "y.pdf", 1), 0.95)])
        hits = retrieve(store, "q", doc_ids=["d1"])
        self.assertEqual({h.doc_id for h in hits}, {"d1"})

    def test_multi_doc_adds_best_chunk_of_close_missing_document(self):
        # d1 fills the global top-k, but d2 has a nearly-as-good chunk.
        items = [(doc(f"a{i}", "d1", "a.pdf", i), 0.90 - i * 0.01) for i in range(4)]
        items.append((doc("b", "d2", "b.pdf", 7), 0.82))
        hits = retrieve(FakeStore(items), "q", k=4, doc_ids=["d1", "d2"])
        self.assertIn("d2", {h.doc_id for h in hits})

    def test_multi_doc_skips_irrelevant_document(self):
        items = [(doc(f"a{i}", "d1", "a.pdf", i), 0.90 - i * 0.01) for i in range(4)]
        items.append((doc("b", "d2", "b.pdf", 7), 0.20))
        hits = retrieve(FakeStore(items), "q", k=4, doc_ids=["d1", "d2"])
        self.assertNotIn("d2", {h.doc_id for h in hits})


if __name__ == "__main__":
    unittest.main()
