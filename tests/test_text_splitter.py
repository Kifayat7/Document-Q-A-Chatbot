"""Splitter tests (need langchain-text-splitters; skipped otherwise)."""
import unittest

from src.errors import ValidationError

try:
    from langchain_core.documents import Document
    import langchain_text_splitters  # noqa: F401

    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False


@unittest.skipUnless(HAVE, "langchain not installed")
class SplitterTests(unittest.TestCase):
    def test_chunks_keep_metadata_and_get_unique_indices(self):
        from src.text_splitter import split_pages

        text = " ".join(f"Sentence number {i} about learning." for i in range(120))
        pages = [Document(page_content=text, metadata={"filename": "a.pdf", "doc_id": "d", "page": 2})]
        chunks = split_pages(pages, chunk_size=300, chunk_overlap=50)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(c.metadata["filename"] == "a.pdf" and c.metadata["page"] == 2 for c in chunks))
        self.assertEqual([c.metadata["chunk"] for c in chunks], list(range(len(chunks))))
        self.assertTrue(all(len(c.page_content) <= 300 for c in chunks))

    def test_invalid_overlap_is_rejected(self):
        from src.text_splitter import split_pages

        with self.assertRaises(ValidationError):
            split_pages([], chunk_size=100, chunk_overlap=100)


if __name__ == "__main__":
    unittest.main()
