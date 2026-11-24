"""Loader tests (need langchain-community + pypdf; skipped otherwise)."""
import io
import unittest
from pathlib import Path

from src.errors import DocumentLoadError

try:
    import langchain_community  # noqa: F401
    from pypdf import PdfWriter

    HAVE = True
except ImportError:  # pragma: no cover
    HAVE = False

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


@unittest.skipUnless(HAVE, "langchain-community / pypdf not installed")
class LoaderTests(unittest.TestCase):
    def test_pages_carry_filename_and_one_based_page_numbers(self):
        from src.document_loader import load_pdf

        res = load_pdf(SAMPLES / "Artificial_Intelligence_Basics.pdf", filename="ai.pdf", doc_id="abc")
        self.assertEqual(res.page_count, 4)
        self.assertEqual([p.metadata["page"] for p in res.pages], [1, 2, 3, 4])
        self.assertTrue(all(p.metadata["filename"] == "ai.pdf" and p.metadata["doc_id"] == "abc" for p in res.pages))
        self.assertNotIn("source", res.pages[0].metadata)  # no server paths leaked

    def test_scanned_pdf_without_ocr_raises_friendly_error(self):
        from src.document_loader import load_pdf

        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        path = Path(__import__("tempfile").mkdtemp()) / "scan.pdf"
        buf = io.BytesIO()
        writer.write(buf)
        path.write_bytes(buf.getvalue())
        with self.assertRaisesRegex(DocumentLoadError, "scanned"):
            load_pdf(path, filename="scan.pdf", doc_id="x")

    def test_corrupt_pdf_raises_load_error(self):
        from src.document_loader import load_pdf

        path = Path(__import__("tempfile").mkdtemp()) / "bad.pdf"
        path.write_bytes(b"%PDF-1.4 this is not really a pdf")
        with self.assertRaises(DocumentLoadError):
            load_pdf(path, filename="bad.pdf", doc_id="x")


if __name__ == "__main__":
    unittest.main()
