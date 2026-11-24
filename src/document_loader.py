"""PDF text extraction with page-level metadata (and optional OCR fallback)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.errors import DocumentLoadError

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    """Pages extracted from a PDF."""

    pages: list  # list[langchain_core.documents.Document], one per non-empty page
    page_count: int  # total pages in the PDF (including empty ones)
    used_ocr: bool = False


def load_pdf(
    path: str | Path,
    *,
    filename: str,
    doc_id: str,
    enable_ocr: bool = False,
    min_total_chars: int = 25,
) -> LoadResult:
    """Extract text from a PDF using LangChain's ``PyPDFLoader``.

    Every returned ``Document`` carries ``filename``, ``doc_id`` and a 1-based
    ``page`` in its metadata. The server file path is deliberately removed.

    Raises:
        DocumentLoadError: unreadable/encrypted PDF, or no extractable text
            (scanned PDF) when OCR is disabled/unavailable.
    """
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document

    try:
        raw_pages = PyPDFLoader(str(path)).load()
    except Exception as exc:  # pypdf raises many different errors
        logger.warning("Could not read %s: %s", filename, exc)
        raise DocumentLoadError(
            f"'{filename}' could not be read. It may be corrupted or password-protected."
        ) from exc

    page_count = len(raw_pages)
    if page_count == 0:
        raise DocumentLoadError(f"'{filename}' contains no pages.")

    pages = [
        Document(
            page_content=doc.page_content.replace("\x00", ""),
            metadata={"filename": filename, "doc_id": doc_id, "page": int(doc.metadata.get("page", i)) + 1},
        )
        for i, doc in enumerate(raw_pages)
    ]

    used_ocr = False
    if sum(len(p.page_content.strip()) for p in pages) < min_total_chars:
        if not enable_ocr:
            raise DocumentLoadError(
                f"'{filename}' looks like a scanned PDF (no selectable text). "
                "Set ENABLE_OCR=true and install the OCR extras to process it."
            )
        pages = _ocr_pages(path, filename, doc_id)
        used_ocr = True
        if sum(len(p.page_content.strip()) for p in pages) < min_total_chars:
            raise DocumentLoadError(f"OCR found no readable text in '{filename}'.")

    pages = [p for p in pages if p.page_content.strip()]
    return LoadResult(pages=pages, page_count=page_count, used_ocr=used_ocr)


def _ocr_pages(path: str | Path, filename: str, doc_id: str) -> list:
    """Optional OCR: render pages with pypdfium2 and read them with Tesseract.

    Requires ``pip install pytesseract`` plus the Tesseract binary on PATH.
    """
    try:
        import pypdfium2 as pdfium
        import pytesseract
        from langchain_core.documents import Document
    except ImportError as exc:
        raise DocumentLoadError(
            "OCR is enabled but 'pytesseract' / 'pypdfium2' are not installed."
        ) from exc

    pages = []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for index in range(len(pdf)):
            image = pdf[index].render(scale=2).to_pil()
            try:
                text = pytesseract.image_to_string(image)
            except Exception as exc:
                raise DocumentLoadError(
                    "Tesseract OCR failed. Is the Tesseract binary installed and on PATH?"
                ) from exc
            pages.append(Document(page_content=text, metadata={"filename": filename, "doc_id": doc_id, "page": index + 1}))
    finally:
        pdf.close()
    return pages
