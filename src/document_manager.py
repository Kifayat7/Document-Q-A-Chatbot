"""Per-user document lifecycle: validate -> store -> extract -> chunk -> embed -> register."""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass
from typing import Callable

from config.settings import Settings
from src.errors import DocumentLoadError, DuplicateDocumentError, ValidationError
from src.utils import human_size, safe_display_name, sha256_hex, user_key, utc_now_iso

logger = logging.getLogger(__name__)
_LOCK = threading.RLock()


@dataclass
class DocRecord:
    """Metadata about one processed document."""

    doc_id: str
    filename: str
    size_bytes: int
    pages: int
    chunks: int
    sha256: str
    status: str = "ready"
    uploaded_at: str = ""
    used_ocr: bool = False

    @property
    def size_label(self) -> str:
        return human_size(self.size_bytes)


class DocumentManager:
    """Owns one user's uploaded PDFs and their registry.

    Files live in ``data/uploads/<user_key>/`` and the registry in
    ``data/processed/<user_key>/registry.json``. ``vector_store_factory`` is
    called lazily so listing documents never loads the embedding model.
    """

    def __init__(self, settings: Settings, user_id: str, vector_store_factory: Callable[[], object]):
        self.settings = settings
        key = user_key(user_id)
        self.upload_dir = settings.upload_dir / key
        self.meta_dir = settings.processed_dir / key
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.meta_dir / "registry.json"
        self._factory = vector_store_factory

    # ---- registry -------------------------------------------------------
    def _read(self) -> list[dict]:
        if not self.registry_path.exists():
            return []
        try:
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.error("Registry %s is unreadable; treating as empty.", self.registry_path)
            return []

    def _write(self, rows: list[dict]) -> None:
        tmp = self.registry_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        os.replace(tmp, self.registry_path)  # atomic on POSIX and Windows

    def list_documents(self) -> list[DocRecord]:
        """All documents, oldest first."""
        return [DocRecord(**row) for row in self._read()]

    def get(self, doc_id: str) -> DocRecord | None:
        return next((d for d in self.list_documents() if d.doc_id == doc_id), None)

    def stats(self) -> dict:
        docs = self.list_documents()
        return {
            "documents": len(docs),
            "pages": sum(d.pages for d in docs),
            "chunks": sum(d.chunks for d in docs),
        }

    # ---- validation -----------------------------------------------------
    def validate_upload(self, filename: str, data: bytes) -> None:
        """Check extension, size and the ``%PDF-`` signature (not just the name)."""
        if not filename.lower().endswith(".pdf"):
            raise ValidationError(f"'{filename}' is not a PDF. Only .pdf files are supported.")
        if not data:
            raise ValidationError(f"'{filename}' is empty.")
        if len(data) > self.settings.max_file_bytes:
            raise ValidationError(
                f"'{filename}' is {human_size(len(data))}; the limit is {self.settings.max_file_mb} MB per file."
            )
        if b"%PDF-" not in data[:1024]:
            raise ValidationError(f"'{filename}' does not look like a valid PDF file.")

    # ---- add / delete ---------------------------------------------------
    def add_document(self, filename: str, data: bytes) -> DocRecord:
        """Validate, extract, chunk, embed and register a PDF.

        Raises ``ValidationError``, ``DuplicateDocumentError`` or ``DocumentLoadError``
        for expected problems. On any failure no file, vector or registry entry is kept.
        """
        from src.document_loader import load_pdf
        from src.text_splitter import split_pages

        name = safe_display_name(filename)
        self.validate_upload(name, data)
        digest = sha256_hex(data)

        with _LOCK:
            if any(d.sha256 == digest for d in self.list_documents()):
                raise DuplicateDocumentError(f"'{name}' has already been uploaded.")

            doc_id = uuid.uuid4().hex[:12]
            path = self.upload_dir / f"{doc_id}.pdf"
            path.write_bytes(data)
            try:
                loaded = load_pdf(path, filename=name, doc_id=doc_id, enable_ocr=self.settings.enable_ocr)
                chunks = split_pages(loaded.pages, self.settings.chunk_size, self.settings.chunk_overlap)
                if not chunks:
                    raise DocumentLoadError(f"No text could be extracted from '{name}'.")
                self._factory().add_chunks(chunks, doc_id)
            except Exception:
                path.unlink(missing_ok=True)
                try:  # remove partially stored vectors, best effort
                    self._factory().delete_document(doc_id)
                except Exception:  # pragma: no cover
                    logger.exception("Cleanup of partial vectors failed for %s", doc_id)
                raise

            record = DocRecord(
                doc_id=doc_id,
                filename=name,
                size_bytes=len(data),
                pages=loaded.page_count,
                chunks=len(chunks),
                sha256=digest,
                uploaded_at=utc_now_iso(),
                used_ocr=loaded.used_ocr,
            )
            rows = self._read()
            rows.append(asdict(record))
            self._write(rows)
            return record

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document's vectors, file and registry entry."""
        with _LOCK:
            rows = self._read()
            if not any(r["doc_id"] == doc_id for r in rows):
                return False
            self._factory().delete_document(doc_id)
            (self.upload_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
            self._write([r for r in rows if r["doc_id"] != doc_id])
            return True

    def delete_all(self) -> None:
        """Remove every document, vector and registry entry of this user."""
        with _LOCK:
            if self._read():
                self._factory().delete_all()
            for pdf in self.upload_dir.glob("*.pdf"):
                pdf.unlink(missing_ok=True)
            self._write([])
