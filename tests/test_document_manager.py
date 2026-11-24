"""Document manager: validation, registry and isolation (no LangChain needed)."""
import tempfile
import unittest
from pathlib import Path

from config.settings import Settings
from src.document_manager import DocRecord, DocumentManager
from src.errors import ValidationError
from src.utils import safe_display_name


class FakeVS:
    def __init__(self):
        self.deleted = []

    def delete_document(self, doc_id):
        self.deleted.append(doc_id)

    def delete_all(self):
        self.deleted.append("*")


class DocumentManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.settings = Settings(data_dir=root / "d", chroma_dir=root / "c", max_file_mb=1)
        self.vs = FakeVS()
        self.dm = DocumentManager(self.settings, "alice", lambda: self.vs)

    def tearDown(self):
        self.tmp.cleanup()

    def test_validation_rules(self):
        ok = b"%PDF-1.4\n" + b"x" * 100
        self.dm.validate_upload("a.pdf", ok)  # no error
        with self.assertRaisesRegex(ValidationError, "not a PDF"):
            self.dm.validate_upload("a.txt", ok)
        with self.assertRaisesRegex(ValidationError, "empty"):
            self.dm.validate_upload("a.pdf", b"")
        with self.assertRaisesRegex(ValidationError, "limit"):
            self.dm.validate_upload("a.pdf", b"%PDF-" + b"x" * (1024 * 1024 + 1))
        with self.assertRaisesRegex(ValidationError, "valid PDF"):
            self.dm.validate_upload("fake.pdf", b"MZ\x90\x00 this is an exe renamed")

    def _seed(self, dm, doc_id="d1"):
        import json
        (dm.upload_dir / f"{doc_id}.pdf").write_bytes(b"%PDF-")
        rec = DocRecord(doc_id, "a.pdf", 5, 3, 7, "hash" + doc_id, uploaded_at="now")
        rows = dm._read()
        rows.append(rec.__dict__)
        dm._write(rows)

    def test_registry_stats_and_delete(self):
        self._seed(self.dm, "d1")
        self._seed(self.dm, "d2")
        self.assertEqual(self.dm.stats(), {"documents": 2, "pages": 6, "chunks": 14})
        self.assertTrue(self.dm.delete_document("d1"))
        self.assertEqual(self.vs.deleted, ["d1"])
        self.assertFalse((self.dm.upload_dir / "d1.pdf").exists())
        self.assertEqual([d.doc_id for d in self.dm.list_documents()], ["d2"])
        self.assertFalse(self.dm.delete_document("nope"))
        self.dm.delete_all()
        self.assertEqual(self.dm.list_documents(), [])

    def test_users_have_separate_registries_and_folders(self):
        self._seed(self.dm, "d1")
        bob = DocumentManager(self.settings, "bob", lambda: self.vs)
        self.assertEqual(bob.list_documents(), [])
        self.assertNotEqual(bob.upload_dir, self.dm.upload_dir)
        self.assertFalse(bob.delete_document("d1"))  # cannot delete Alice's doc

    def test_corrupt_registry_is_treated_as_empty(self):
        self.dm.registry_path.write_text("{not json")
        self.assertEqual(self.dm.list_documents(), [])

    def test_safe_display_name_strips_paths(self):
        self.assertEqual(safe_display_name("../../etc/passwd.pdf"), "passwd.pdf")
        self.assertEqual(safe_display_name("C:\\x\\y\\report.pdf"), "report.pdf")
        self.assertEqual(safe_display_name(""), "document.pdf")


if __name__ == "__main__":
    unittest.main()
