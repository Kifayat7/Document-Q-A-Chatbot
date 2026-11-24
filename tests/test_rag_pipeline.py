
"""RAG pipeline tests.

PipelineLogicTests are pure Python.
IntegrationTests use real LangChain + Chroma with fake embeddings
and a fake LLM (no network).
"""

import tempfile
import unittest
import io
from pathlib import Path

import pytest

from src.errors import ValidationError
from src.prompt_template import (
    NOT_FOUND_MESSAGE,
    SYSTEM_PROMPT,
    build_messages,
)
from src.rag_pipeline import RAGPipeline
from tests.test_retriever import FakeStore, doc


try:
    import chromadb
    import langchain_chroma
    import langchain_community
    import langchain_text_splitters

    from langchain_core.embeddings import DeterministicFakeEmbedding
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    HAVE_LANGCHAIN = True

except ImportError:
    HAVE_LANGCHAIN = False


class FakeLLM:

    def __init__(self, reply):
        self.reply = reply
        self.last_messages = None
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        self.last_messages = messages

        return type(
            "Msg",
            (),
            {"content": self.reply}
        )()


def make_store():

    return FakeStore([
        (
            doc(
                "Machine Learning lets computers learn from data.",
                "d1",
                "AI.pdf",
                3
            ),
            0.9
        ),
        (
            doc(
                "AI offers efficiency and better decisions.",
                "d1",
                "AI.pdf",
                5
            ),
            0.8
        ),
    ])


class PipelineLogicTests(unittest.TestCase):

    def test_answer_has_sources_from_cited_chunks_only(self):

        llm = FakeLLM(
            "Machine Learning is learning from data [1]."
        )

        ans = RAGPipeline(
            make_store(),
            llm
        ).ask("What is ML?", ["d1"])

        self.assertTrue(ans.found)

        self.assertEqual(
            [(s.filename, s.page) for s in ans.sources],
            [("AI.pdf", 3)]
        )

        self.assertIn(
            "learn from data",
            ans.sources[0].excerpt
        )

    def test_uncited_answer_falls_back_to_all_retrieved_sources(self):

        ans = RAGPipeline(
            make_store(),
            FakeLLM("Some answer.")
        ).ask("q", ["d1"])

        self.assertEqual(len(ans.sources), 2)

    def test_no_chunks_means_not_found_and_no_llm_call(self):

        llm = FakeLLM("should not be used")

        ans = RAGPipeline(
            FakeStore([]),
            llm
        ).ask("q", ["d1"])

        self.assertFalse(ans.found)
        self.assertEqual(ans.text, NOT_FOUND_MESSAGE)
        self.assertEqual(llm.calls, 0)

    def test_llm_not_found_reply_hides_sources(self):

        ans = RAGPipeline(
            make_store(),
            FakeLLM(NOT_FOUND_MESSAGE)
        ).ask("q", ["d1"])

        self.assertFalse(ans.found)
        self.assertEqual(ans.sources, [])

    def test_requires_question_and_documents(self):

        pipe = RAGPipeline(
            make_store(),
            FakeLLM("x")
        )

        with self.assertRaises(ValidationError):
            pipe.ask("   ", ["d1"])

        with self.assertRaises(ValidationError):
            pipe.ask("q", [])

    def test_list_content_blocks_are_flattened(self):

        llm = FakeLLM([
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world [1]"}
        ])

        ans = RAGPipeline(
            make_store(),
            llm
        ).ask("q", ["d1"])

        self.assertEqual(
            ans.text,
            "Hello world [1]"
        )

    def test_prompt_contains_rules_numbered_context_and_limited_history(self):

        chunks = [
            type(
                "C",
                (),
                {
                    "filename": "AI.pdf",
                    "page": 3,
                    "text": "ctx"
                }
            )()
        ]

        history = [
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"m{i}"
            }
            for i in range(10)
        ]

        msgs = build_messages(
            "What now?",
            chunks,
            history,
            max_turns=2
        )

        self.assertEqual(
            msgs[0]["content"],
            SYSTEM_PROMPT
        )

        self.assertIn("ONLY", SYSTEM_PROMPT)

        self.assertEqual(
            len(msgs),
            1 + 4 + 1
        )

        self.assertIn(
            "[1] File: AI.pdf | Page: 3",
            msgs[-1]["content"]
        )

        self.assertEqual(
            build_messages(
                "q",
                chunks,
                history,
                max_turns=0
            )[1]["role"],
            "user"
        )


@unittest.skipUnless(
    HAVE_LANGCHAIN,
    "LangChain/Chroma not installed"
)
class IntegrationTests(unittest.TestCase):

    SAMPLES = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "samples"
    )

    def setUp(self):

        from config.settings import Settings
        from src.document_manager import DocumentManager
        from src.vector_store import VectorStore

        self.Settings = Settings
        self.DocumentManager = DocumentManager
        self.VectorStore = VectorStore

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)

        self.settings = Settings(
            data_dir=root / "data",
            chroma_dir=root / "chroma",
            embedding_model="fake-test-model"
        )

        self.settings.ensure_dirs()

        self.emb = DeterministicFakeEmbedding(size=64)

        self.stores = []

        def make(uid):

            vs = VectorStore(
                self.settings,
                self.emb,
                uid
            )

            self.stores.append(vs)

            dm = DocumentManager(
                self.settings,
                uid,
                lambda: vs
            )

            return vs, dm

        self.make = make

    def tearDown(self):

        for vs in self.stores:

            try:
                if hasattr(vs, "close"):
                    vs.close()

            except Exception:
                pass

        self.stores.clear()

        self.tmp.cleanup()

    def _bytes(self, name):

        return (
            self.SAMPLES / name
        ).read_bytes()

    def test_end_to_end_ingest_ask_cite_delete(self):

        vs, dm = self.make("alice")

        rec = dm.add_document(
            "Artificial_Intelligence_Basics.pdf",
            self._bytes("Artificial_Intelligence_Basics.pdf")
        )

        self.assertEqual(rec.pages, 4)
        self.assertGreater(vs.count(), 0)

        pipe = RAGPipeline(
            vs,
            FakeListChatModel(
                responses=["ML learns from data [1]."]
            ),
            top_k=3
        )

        ans = pipe.ask(
            "What is machine learning?",
            [rec.doc_id]
        )

        self.assertTrue(ans.found)

        self.assertEqual(
            ans.sources[0].filename,
            "Artificial_Intelligence_Basics.pdf"
        )

        self.assertTrue(
            1 <= ans.sources[0].page <= 4
        )

        self.assertTrue(
            dm.delete_document(rec.doc_id)
        )

        self.assertEqual(vs.count(), 0)
        self.assertEqual(dm.list_documents(), [])

    def test_users_are_isolated(self):

        vs_a, dm_a = self.make("alice")
        vs_b, dm_b = self.make("bob")

        rec = dm_a.add_document(
            "ai.pdf",
            self._bytes("Artificial_Intelligence_Basics.pdf")
        )

        self.assertEqual(
            dm_b.list_documents(),
            []
        )

        self.assertEqual(
            vs_b.search(
                "machine learning",
                k=4,
                doc_ids=[rec.doc_id]
            ),
            []
        )

    def test_duplicate_and_scanned_pdfs_are_rejected_cleanly(self):

        from pypdf import PdfWriter
        from src.errors import (
            DocumentLoadError,
            DuplicateDocumentError
        )

        # Create vector store and document manager.
        vs, dm = self.make("alice")

        # Add the original PDF.
        data = self._bytes(
            "Digital_Marketing_SEO_Guide.pdf"
        )

        dm.add_document(
            "seo.pdf",
            data
        )

        # Duplicate PDF should be rejected.
        with self.assertRaises(DuplicateDocumentError):

            dm.add_document(
                "seo-copy.pdf",
                data
            )

        # Create a blank PDF to simulate a scanned PDF
        # without extractable text.
        blank = PdfWriter()

        blank.add_blank_page(
            width=200,
            height=200
        )

        buf = io.BytesIO()
        blank.write(buf)

        # Scanned/empty PDF should be rejected.
        with self.assertRaises(DocumentLoadError):

            dm.add_document(
                "scan.pdf",
                buf.getvalue()
            )

        # Only the original PDF should remain.
        self.assertEqual(
            len(dm.list_documents()),
            1
        )

        self.assertEqual(
            len(list(dm.upload_dir.glob("*.pdf"))),
            1
        )
