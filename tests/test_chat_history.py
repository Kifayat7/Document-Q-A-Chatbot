"""Chat history persistence and per-user isolation (pure sqlite)."""
import tempfile
import unittest
from pathlib import Path

from database.chat_history import ChatHistory


class ChatHistoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = ChatHistory(Path(self.tmp.name) / "h.db")

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_with_sources(self):
        cid = self.db.create_conversation("alice", "What is ML?")
        self.db.add_message("alice", cid, "user", "What is ML?")
        src = [{"ref": 1, "filename": "AI.pdf", "page": 3, "excerpt": "x", "score": 0.9}]
        self.db.add_message("alice", cid, "assistant", "A branch of AI [1].", src)
        msgs = self.db.get_messages("alice", cid)
        self.assertEqual([m["role"] for m in msgs], ["user", "assistant"])
        self.assertEqual(msgs[1]["sources"][0]["page"], 3)
        listing = self.db.list_conversations("alice")
        self.assertEqual((listing[0]["title"], listing[0]["message_count"]), ("What is ML?", 2))
        self.assertEqual(self.db.count_questions("alice"), 1)

    def test_other_users_cannot_read_write_or_delete(self):
        cid = self.db.create_conversation("alice", "secret")
        self.db.add_message("alice", cid, "user", "hi")
        self.assertEqual(self.db.get_messages("bob", cid), [])
        self.assertFalse(self.db.add_message("bob", cid, "user", "intrusion"))
        self.assertFalse(self.db.delete_conversation("bob", cid))
        self.assertEqual(self.db.list_conversations("bob"), [])
        self.db.clear_messages("bob", cid)
        self.assertEqual(len(self.db.get_messages("alice", cid)), 1)

    def test_clear_delete_and_delete_all(self):
        c1, c2 = self.db.create_conversation("a", "one"), self.db.create_conversation("a", "two")
        self.db.add_message("a", c1, "user", "q")
        self.db.clear_messages("a", c1)
        self.assertEqual(self.db.get_messages("a", c1), [])
        self.assertEqual(len(self.db.list_conversations("a")), 2)
        self.assertTrue(self.db.delete_conversation("a", c1))
        self.db.add_message("a", c2, "user", "q")
        self.db.delete_all("a")
        self.assertEqual((self.db.list_conversations("a"), self.db.count_questions("a")), ([], 0))

    def test_raw_user_id_is_not_stored(self):
        self.db.create_conversation("alice@example.com", "t")
        self.assertNotIn(b"alice@example.com", (Path(self.tmp.name) / "h.db").read_bytes())


if __name__ == "__main__":
    unittest.main()
