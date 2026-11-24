"""SQLite chat-history store. Every query is scoped to the (hashed) user."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

from src.utils import user_key, utc_now_iso

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_key TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_key TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_key, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id, id);
"""


class ChatHistory:
    """Create, read and delete conversations for individual users."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        with closing(sqlite3.connect(self.db_path, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def create_conversation(self, user_id: str, title: str = "New conversation") -> str:
        conv_id, now = uuid.uuid4().hex, utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, user_key, title, created_at, updated_at) VALUES (?,?,?,?,?)",
                (conv_id, user_key(user_id), title[:80], now, now),
            )
        return conv_id

    def _owns(self, conn, user_id: str, conv_id: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE id=? AND user_key=?", (conv_id, user_key(user_id))
        ).fetchone()
        return row is not None

    def add_message(self, user_id: str, conv_id: str, role: str, content: str, sources: list[dict] | None = None) -> bool:
        """Append a message. Returns False if the conversation is not this user's."""
        with self._connect() as conn:
            if not self._owns(conn, user_id, conv_id):
                return False
            now = utc_now_iso()
            conn.execute(
                "INSERT INTO messages (conversation_id, user_key, role, content, sources, created_at) VALUES (?,?,?,?,?,?)",
                (conv_id, user_key(user_id), role, content, json.dumps(sources or []), now),
            )
            conn.execute("UPDATE conversations SET updated_at=? WHERE id=?", (now, conv_id))
            return True

    def get_messages(self, user_id: str, conv_id: str) -> list[dict]:
        with self._connect() as conn:
            if not self._owns(conn, user_id, conv_id):
                return []
            rows = conn.execute(
                "SELECT role, content, sources, created_at FROM messages WHERE conversation_id=? ORDER BY id",
                (conv_id,),
            ).fetchall()
        return [
            {"role": r["role"], "content": r["content"], "sources": json.loads(r["sources"] or "[]"), "created_at": r["created_at"]}
            for r in rows
        ]

    def list_conversations(self, user_id: str) -> list[dict]:
        """Most recently updated first, with a message count."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT c.id, c.title, c.created_at, c.updated_at,
                          (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
                   FROM conversations c WHERE c.user_key=? ORDER BY c.updated_at DESC""",
                (user_key(user_id),),
            ).fetchall()
        return [dict(r) for r in rows]

    def rename_conversation(self, user_id: str, conv_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE conversations SET title=? WHERE id=? AND user_key=?", (title[:80], conv_id, user_key(user_id)))

    def clear_messages(self, user_id: str, conv_id: str) -> None:
        """Delete all messages but keep the (now empty) conversation."""
        with self._connect() as conn:
            if self._owns(conn, user_id, conv_id):
                conn.execute("DELETE FROM messages WHERE conversation_id=?", (conv_id,))

    def delete_conversation(self, user_id: str, conv_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM conversations WHERE id=? AND user_key=?", (conv_id, user_key(user_id)))
            return cur.rowcount > 0

    def delete_all(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE user_key=?", (user_key(user_id),))

    def count_questions(self, user_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM messages WHERE user_key=? AND role='user'", (user_key(user_id),)).fetchone()
        return int(row["n"])
