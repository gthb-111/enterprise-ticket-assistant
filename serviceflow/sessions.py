from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session
ON chat_messages(session_id, id);
"""


class SessionStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create(self, title: str = "") -> dict[str, Any]:
        now = _now()
        session_id = f"S-{datetime.now(timezone.utc):%y%m%d}-{uuid.uuid4().hex[:8].upper()}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, title.strip()[:100], now, now),
            )
        return self.get(session_id) or {}

    def get(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["messages"] = self.messages(session_id)
        return item

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.*, COUNT(m.id) AS message_count
                FROM chat_sessions s
                LEFT JOIN chat_messages m ON m.session_id = s.id
                GROUP BY s.id
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def messages(
        self,
        session_id: str,
        limit: int = 30,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM (
                    SELECT * FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, max(1, min(limit, 100))),
            ).fetchall()
        return [_deserialize_message(row) for row in rows]

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("Unsupported chat role")
        if self.get(session_id) is None:
            raise KeyError("Session not found")
        created_at = _now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_messages (
                    session_id, role, content, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    role,
                    content[:8000],
                    json.dumps(metadata or {}, ensure_ascii=False),
                    created_at,
                ),
            )
            conn.execute(
                """
                UPDATE chat_sessions
                SET updated_at = ?,
                    title = CASE
                        WHEN title = '' AND ? = 'user' THEN ?
                        ELSE title
                    END
                WHERE id = ?
                """,
                (created_at, role, content[:100], session_id),
            )
        return {
            "id": cursor.lastrowid,
            "session_id": session_id,
            "role": role,
            "content": content[:8000],
            "metadata": metadata or {},
            "created_at": created_at,
        }


def _deserialize_message(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["metadata"] = json.loads(item.pop("metadata_json"))
    return item


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
