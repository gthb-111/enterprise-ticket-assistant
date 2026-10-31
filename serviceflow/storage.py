from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    message_original TEXT NOT NULL,
    message_masked TEXT NOT NULL,
    customer_name TEXT NOT NULL,
    customer_tier TEXT NOT NULL,
    order_id TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL,
    category_label TEXT NOT NULL,
    priority TEXT NOT NULL,
    confidence REAL NOT NULL,
    relevance REAL NOT NULL,
    status TEXT NOT NULL,
    assigned_team TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations_json TEXT NOT NULL,
    reasons_json TEXT NOT NULL,
    generation_mode TEXT NOT NULL,
    resolution_minutes INTEGER
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    rating TEXT NOT NULL,
    comment TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(ticket_id) REFERENCES tickets(id)
);

CREATE INDEX IF NOT EXISTS idx_tickets_created_at ON tickets(created_at);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);
"""


class TicketStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_ticket(self, payload: dict[str, Any]) -> dict[str, Any]:
        ticket_id = f"TK-{datetime.now(timezone.utc):%y%m%d}-{uuid.uuid4().hex[:6].upper()}"
        record = {
            "id": ticket_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **payload,
            "citations_json": json.dumps(payload.get("citations", []), ensure_ascii=False),
            "reasons_json": json.dumps(payload.get("reasons", []), ensure_ascii=False),
        }
        columns = [
            "id",
            "created_at",
            "message_original",
            "message_masked",
            "customer_name",
            "customer_tier",
            "order_id",
            "category",
            "category_label",
            "priority",
            "confidence",
            "relevance",
            "status",
            "assigned_team",
            "answer",
            "citations_json",
            "reasons_json",
            "generation_mode",
            "resolution_minutes",
        ]
        with self._lock, self._connect() as conn:
            conn.execute(
                f"INSERT INTO tickets ({', '.join(columns)}) VALUES "
                f"({', '.join('?' for _ in columns)})",
                [record.get(column) for column in columns],
            )
        return self.get_ticket(ticket_id) or {}

    def get_ticket(self, ticket_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tickets WHERE id = ?", (ticket_id,)
            ).fetchone()
        return _deserialize(row) if row else None

    def list_tickets(
        self,
        limit: int = 30,
        status: str | None = None,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if category:
            clauses.append("category = ?")
            params.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 100)))
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM tickets {where} ORDER BY created_at DESC LIMIT ?",
                params,
            ).fetchall()
        return [_deserialize(row) for row in rows]

    def update_status(
        self, ticket_id: str, status: str, resolution_minutes: int | None = None
    ) -> dict[str, Any] | None:
        if status not in {"open", "pending_agent", "escalated", "resolved"}:
            raise ValueError("Unsupported ticket status")
        with self._lock, self._connect() as conn:
            if resolution_minutes is None:
                conn.execute(
                    "UPDATE tickets SET status = ? WHERE id = ?",
                    (status, ticket_id),
                )
            else:
                conn.execute(
                    "UPDATE tickets SET status = ?, resolution_minutes = ? WHERE id = ?",
                    (status, resolution_minutes, ticket_id),
                )
        return self.get_ticket(ticket_id)

    def add_feedback(self, ticket_id: str, rating: str, comment: str = "") -> None:
        if rating not in {"helpful", "unhelpful"}:
            raise ValueError("Rating must be helpful or unhelpful")
        if not self.get_ticket(ticket_id):
            raise KeyError("Ticket not found")
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback "
                "(ticket_id, rating, comment, created_at) VALUES (?, ?, ?, ?)",
                (
                    ticket_id,
                    rating,
                    comment[:500],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def metrics(self) -> dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
            auto_resolved = conn.execute(
                "SELECT COUNT(*) FROM tickets WHERE status = 'auto_resolved'"
            ).fetchone()[0]
            escalated = conn.execute(
                "SELECT COUNT(*) FROM tickets WHERE status IN "
                "('pending_agent', 'escalated')"
            ).fetchone()[0]
            by_category = {
                row["category_label"]: row["count"]
                for row in conn.execute(
                    "SELECT category_label, COUNT(*) AS count FROM tickets "
                    "GROUP BY category_label ORDER BY count DESC"
                )
            }
            by_priority = {
                row["priority"]: row["count"]
                for row in conn.execute(
                    "SELECT priority, COUNT(*) AS count FROM tickets "
                    "GROUP BY priority ORDER BY priority"
                )
            }
            feedback = {
                row["rating"]: row["count"]
                for row in conn.execute(
                    "SELECT rating, COUNT(*) AS count FROM feedback GROUP BY rating"
                )
            }
        positive = feedback.get("helpful", 0)
        feedback_total = positive + feedback.get("unhelpful", 0)
        return {
            "total": total,
            "auto_resolved": auto_resolved,
            "escalated": escalated,
            "auto_resolve_rate": round(auto_resolved / total, 3) if total else 0,
            "escalation_rate": round(escalated / total, 3) if total else 0,
            "helpful_rate": round(positive / feedback_total, 3) if feedback_total else None,
            "feedback_count": feedback_total,
            "by_category": by_category,
            "by_priority": by_priority,
        }


def _deserialize(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    item["citations"] = json.loads(item.pop("citations_json"))
    item["reasons"] = json.loads(item.pop("reasons_json"))
    return item

