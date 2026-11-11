from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .knowledge import KnowledgeBase, RetrievedChunk
from .retrieval import heuristic_rerank
from .storage import TicketStore


TOOL_MANIFESTS = (
    {
        "name": "knowledge_search",
        "description": "检索企业知识库并返回经过 Rerank 的证据段落。",
        "input": {
            "query": "string",
            "category": "string | null",
            "candidate_k": "integer",
            "limit": "integer",
        },
        "side_effect": False,
    },
    {
        "name": "get_ticket",
        "description": "按工单编号查询当前工单状态和处理信息。",
        "input": {"ticket_id": "string"},
        "side_effect": False,
    },
    {
        "name": "create_ticket",
        "description": "为已生成答复的请求创建并持久化工单。",
        "input": {"ticket_payload": "object"},
        "side_effect": True,
    },
    {
        "name": "handoff_ticket",
        "description": "创建待人工或已升级工单，并记录负责团队。",
        "input": {"ticket_payload": "object"},
        "side_effect": True,
    },
)


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    status: str
    arguments: dict[str, Any]
    output: dict[str, Any]
    duration_ms: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "arguments": self.arguments,
            "output": self.output,
            "duration_ms": self.duration_ms,
        }


class AgentToolbox:
    def __init__(self, kb: KnowledgeBase, store: TicketStore):
        self.kb = kb
        self.store = store

    @staticmethod
    def manifests() -> list[dict[str, Any]]:
        return [dict(item) for item in TOOL_MANIFESTS]

    def search_knowledge(
        self,
        query: str,
        category: str | None,
        candidate_k: int = 12,
        limit: int = 4,
    ) -> tuple[list[RetrievedChunk], ToolCallRecord]:
        started = time.perf_counter()
        candidates = self.kb.search(query, category=category, top_k=candidate_k)
        matches = heuristic_rerank(query, candidates, limit=limit)
        duration = round((time.perf_counter() - started) * 1000, 2)
        record = ToolCallRecord(
            name="knowledge_search",
            status="ok",
            arguments={
                "query": query,
                "category": category,
                "candidate_k": candidate_k,
                "limit": limit,
            },
            output={
                "candidate_count": len(candidates),
                "selected": [
                    {
                        "chunk_id": item.chunk.chunk_id,
                        "source": item.chunk.source,
                        "relevance": item.relevance,
                    }
                    for item in matches
                ],
            },
            duration_ms=duration,
        )
        return matches, record

    def get_ticket(
        self,
        ticket_id: str,
    ) -> tuple[dict[str, Any] | None, ToolCallRecord]:
        started = time.perf_counter()
        ticket = self.store.get_ticket(ticket_id)
        record = ToolCallRecord(
            name="get_ticket",
            status="ok" if ticket else "not_found",
            arguments={"ticket_id": ticket_id},
            output={
                "found": ticket is not None,
                "status": ticket["status"] if ticket else None,
            },
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return ticket, record

    def create_ticket(
        self,
        payload: dict[str, Any],
        handoff: bool = False,
    ) -> tuple[dict[str, Any], ToolCallRecord]:
        started = time.perf_counter()
        name = "handoff_ticket" if handoff else "create_ticket"
        ticket = self.store.create_ticket(payload)
        record = ToolCallRecord(
            name=name,
            status="ok",
            arguments={
                "category": payload.get("category"),
                "priority": payload.get("priority"),
                "status": payload.get("status"),
            },
            output={
                "ticket_id": ticket["id"],
                "status": ticket["status"],
                "assigned_team": ticket["assigned_team"],
            },
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return ticket, record
