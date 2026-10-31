from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .classifier import TEAM_BY_CATEGORY, Classification, classify
from .config import Settings
from .knowledge import KnowledgeBase, RetrievedChunk, best_sentences
from .llm import polish_answer
from .pii import mask_pii
from .storage import TicketStore


@dataclass
class AssistResult:
    ticket: dict[str, Any]
    decision: str
    explanation: str


class TriageEngine:
    def __init__(self, kb: KnowledgeBase, store: TicketStore, settings: Settings):
        self.kb = kb
        self.store = store
        self.settings = settings

    def assist(
        self,
        message: str,
        customer_name: str = "匿名客户",
        customer_tier: str = "standard",
        order_id: str = "",
    ) -> AssistResult:
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be empty")
        if len(message) > 4000:
            raise ValueError("Message exceeds 4000 characters")
        if customer_tier not in {"standard", "vip", "enterprise"}:
            raise ValueError("Unsupported customer tier")

        masked = mask_pii(message)
        classification = classify(masked.text, customer_tier)
        matches = self.kb.search(masked.text, classification.category, top_k=4)
        top_relevance = matches[0].relevance if matches else 0.0
        relevant = [
            match
            for match in matches
            if match.relevance >= self.settings.min_relevance_threshold
        ][:3]
        citations = self._citations(relevant)
        answer, generation_mode = self._answer(
            masked.text, relevant, classification, top_relevance
        )
        decision = self._decision(
            classification, top_relevance, bool(relevant), answer
        )
        status = {
            "auto_reply": "auto_resolved",
            "human_review": "pending_agent",
            "escalate": "escalated",
        }[decision]
        sla_minutes = {"P0": 10, "P1": 30, "P2": 120, "P3": 480}[
            classification.priority
        ]
        explanation = self._explanation(
            classification, top_relevance, relevant, decision
        )
        ticket = self.store.create_ticket(
            {
                "message_original": message,
                "message_masked": masked.text,
                "customer_name": customer_name.strip()[:50] or "匿名客户",
                "customer_tier": customer_tier,
                "order_id": order_id.strip()[:40],
                "category": classification.category,
                "category_label": classification.category_label,
                "priority": classification.priority,
                "confidence": classification.confidence,
                "relevance": top_relevance,
                "status": status,
                "assigned_team": TEAM_BY_CATEGORY[classification.category],
                "answer": answer,
                "citations": citations,
                "reasons": classification.reasons,
                "generation_mode": generation_mode,
                "resolution_minutes": 0 if decision == "auto_reply" else None,
            }
        )
        return AssistResult(ticket=ticket, decision=decision, explanation=explanation)

    def _answer(
        self,
        query: str,
        matches: list[RetrievedChunk],
        classification: Classification,
        relevance: float,
    ) -> tuple[str, str]:
        if not matches or relevance < self.settings.min_relevance_threshold:
            return (
                "当前知识库没有找到足够可靠的处理依据。已为你创建工单，"
                f"将转交{TEAM_BY_CATEGORY[classification.category]}进一步核实，"
                "请勿重复提交，以免影响处理进度。",
                "safe_fallback",
            )
        evidence = [
            {
                "id": item.chunk.chunk_id,
                "title": item.chunk.title,
                "text": item.chunk.text,
            }
            for item in matches
        ]
        llm_result = polish_answer(query, evidence, self.settings)
        if llm_result:
            return llm_result.text, llm_result.mode
        sentences = best_sentences(matches, query, limit=3)
        citation_ids = " ".join(
            f"[{item.chunk.chunk_id}]" for item in matches[:2]
        )
        answer = "根据知识库，你可以这样处理：\n\n" + "\n".join(
            f"{index + 1}. {sentence}" for index, sentence in enumerate(sentences)
        )
        if matches[0].chunk.category == classification.category:
            answer += f"\n\n依据：{citation_ids}"
        return answer, "extractive_grounded"

    def _decision(
        self,
        classification: Classification,
        relevance: float,
        has_evidence: bool,
        answer: str,
    ) -> str:
        if (
            classification.risk_flags
            or classification.priority == "P0"
            or answer.startswith("当前知识库没有找到")
        ):
            return "escalate"
        if (
            classification.requires_human
            or classification.confidence < 0.56
            or relevance < self.settings.auto_resolve_threshold
            or not has_evidence
        ):
            return "human_review"
        return "auto_reply"

    @staticmethod
    def _citations(matches: list[RetrievedChunk]) -> list[dict[str, Any]]:
        return [
            {
                "id": item.chunk.chunk_id,
                "doc_id": item.chunk.doc_id,
                "title": item.chunk.title,
                "source": item.chunk.source,
                "snippet": item.chunk.text,
                "relevance": item.relevance,
            }
            for item in matches
        ]

    @staticmethod
    def _explanation(
        classification: Classification,
        relevance: float,
        matches: list[RetrievedChunk],
        decision: str,
    ) -> str:
        decision_text = {
            "auto_reply": "自动答复",
            "human_review": "转人工复核",
            "escalate": "升级处理",
        }[decision]
        source_titles = "、".join(dict.fromkeys(item.chunk.title for item in matches))
        if source_titles:
            evidence_text = f"检索命中《{source_titles}》"
        else:
            evidence_text = "未命中可用知识"
        return (
            f"分流为{classification.category_label}，优先级"
            f"{classification.priority}；{evidence_text}，相关性 {relevance:.0%}。"
            f"策略判定：{decision_text}。"
        )

