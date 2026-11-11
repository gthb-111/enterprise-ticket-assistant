from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .classifier import TEAM_BY_CATEGORY, Classification, classify
from .config import Settings
from .knowledge import KnowledgeBase, RetrievedChunk, best_sentences
from .llm import polish_answer
from .pii import mask_pii
from .retrieval import build_contextual_query, heuristic_rerank, rewrite_query
from .storage import TicketStore


@dataclass
class AssistResult:
    ticket: dict[str, Any]
    decision: str
    explanation: str
    retrieval: dict[str, Any]


class AgentState(TypedDict, total=False):
    message: str
    customer_name: str
    customer_tier: str
    order_id: str
    history: list[dict[str, Any]]
    masked_text: str
    classification: Classification
    rewritten: Any
    contextual_query: str
    memory_used: bool
    candidates: list[RetrievedChunk]
    matches: list[RetrievedChunk]
    relevant: list[RetrievedChunk]
    top_relevance: float
    citations: list[dict[str, Any]]
    answer: str
    generation_mode: str
    decision: str
    explanation: str
    ticket: dict[str, Any]
    result: AssistResult
    trace: list[dict[str, Any]]


class TriageEngine:
    def __init__(self, kb: KnowledgeBase, store: TicketStore, settings: Settings):
        self.kb = kb
        self.store = store
        self.settings = settings
        self.workflow_nodes = (
            "prepare",
            "knowledge_search",
            "compose",
            "persist_ticket",
            "human_handoff",
        )
        self.graph = self._build_graph()

    def assist(
        self,
        message: str,
        customer_name: str = "匿名客户",
        customer_tier: str = "standard",
        order_id: str = "",
        history: list[dict[str, Any]] | None = None,
    ) -> AssistResult:
        final_state = self.graph.invoke(
            self._initial_state(
                message,
                customer_name,
                customer_tier,
                order_id,
                history,
            )
        )
        return final_state["result"]

    def stream_assist(
        self,
        message: str,
        customer_name: str = "匿名客户",
        customer_tier: str = "standard",
        order_id: str = "",
        history: list[dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        final_result: AssistResult | None = None
        for update in self.graph.stream(
            self._initial_state(
                message,
                customer_name,
                customer_tier,
                order_id,
                history,
            ),
            stream_mode="updates",
        ):
            for node, node_update in update.items():
                if not isinstance(node_update, dict):
                    continue
                trace = node_update.get("trace", [])
                detail = trace[-1]["detail"] if trace else ""
                yield {
                    "event": "node",
                    "data": {"node": node, "status": "ok", "detail": detail},
                }
                if node_update.get("result") is not None:
                    final_result = node_update["result"]
        if final_result is None:
            raise RuntimeError("Agent workflow did not produce a result")
        answer = final_result.ticket["answer"]
        for start in range(0, len(answer), 28):
            yield {
                "event": "answer_delta",
                "data": {"delta": answer[start : start + 28]},
            }
        yield {
            "event": "final",
            "data": {
                "ticket": final_result.ticket,
                "decision": final_result.decision,
                "explanation": final_result.explanation,
                "retrieval": final_result.retrieval,
            },
        }

    @staticmethod
    def _initial_state(
        message: str,
        customer_name: str,
        customer_tier: str,
        order_id: str,
        history: list[dict[str, Any]] | None,
    ) -> AgentState:
        return {
            "message": message,
            "customer_name": customer_name,
            "customer_tier": customer_tier,
            "order_id": order_id,
            "history": history or [],
            "trace": [],
        }

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("prepare", self._prepare_node)
        builder.add_node("knowledge_search", self._retrieve_node)
        builder.add_node("compose", self._compose_node)
        builder.add_node("persist_ticket", self._auto_persist_node)
        builder.add_node("human_handoff", self._human_handoff_node)
        builder.add_edge(START, "prepare")
        builder.add_edge("prepare", "knowledge_search")
        builder.add_edge("knowledge_search", "compose")
        builder.add_conditional_edges(
            "compose",
            self._route_after_compose,
            {
                "auto": "persist_ticket",
                "human": "human_handoff",
            },
        )
        builder.add_edge("persist_ticket", END)
        builder.add_edge("human_handoff", END)
        return builder.compile()

    def _prepare_node(self, state: AgentState) -> AgentState:
        message = state.get("message", "").strip()
        if not message:
            raise ValueError("Message cannot be empty")
        if len(message) > 4000:
            raise ValueError("Message exceeds 4000 characters")
        customer_tier = state.get("customer_tier", "standard")
        if customer_tier not in {"standard", "vip", "enterprise"}:
            raise ValueError("Unsupported customer tier")
        masked = mask_pii(message)
        classification = classify(masked.text, customer_tier)
        return {
            "message": message,
            "masked_text": masked.text,
            "classification": classification,
            "trace": self._append_trace(
                state,
                "prepare",
                f"意图={classification.category_label}，"
                f"优先级={classification.priority}，"
                f"置信度={classification.confidence:.3f}",
            ),
        }

    def _retrieve_node(self, state: AgentState) -> AgentState:
        classification = state["classification"]
        masked_text = state["masked_text"]
        contextual_query, memory_used = build_contextual_query(
            masked_text,
            state.get("history", []),
        )
        rewritten = rewrite_query(contextual_query)
        candidates = self.kb.search(
            rewritten.rewritten_query,
            classification.category,
            top_k=12,
        )
        matches = heuristic_rerank(
            rewritten.rewritten_query,
            candidates,
            limit=4,
        )
        return {
            "rewritten": rewritten,
            "contextual_query": contextual_query,
            "memory_used": memory_used,
            "candidates": candidates,
            "matches": matches,
            "trace": self._append_trace(
                state,
                "knowledge_search",
                f"上下文={'已使用' if memory_used else '未使用'}；"
                f"改写={rewritten.rewritten_query}；"
                f"候选={len(candidates)}；重排={len(matches)}",
            ),
        }

    def _compose_node(self, state: AgentState) -> AgentState:
        masked_text = state["masked_text"]
        classification = state["classification"]
        matches = state["matches"]
        top_relevance = matches[0].relevance if matches else 0.0
        relevant = [
            match
            for match in matches
            if match.relevance >= self.settings.min_relevance_threshold
        ][:3]
        citations = self._citations(relevant)
        answer, generation_mode = self._answer(
            masked_text, relevant, classification, top_relevance
        )
        decision = self._decision(
            classification, top_relevance, bool(relevant), answer
        )
        explanation = self._explanation(
            classification, top_relevance, relevant, decision
        )
        return {
            "top_relevance": top_relevance,
            "relevant": relevant,
            "citations": citations,
            "answer": answer,
            "generation_mode": generation_mode,
            "decision": decision,
            "explanation": explanation,
            "trace": self._append_trace(
                state,
                "compose",
                f"生成模式={generation_mode}；决策={decision}",
            ),
        }

    @staticmethod
    def _route_after_compose(state: AgentState) -> str:
        return "auto" if state["decision"] == "auto_reply" else "human"

    def _auto_persist_node(self, state: AgentState) -> AgentState:
        return self._persist_ticket(state, "persist_ticket")

    def _human_handoff_node(self, state: AgentState) -> AgentState:
        return self._persist_ticket(state, "human_handoff")

    def _persist_ticket(
        self,
        state: AgentState,
        workflow_node: str,
    ) -> AgentState:
        message = state["message"]
        masked_text = state["masked_text"]
        classification = state["classification"]
        top_relevance = state["top_relevance"]
        relevant = state["relevant"]
        answer = state["answer"]
        decision = state["decision"]
        status = {
            "auto_reply": "auto_resolved",
            "human_review": "pending_agent",
            "escalate": "escalated",
        }[decision]
        customer_name = state.get("customer_name", "匿名客户")
        order_id = state.get("order_id", "")
        ticket = self.store.create_ticket(
            {
                "message_original": message,
                "message_masked": masked_text,
                "customer_name": customer_name.strip()[:50] or "匿名客户",
                "customer_tier": state.get("customer_tier", "standard"),
                "order_id": order_id.strip()[:40],
                "category": classification.category,
                "category_label": classification.category_label,
                "priority": classification.priority,
                "confidence": classification.confidence,
                "relevance": top_relevance,
                "status": status,
                "assigned_team": TEAM_BY_CATEGORY[classification.category],
                "answer": answer,
                "citations": state["citations"],
                "reasons": classification.reasons,
                "generation_mode": state["generation_mode"],
                "resolution_minutes": 0 if decision == "auto_reply" else None,
            }
        )
        trace = self._append_trace(
            state,
            workflow_node,
            f"工单={ticket['id']}；状态={status}；团队={ticket['assigned_team']}",
        )
        rewritten = state["rewritten"]
        candidates = state["candidates"]
        matches = state["matches"]
        retrieval = {
            "original_query": masked_text,
            "contextual_query": state.get("contextual_query", masked_text),
            "memory_used": state.get("memory_used", False),
            "rewritten_query": rewritten.rewritten_query,
            "expansions": list(rewritten.expansions),
            "candidate_count": len(candidates),
            "selected": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "source": item.chunk.source,
                    "hybrid_relevance": item.hybrid_relevance,
                    "rerank_score": item.rerank_score,
                }
                for item in matches
            ],
            "agent": {
                "framework": "langgraph",
                "workflow_nodes": list(self.workflow_nodes),
                "trace": trace,
            },
        }
        result = AssistResult(
            ticket=ticket,
            decision=decision,
            explanation=state["explanation"],
            retrieval=retrieval,
        )
        return {"ticket": ticket, "trace": trace, "result": result}

    @staticmethod
    def _append_trace(
        state: AgentState,
        node: str,
        detail: str,
    ) -> list[dict[str, Any]]:
        return [
            *state.get("trace", []),
            {
                "node": node,
                "status": "ok",
                "detail": detail,
            },
        ]

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

