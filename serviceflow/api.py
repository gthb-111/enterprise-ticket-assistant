from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import PROJECT_ROOT, Settings
from .documents import MAX_DOCUMENT_BYTES, DocumentStore, parse_document
from .engine import TriageEngine
from .knowledge import KnowledgeBase
from .llm import ModelCallError
from .retrieval import heuristic_rerank, rewrite_query
from .sessions import SessionStore
from .storage import TicketStore


class AssistRequest(BaseModel):
    message: str = Field(default="", max_length=4000)
    customer_name: str = Field(default="匿名客户", max_length=50)
    customer_tier: str = "standard"
    order_id: str = Field(default="", max_length=40)


class TicketStatusRequest(BaseModel):
    status: str
    resolution_minutes: int | None = Field(default=None, ge=0)


class FeedbackRequest(BaseModel):
    rating: str
    comment: str = Field(default="", max_length=500)


class SessionCreateRequest(BaseModel):
    title: str = Field(default="", max_length=100)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None
    customer_name: str = Field(default="匿名客户", max_length=50)
    customer_tier: str = "standard"
    order_id: str = Field(default="", max_length=40)


class ToolSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    category: str | None = None
    candidate_k: int = Field(default=12, ge=1, le=30)
    limit: int = Field(default=4, ge=1, le=20)


class ToolTicketRequest(BaseModel):
    ticket_id: str = Field(min_length=1, max_length=40)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = TicketStore(settings.db_path)
    document_store = DocumentStore(settings.db_path)
    session_store = SessionStore(settings.db_path)
    kb = _load_knowledge(settings, document_store)
    engine = TriageEngine(kb, store, settings)

    app = FastAPI(
        title="ServiceFlow AI",
        version="1.1.0",
        description="企业工单分流、知识库问答与人工升级服务",
    )
    app.state.settings = settings
    app.state.knowledge = kb
    app.state.store = store
    app.state.document_store = document_store
    app.state.session_store = session_store
    app.state.engine = engine

    def refresh_knowledge() -> KnowledgeBase:
        refreshed = _load_knowledge(settings, document_store)
        app.state.knowledge = refreshed
        engine.set_knowledge(refreshed)
        return refreshed

    def ensure_session(session_id: str | None, first_message: str = "") -> str:
        if session_id is None:
            return session_store.create(first_message[:100])["id"]
        if session_store.get(session_id) is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return session_id

    def run_chat(payload: ChatRequest) -> dict[str, Any]:
        session_id = ensure_session(payload.session_id, payload.message)
        history = session_store.messages(session_id, limit=20)
        result = engine.assist(
            message=payload.message,
            customer_name=payload.customer_name,
            customer_tier=payload.customer_tier,
            order_id=payload.order_id,
            history=history,
        )
        session_store.append(
            session_id,
            "user",
            result.ticket["message_masked"],
            {"ticket_id": result.ticket["id"]},
        )
        session_store.append(
            session_id,
            "assistant",
            result.ticket["answer"],
            {
                "ticket_id": result.ticket["id"],
                "decision": result.decision,
                "sources": [
                    item["source"] for item in result.ticket["citations"]
                ],
            },
        )
        return {
            "session_id": session_id,
            "ticket": result.ticket,
            "decision": result.decision,
            "explanation": result.explanation,
            "retrieval": result.retrieval,
        }

    @app.exception_handler(HTTPException)
    async def handle_http_error(
        _request: Request, exc: HTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc.detail)},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        first_error = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(part) for part in first_error.get("loc", []))
        message = first_error.get("msg", "请求参数不合法")
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"error": f"{location}: {message}" if location else message},
        )

    @app.exception_handler(ValueError)
    async def handle_value_error(
        _request: Request, exc: ValueError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(exc)},
        )

    @app.exception_handler(KeyError)
    async def handle_key_error(
        _request: Request, exc: KeyError
    ) -> JSONResponse:
        message = str(exc.args[0]) if exc.args else "资源不存在"
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": message},
        )

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "knowledge_chunks": len(app.state.knowledge.chunks),
            "llm_enabled": settings.llm_enabled,
            "agent_framework": "langgraph",
            "tool_count": len(engine.tools.manifests()),
            "model_mode": engine.model.info()["mode"],
            "model_configured": engine.model.configured,
        }

    @app.get("/api/model/info")
    def model_info() -> dict[str, Any]:
        return engine.model.info()

    @app.get("/api/evaluation")
    def evaluation_results() -> dict[str, Any]:
        path = PROJECT_ROOT / "docs" / "evaluation-results.json"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="评测结果尚未生成")
        return json.loads(path.read_text(encoding="utf-8"))

    @app.post("/api/model/test")
    def model_test() -> dict[str, Any]:
        try:
            result = engine.model.ping()
        except ModelCallError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "ok": True,
            "text": result.text,
            "model": result.model,
            "attempts": result.attempts,
            "latency_ms": result.latency_ms,
            "usage": result.usage,
        }

    @app.get("/api/agent/info")
    def agent_info() -> dict[str, Any]:
        return {
            "framework": "langgraph",
            "nodes": list(engine.workflow_nodes),
            "routing": {
                "auto_reply": "persist_ticket",
                "human_review": "human_handoff",
                "escalate": "human_handoff",
            },
            "trace_returned_in": "/api/assist -> retrieval.agent.trace",
            "tools_returned_in": "/api/assist -> retrieval.agent.tool_calls",
        }

    @app.get("/api/tools")
    def list_tools() -> dict[str, Any]:
        return {"items": engine.tools.manifests()}

    @app.post("/api/tools/knowledge_search")
    def invoke_knowledge_search(payload: ToolSearchRequest) -> dict[str, Any]:
        rewritten = rewrite_query(payload.query)
        matches, record = engine.tools.search_knowledge(
            rewritten.rewritten_query,
            payload.category,
            candidate_k=payload.candidate_k,
            limit=payload.limit,
        )
        return {
            "tool_call": record.as_dict(),
            "query": rewritten.original_query,
            "rewritten_query": rewritten.rewritten_query,
            "items": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "title": item.chunk.title,
                    "source": item.chunk.source,
                    "text": item.chunk.text,
                    "relevance": item.relevance,
                }
                for item in matches
            ],
        }

    @app.post("/api/tools/get_ticket")
    def invoke_get_ticket(payload: ToolTicketRequest) -> dict[str, Any]:
        ticket, record = engine.tools.get_ticket(payload.ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        return {"tool_call": record.as_dict(), "ticket": ticket}

    @app.post("/api/assist", status_code=status.HTTP_201_CREATED)
    def assist(payload: AssistRequest) -> dict[str, Any]:
        result = engine.assist(
            message=payload.message,
            customer_name=payload.customer_name,
            customer_tier=payload.customer_tier,
            order_id=payload.order_id,
        )
        return {
            "ticket": result.ticket,
            "decision": result.decision,
            "explanation": result.explanation,
            "retrieval": result.retrieval,
        }

    @app.get("/api/tickets")
    def list_tickets(
        ticket_status: Annotated[str | None, Query(alias="status")] = None,
        category: str | None = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 30,
    ) -> dict[str, Any]:
        return {
            "items": store.list_tickets(
                limit=limit,
                status=ticket_status,
                category=category,
            )
        }

    @app.get("/api/tickets/{ticket_id}")
    def get_ticket(ticket_id: str) -> dict[str, Any]:
        ticket = store.get_ticket(ticket_id)
        if ticket is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        return ticket

    @app.post("/api/tickets/{ticket_id}/status")
    def update_ticket_status(
        ticket_id: str, payload: TicketStatusRequest
    ) -> dict[str, Any]:
        ticket = store.update_status(
            ticket_id,
            payload.status,
            payload.resolution_minutes,
        )
        if ticket is None:
            raise HTTPException(status_code=404, detail="工单不存在")
        return ticket

    @app.post(
        "/api/tickets/{ticket_id}/feedback",
        status_code=status.HTTP_201_CREATED,
    )
    def add_feedback(
        ticket_id: str, payload: FeedbackRequest
    ) -> dict[str, bool]:
        store.add_feedback(ticket_id, payload.rating, payload.comment)
        return {"ok": True}

    @app.get("/api/metrics")
    def metrics() -> dict[str, Any]:
        return store.metrics()

    @app.get("/api/knowledge")
    def knowledge() -> dict[str, Any]:
        grouped: dict[str, dict[str, Any]] = {}
        for chunk in app.state.knowledge.chunks:
            group = grouped.setdefault(
                chunk.doc_id,
                {
                    "id": chunk.doc_id,
                    "title": chunk.title,
                    "category": chunk.category,
                    "team": chunk.team,
                    "source": chunk.source,
                    "chunks": 0,
                },
            )
            group["chunks"] += 1
        return {"items": list(grouped.values())}

    @app.get("/api/retrieval/search")
    def retrieval_search(
        q: Annotated[str, Query(min_length=1, max_length=4000)],
        category: str | None = None,
        top_k: Annotated[int, Query(ge=1, le=20)] = 5,
    ) -> dict[str, Any]:
        rewritten = rewrite_query(q)
        candidates = app.state.knowledge.search(
            rewritten.rewritten_query,
            category=category,
            top_k=max(12, top_k * 3),
        )
        matches = heuristic_rerank(
            rewritten.rewritten_query,
            candidates,
            limit=top_k,
        )
        return {
            "query": rewritten.original_query,
            "rewritten_query": rewritten.rewritten_query,
            "expansions": list(rewritten.expansions),
            "retrieval_mode": "hybrid_then_heuristic_rerank",
            "vector_weight": app.state.knowledge.vector_weight,
            "candidate_count": len(candidates),
            "items": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "document_id": item.chunk.doc_id,
                    "title": item.chunk.title,
                    "category": item.chunk.category,
                    "source": item.chunk.source,
                    "text": item.chunk.text,
                    "lexical_score": round(item.score, 4),
                    "lexical_relevance": item.lexical_relevance,
                    "vector_score": item.vector_score,
                    "hybrid_relevance": item.hybrid_relevance,
                    "rerank_score": item.rerank_score,
                    "relevance": item.relevance,
                }
                for item in matches
            ],
        }

    @app.get("/api/documents")
    def list_documents() -> dict[str, Any]:
        return {"items": document_store.list()}

    @app.get("/api/documents/{document_id}")
    def get_document(document_id: str) -> dict[str, Any]:
        document = document_store.get(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        return document

    @app.post(
        "/api/documents",
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_document(
        file: Annotated[UploadFile, File(description="Markdown 或 TXT 文档")],
        title: Annotated[str, Form(max_length=100)] = "",
        category: Annotated[str, Form(max_length=40)] = "general",
        team: Annotated[str, Form(max_length=50)] = "一线服务组",
    ) -> dict[str, Any]:
        raw = await file.read(MAX_DOCUMENT_BYTES + 1)
        document = parse_document(
            filename=file.filename or "document.txt",
            raw=raw,
            media_type=file.content_type or "text/plain",
            title=title,
            category=category,
            team=team,
        )
        created = document_store.create(document)
        refreshed = refresh_knowledge()
        return {
            "document": created,
            "knowledge_chunks": len(refreshed.chunks),
        }

    @app.get("/api/sessions")
    def list_sessions() -> dict[str, Any]:
        return {"items": session_store.list()}

    @app.post("/api/sessions", status_code=status.HTTP_201_CREATED)
    def create_session(payload: SessionCreateRequest) -> dict[str, Any]:
        return session_store.create(payload.title)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, Any]:
        session = session_store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return session

    @app.post("/api/chat", status_code=status.HTTP_201_CREATED)
    def chat(payload: ChatRequest) -> dict[str, Any]:
        return run_chat(payload)

    @app.post("/api/chat/stream")
    def chat_stream(payload: ChatRequest) -> StreamingResponse:
        def event_stream():
            try:
                session_id = ensure_session(payload.session_id, payload.message)
                history = session_store.messages(session_id, limit=20)
                yield _sse("session", {"session_id": session_id})
                for event in engine.stream_assist(
                    message=payload.message,
                    customer_name=payload.customer_name,
                    customer_tier=payload.customer_tier,
                    order_id=payload.order_id,
                    history=history,
                ):
                    if event["event"] == "final":
                        final = event["data"]
                        session_store.append(
                            session_id,
                            "user",
                            final["ticket"]["message_masked"],
                            {"ticket_id": final["ticket"]["id"]},
                        )
                        session_store.append(
                            session_id,
                            "assistant",
                            final["ticket"]["answer"],
                            {
                                "ticket_id": final["ticket"]["id"],
                                "decision": final["decision"],
                                "sources": [
                                    item["source"]
                                    for item in final["ticket"]["citations"]
                                ],
                            },
                        )
                    yield _sse(event["event"], event["data"])
            except Exception as exc:
                yield _sse(
                    "error",
                    {"error": str(exc) or exc.__class__.__name__},
                )

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/knowledge/reload")
    def reload_knowledge() -> dict[str, Any]:
        refreshed = refresh_knowledge()
        return {"ok": True, "knowledge_chunks": len(refreshed.chunks)}

    @app.api_route(
        "/api/{unknown_path:path}",
        methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        include_in_schema=False,
    )
    def unknown_api(unknown_path: str) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": f"API 不存在：/api/{unknown_path}"},
        )

    app.mount(
        "/",
        StaticFiles(directory=settings.web_dir, html=True),
        name="web",
    )
    return app


def _load_knowledge(
    settings: Settings, document_store: DocumentStore
) -> KnowledgeBase:
    kb = KnowledgeBase.load(
        settings.knowledge_dir,
        vector_weight=settings.vector_weight,
        vector_dimensions=settings.vector_dimensions,
    )
    kb.add_chunks(document_store.list_chunks())
    return kb


def _sse(event: str, data: dict[str, Any]) -> str:
    import json

    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


app = create_app()
