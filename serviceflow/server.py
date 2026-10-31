from __future__ import annotations

import json
import mimetypes
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import Settings
from .engine import TriageEngine
from .knowledge import KnowledgeBase
from .storage import TicketStore


class ServiceFlowApp:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.kb = KnowledgeBase.load(settings.knowledge_dir)
        self.store = TicketStore(settings.db_path)
        self.engine = TriageEngine(self.kb, self.store, settings)

    def reload_knowledge(self) -> int:
        self.kb = KnowledgeBase.load(self.settings.knowledge_dir)
        self.engine.kb = self.kb
        return len(self.kb.chunks)


class APIHandler(BaseHTTPRequestHandler):
    app: ServiceFlowApp
    server_version = "ServiceFlow/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self._handle_api_get(parsed.path, parse_qs(parsed.query))
        else:
            self._serve_static(parsed.path)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            payload = self._read_json()
            self._handle_api_post(parsed.path, payload)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except KeyError as exc:
            self._send_json({"error": f"资源不存在：{exc.args[0]}"}, HTTPStatus.NOT_FOUND)
        except Exception:
            self._send_json(
                {"error": "服务端处理失败，请稍后重试"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def _handle_api_get(self, path: str, query: dict[str, list[str]]) -> None:
        if path == "/api/health":
            self._send_json(
                {
                    "status": "ok",
                    "knowledge_chunks": len(self.app.kb.chunks),
                    "llm_enabled": self.app.settings.llm_enabled,
                }
            )
            return
        if path == "/api/tickets":
            status = _first(query, "status")
            category = _first(query, "category")
            limit = int(_first(query, "limit") or "30")
            self._send_json(
                {"items": self.app.store.list_tickets(limit, status, category)}
            )
            return
        if path == "/api/metrics":
            self._send_json(self.app.store.metrics())
            return
        if path == "/api/knowledge":
            grouped: dict[str, dict[str, Any]] = {}
            for chunk in self.app.kb.chunks:
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
            self._send_json({"items": list(grouped.values())})
            return
        ticket_match = re.fullmatch(r"/api/tickets/([^/]+)", path)
        if ticket_match:
            ticket = self.app.store.get_ticket(ticket_match.group(1))
            if not ticket:
                self._send_json({"error": "工单不存在"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(ticket)
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _handle_api_post(self, path: str, payload: dict[str, Any]) -> None:
        if path == "/api/assist":
            message = str(payload.get("message", ""))
            result = self.app.engine.assist(
                message=message,
                customer_name=str(payload.get("customer_name", "匿名客户")),
                customer_tier=str(payload.get("customer_tier", "standard")),
                order_id=str(payload.get("order_id", "")),
            )
            self._send_json(
                {
                    "ticket": result.ticket,
                    "decision": result.decision,
                    "explanation": result.explanation,
                },
                HTTPStatus.CREATED,
            )
            return
        status_match = re.fullmatch(r"/api/tickets/([^/]+)/status", path)
        if status_match:
            status = str(payload.get("status", ""))
            resolution_minutes = payload.get("resolution_minutes")
            if resolution_minutes is not None:
                resolution_minutes = int(resolution_minutes)
            ticket = self.app.store.update_status(
                status_match.group(1), status, resolution_minutes
            )
            self._send_json(ticket)
            return
        feedback_match = re.fullmatch(r"/api/tickets/([^/]+)/feedback", path)
        if feedback_match:
            self.app.store.add_feedback(
                feedback_match.group(1),
                str(payload.get("rating", "")),
                str(payload.get("comment", "")),
            )
            self._send_json({"ok": True}, HTTPStatus.CREATED)
            return
        if path == "/api/knowledge/reload":
            count = self.app.reload_knowledge()
            self._send_json({"ok": True, "knowledge_chunks": count})
            return
        self._send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > 128 * 1024:
            raise ValueError("请求体过大")
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体必须为合法 JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("请求体必须为 JSON 对象")
        return value

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path in {"", "/"} else path.lstrip("/")
        target = (self.app.settings.web_dir / relative).resolve()
        web_root = self.app.settings.web_dir.resolve()
        if web_root not in target.parents and target != web_root:
            self._send_json({"error": "Forbidden"}, HTTPStatus.FORBIDDEN)
            return
        if not target.is_file():
            target = web_root / "index.html"
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the default server log concise and avoid logging request bodies.
        print(f"[http] {self.address_string()} {format % args}")


def _first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def create_server(settings: Settings) -> ThreadingHTTPServer:
    app = ServiceFlowApp(settings)
    handler = type("BoundAPIHandler", (APIHandler,), {"app": app})
    server = ThreadingHTTPServer((settings.host, settings.port), handler)
    return server

