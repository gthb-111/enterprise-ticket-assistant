import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from docx import Document
from fastapi.testclient import TestClient

from serviceflow.api import create_app
from serviceflow.cli import main
from serviceflow.config import Settings


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        base = Settings.from_env()
        self.settings = Settings(
            **{
                **base.__dict__,
                "db_path": Path(self.tempdir.name) / "api-test.db",
                "llm_enabled": False,
                "openai_api_key": "",
            }
        )
        self.client = TestClient(create_app(self.settings))

    def tearDown(self):
        self.client.close()
        self.tempdir.cleanup()

    def test_health_and_openapi_are_available(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["agent_framework"], "langgraph")
        self.assertEqual(response.json()["model_mode"], "offline_fallback")

        schema = self.client.get("/openapi.json")
        self.assertEqual(schema.status_code, 200)
        self.assertIn("/api/assist", schema.json()["paths"])

    def test_model_info_is_safe_without_configuration(self):
        response = self.client.get("/api/model/info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["configured"])
        self.assertEqual(payload["mode"], "offline_fallback")
        self.assertNotIn("api_key", payload)

    def test_evaluation_results_are_available(self):
        response = self.client.get("/api/evaluation")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(payload["cases"], 50)
        self.assertEqual(
            {item["name"] for item in payload["retrieval_modes"]},
            {"lexical", "vector", "hybrid", "hybrid_rerank"},
        )

    def test_assist_creates_ticket_and_metrics(self):
        response = self.client.post(
            "/api/assist",
            json={
                "message": "退款审核通过后多久到账？",
                "customer_name": "林女士",
                "customer_tier": "standard",
            },
        )
        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["decision"], "auto_reply")
        self.assertTrue(payload["ticket"]["id"].startswith("TK-"))
        trace = payload["retrieval"]["agent"]["trace"]
        self.assertEqual(
            [item["node"] for item in trace],
            [
                "prepare",
                "knowledge_search",
                "compose",
                "persist_ticket",
            ],
        )
        tool_names = [
            item["name"]
            for item in payload["retrieval"]["agent"]["tool_calls"]
        ]
        self.assertEqual(tool_names, ["knowledge_search", "create_ticket"])

        metrics = self.client.get("/api/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(metrics.json()["total"], 1)

    def test_domain_validation_errors_keep_error_field(self):
        response = self.client.post("/api/assist", json={"message": ""})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json())

    def test_unknown_ticket_returns_json_404(self):
        response = self.client.get("/api/tickets/TK-NOT-FOUND")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"error": "工单不存在"})

    def test_uploaded_document_is_persisted_and_searchable(self):
        upload = self.client.post(
            "/api/documents",
            data={
                "title": "供应商结算 SLA",
                "category": "billing",
                "team": "财务支持组",
            },
            files={
                "file": (
                    "supplier_sla.md",
                    "# 供应商结算 SLA\n\n"
                    "星港供应商的对账单必须在每月八日前提交。\n\n"
                    "财务复核通常在两个工作日内完成，节假日顺延。",
                    "text/markdown",
                )
            },
        )
        self.assertEqual(upload.status_code, 201)
        document_id = upload.json()["document"]["id"]
        self.assertGreaterEqual(upload.json()["document"]["chunk_count"], 1)

        documents = self.client.get("/api/documents").json()["items"]
        self.assertEqual(documents[0]["id"], document_id)

        answer = self.client.post(
            "/api/assist",
            json={
                "message": "星港供应商的对账单什么时候提交？",
                "customer_tier": "standard",
            },
        )
        self.assertEqual(answer.status_code, 201)
        citations = answer.json()["ticket"]["citations"]
        self.assertTrue(
            any(item["source"] == "upload://supplier_sla.md" for item in citations)
        )
        self.assertIn("每月八日前", answer.json()["ticket"]["answer"])

        restarted_client = TestClient(create_app(self.settings))
        try:
            persisted = restarted_client.get("/api/documents")
            self.assertEqual(persisted.status_code, 200)
            self.assertEqual(persisted.json()["items"][0]["id"], document_id)
        finally:
            restarted_client.close()

    def test_unsupported_document_type_is_rejected(self):
        response = self.client.post(
            "/api/documents",
            files={"file": ("payload.exe", b"not a document", "application/octet-stream")},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("当前仅支持", response.json()["error"])

    def test_docx_document_is_extracted(self):
        document = Document()
        document.add_heading("办公用品采购规则", level=1)
        document.add_paragraph("单笔超过五千元需要部门负责人审批。")
        buffer = BytesIO()
        document.save(buffer)
        response = self.client.post(
            "/api/documents",
            data={"title": "办公用品采购规则", "category": "general"},
            files={
                "file": (
                    "procurement.docx",
                    buffer.getvalue(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        self.assertEqual(response.status_code, 201)
        self.assertGreaterEqual(response.json()["document"]["chunk_count"], 1)

    def test_retrieval_debug_endpoint_returns_hybrid_scores(self):
        response = self.client.get(
            "/api/retrieval/search",
            params={
                "q": "退款什么时候到账",
                "category": "refund",
                "top_k": 3,
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["retrieval_mode"], "hybrid_then_heuristic_rerank")
        self.assertIn("rewritten_query", payload)
        self.assertTrue(payload["items"])
        self.assertGreater(payload["items"][0]["vector_score"], 0)
        self.assertGreater(payload["items"][0]["rerank_score"], 0)

    def test_agent_info_and_human_handoff_branch(self):
        info = self.client.get("/api/agent/info")
        self.assertEqual(info.status_code, 200)
        self.assertEqual(info.json()["framework"], "langgraph")

        response = self.client.post(
            "/api/assist",
            json={
                "message": "账号被盗，非本人刷卡消费，请马上冻结账号",
                "customer_tier": "vip",
            },
        )
        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["decision"], "escalate")
        trace = payload["retrieval"]["agent"]["trace"]
        self.assertEqual(trace[-1]["node"], "human_handoff")
        tool_names = [
            item["name"]
            for item in payload["retrieval"]["agent"]["tool_calls"]
        ]
        self.assertEqual(tool_names, ["knowledge_search", "handoff_ticket"])

    def test_tool_manifests_and_knowledge_search(self):
        manifests = self.client.get("/api/tools")
        self.assertEqual(manifests.status_code, 200)
        names = [item["name"] for item in manifests.json()["items"]]
        self.assertEqual(
            names,
            [
                "knowledge_search",
                "get_ticket",
                "create_ticket",
                "handoff_ticket",
            ],
        )
        search = self.client.post(
            "/api/tools/knowledge_search",
            json={
                "query": "退款多久到账",
                "category": "refund",
                "candidate_k": 8,
                "limit": 3,
            },
        )
        self.assertEqual(search.status_code, 200)
        self.assertEqual(
            search.json()["tool_call"]["name"],
            "knowledge_search",
        )
        self.assertEqual(search.json()["tool_call"]["status"], "ok")
        self.assertTrue(search.json()["items"])

    def test_chat_session_keeps_follow_up_context(self):
        self.client.post(
            "/api/documents",
            data={
                "title": "差旅报销时效说明",
                "category": "billing",
                "team": "财务支持组",
            },
            files={
                "file": (
                    "travel.md",
                    "# 差旅报销时效说明\n\n"
                    "杭州地区网约车票据应在行程结束后六十天内提交。\n\n"
                    "超过六十天需要部门负责人补充审批说明。",
                    "text/markdown",
                )
            },
        )
        session = self.client.post("/api/sessions", json={"title": "报销咨询"})
        self.assertEqual(session.status_code, 201)
        session_id = session.json()["id"]

        first = self.client.post(
            "/api/chat",
            json={
                "session_id": session_id,
                "message": "打车费多久能报销？",
            },
        )
        self.assertEqual(first.status_code, 201)

        second = self.client.post(
            "/api/chat",
            json={
                "session_id": session_id,
                "message": "那超过六十天怎么办？",
            },
        )
        self.assertEqual(second.status_code, 201)
        self.assertTrue(second.json()["retrieval"]["memory_used"])
        self.assertIn(
            "打车费多久能报销",
            second.json()["retrieval"]["contextual_query"],
        )

        history = self.client.get(f"/api/sessions/{session_id}")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()["messages"]), 4)

    def test_chat_stream_returns_nodes_and_answer_deltas(self):
        with self.client.stream(
            "POST",
            "/api/chat/stream",
            json={"message": "退款审核通过后多久到账？"},
        ) as response:
            self.assertEqual(response.status_code, 200)
            body = "".join(response.iter_text())
        self.assertIn("event: session", body)
        self.assertIn("event: node", body)
        self.assertIn("event: answer_delta", body)
        self.assertIn("event: final", body)

    def test_root_serves_workbench(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ServiceFlow", response.text)

    def test_cli_serve_delegates_to_uvicorn_run(self):
        with patch("serviceflow.cli.create_server") as create_server_mock, patch(
            "sys.argv", ["serviceflow", "serve"]
        ):
            main()
        create_server_mock.return_value.run.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
