import tempfile
import unittest
from pathlib import Path

from serviceflow.config import PROJECT_ROOT, Settings
from serviceflow.engine import TriageEngine
from serviceflow.knowledge import KnowledgeBase
from serviceflow.storage import TicketStore


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        settings = Settings.from_env()
        self.settings = Settings(
            **{
                **settings.__dict__,
                "db_path": Path(self.tempdir.name) / "test.db",
                "llm_enabled": False,
                "openai_api_key": "",
            }
        )
        self.store = TicketStore(self.settings.db_path)
        self.engine = TriageEngine(
            KnowledgeBase.load(PROJECT_ROOT / "knowledge"),
            self.store,
            self.settings,
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def test_clear_faq_is_auto_resolved(self):
        result = self.engine.assist("退款审核通过后多久到账到银行卡？", "林女士")
        self.assertEqual(result.decision, "auto_reply")
        self.assertEqual(result.ticket["status"], "auto_resolved")
        self.assertEqual(result.ticket["category"], "refund")
        self.assertIn("3 至 7 个工作日", result.ticket["answer"])
        self.assertTrue(result.ticket["citations"])

    def test_high_risk_is_escalated(self):
        result = self.engine.assist(
            "账号被盗，非本人刷卡消费，请马上冻结账号",
            "赵先生",
            "vip",
        )
        self.assertEqual(result.decision, "escalate")
        self.assertEqual(result.ticket["priority"], "P0")

    def test_unknown_issue_uses_safe_fallback(self):
        result = self.engine.assist("我想咨询一个火星天气问题")
        self.assertEqual(result.decision, "escalate")
        self.assertEqual(result.ticket["generation_mode"], "safe_fallback")
        self.assertFalse(result.ticket["citations"])

    def test_pii_not_stored_in_masked_message(self):
        result = self.engine.assist("请联系我 13800138000 处理退款")
        self.assertNotIn("13800138000", result.ticket["message_masked"])
        self.assertIn("PHONE_", result.ticket["message_masked"])

    def test_feedback_updates_metrics(self):
        result = self.engine.assist("退款多久到账？")
        self.store.add_feedback(result.ticket["id"], "helpful")
        metrics = self.store.metrics()
        self.assertEqual(metrics["feedback_count"], 1)
        self.assertEqual(metrics["helpful_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()

