import unittest
from pathlib import Path

from serviceflow.knowledge import KnowledgeBase


ROOT = Path(__file__).resolve().parent.parent


class KnowledgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = KnowledgeBase.load(ROOT / "knowledge")

    def test_refund_timing_is_retrieved(self):
        matches = self.kb.search("退款审核通过后多久到账", "refund")
        self.assertTrue(matches)
        self.assertEqual(matches[0].chunk.doc_id, "refund_timing")
        self.assertGreater(matches[0].relevance, 0.6)

    def test_category_boost_helps_disambiguation(self):
        matches = self.kb.search("会员续费后怎么取消", "billing")
        self.assertTrue(matches)
        self.assertEqual(matches[0].chunk.doc_id, "subscription_cancel")

    def test_unknown_query_has_no_strong_match(self):
        matches = self.kb.search("火星天气如何")
        self.assertFalse(matches)


if __name__ == "__main__":
    unittest.main()

