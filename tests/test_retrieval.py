import unittest

from serviceflow.knowledge import KnowledgeBase
from serviceflow.retrieval import heuristic_rerank, rewrite_query


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from serviceflow.config import PROJECT_ROOT

        cls.kb = KnowledgeBase.load(PROJECT_ROOT / "knowledge")

    def test_query_rewrite_adds_business_terms(self):
        result = rewrite_query("打车费多久能报销")
        self.assertIn("网约车", result.expansions)
        self.assertIn("交通票据", result.expansions)
        self.assertIn("时效", result.expansions)

    def test_heuristic_rerank_sets_final_scores(self):
        candidates = self.kb.search("退款到账时效", "refund", top_k=5)
        matches = heuristic_rerank("退款到账时效", candidates, limit=3)
        self.assertTrue(matches)
        self.assertGreater(matches[0].rerank_score, 0)
        self.assertEqual(matches[0].relevance, matches[0].rerank_score)


if __name__ == "__main__":
    unittest.main()
