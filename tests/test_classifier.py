import unittest

from serviceflow.classifier import classify


class ClassifierTests(unittest.TestCase):
    def test_refund_intent(self):
        result = classify("退款审核通过后多久到账？")
        self.assertEqual(result.category, "refund")
        self.assertGreater(result.confidence, 0.7)

    def test_logistics_intent(self):
        result = classify("包裹运输五天没有物流更新")
        self.assertEqual(result.category, "logistics")

    def test_financial_risk_is_p0_and_human(self):
        result = classify("怀疑账号被盗，银行卡被盗刷")
        self.assertEqual(result.category, "account")
        self.assertEqual(result.priority, "P0")
        self.assertTrue(result.requires_human)

    def test_regulatory_word_escalates(self):
        result = classify("客服不处理，我要向12315投诉")
        self.assertEqual(result.category, "service")
        self.assertEqual(result.priority, "P0")

    def test_vip_negative_raises_priority(self):
        result = classify("我是会员，对这个结果很失望", customer_tier="vip")
        self.assertEqual(result.priority, "P1")


if __name__ == "__main__":
    unittest.main()

