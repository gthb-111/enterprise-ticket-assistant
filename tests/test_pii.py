import unittest

from serviceflow.pii import mask_pii


class PiiMaskTests(unittest.TestCase):
    def test_common_identifiers_are_masked(self):
        result = mask_pii(
            "手机13800138000，邮箱user@example.com，订单SF202610300001"
        )
        self.assertNotIn("13800138000", result.text)
        self.assertNotIn("user@example.com", result.text)
        self.assertNotIn("SF202610300001", result.text)
        self.assertEqual(result.found, ["EMAIL", "ORDER_ID", "PHONE"])

    def test_non_sensitive_numbers_remain(self):
        result = mask_pii("退款需要3到7个工作日")
        self.assertEqual(result.text, "退款需要3到7个工作日")
        self.assertEqual(result.found, [])


if __name__ == "__main__":
    unittest.main()

