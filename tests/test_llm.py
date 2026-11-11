import unittest

import httpx

from serviceflow.config import Settings
from serviceflow.llm import GroundedModel, ModelCallError


class ModelTests(unittest.TestCase):
    def _settings(self, **overrides):
        base = Settings.from_env()
        values = {
            **base.__dict__,
            "llm_enabled": True,
            "openai_api_key": "test-secret",
            "openai_base_url": "https://example.test",
            "openai_model": "test-model",
            "llm_timeout_seconds": 1,
            "llm_max_retries": 1,
            "llm_max_concurrency": 1,
            "llm_max_tokens": 200,
        }
        values.update(overrides)
        return Settings(**values)

    def test_info_never_exposes_secret(self):
        model = GroundedModel(self._settings())
        info = model.info()
        self.assertTrue(info["configured"])
        self.assertEqual(info["model"], "test-model")
        self.assertNotIn("test-secret", str(info))

    def test_chat_parses_content_and_usage(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/chat/completions")
            self.assertEqual(
                request.headers["Authorization"],
                "Bearer test-secret",
            )
            return httpx.Response(
                200,
                json={
                    "model": "test-model",
                    "choices": [
                        {"message": {"content": "测试回答 [c1]"}}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        model = GroundedModel(self._settings(), client=client)
        result = model.chat([{"role": "user", "content": "测试"}])
        self.assertEqual(result.text, "测试回答 [c1]")
        self.assertEqual(result.usage["completion_tokens"], 5)

    def test_unconfigured_model_raises(self):
        model = GroundedModel(
            self._settings(openai_api_key="", llm_enabled=False)
        )
        with self.assertRaises(ModelCallError):
            model.ping()

    def test_stream_chat_forwards_delta_chunks(self):
        stream_body = (
            'data: {"model":"test-model","choices":[{"delta":{"content":"你"}}]}\n\n'
            'data: {"model":"test-model","choices":[{"delta":{"content":"好"}}]}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=stream_body.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
            )

        client = httpx.Client(transport=httpx.MockTransport(handler))
        model = GroundedModel(self._settings(), client=client)
        deltas = []
        result = model.chat(
            [{"role": "user", "content": "测试"}],
            on_delta=deltas.append,
        )
        self.assertEqual(deltas, ["你", "好"])
        self.assertEqual(result.text, "你好")


if __name__ == "__main__":
    unittest.main()
