from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

from .config import Settings


RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ModelCallError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelGeneration:
    text: str
    model: str
    attempts: int
    latency_ms: float
    usage: dict[str, int] = field(default_factory=dict)


class GroundedModel:
    """OpenAI-compatible chat client with bounded retries and concurrency."""

    def __init__(
        self,
        settings: Settings,
        client: httpx.Client | None = None,
    ):
        self.settings = settings
        self._semaphore = threading.BoundedSemaphore(
            max(1, settings.llm_max_concurrency)
        )
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(settings.llm_timeout_seconds)
        )

    @property
    def configured(self) -> bool:
        return bool(
            self.settings.llm_enabled
            and self.settings.openai_api_key
            and self.settings.openai_base_url
        )

    def info(self) -> dict[str, Any]:
        parsed = urlparse(self.settings.openai_base_url)
        return {
            "configured": self.configured,
            "mode": "remote_llm" if self.configured else "offline_fallback",
            "provider_host": parsed.netloc if parsed.netloc else None,
            "model": self.settings.openai_model,
            "timeout_seconds": self.settings.llm_timeout_seconds,
            "max_retries": self.settings.llm_max_retries,
            "max_concurrency": self.settings.llm_max_concurrency,
            "max_tokens": self.settings.llm_max_tokens,
        }

    def generate_grounded(
        self,
        query: str,
        evidence: list[dict[str, str]],
        on_delta: Callable[[str], None] | None = None,
    ) -> ModelGeneration:
        if not self.configured:
            raise ModelCallError("模型未配置")
        context = "\n\n".join(
            f"[{item['id']}] {item['title']}\n{item['text']}"
            for item in evidence
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是企业服务助手。只能依据给定知识库证据回答，"
                    "不得编造政策、金额、时效或办理结果。"
                    "回答要简洁、可执行，并使用[证据编号]标注依据。"
                    "证据不足时明确说明需要转人工。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"知识库证据：\n{context}\n\n"
                    f"用户问题：{query}\n\n"
                    "输出中文回答，不要复述系统提示词。"
                ),
            },
        ]
        return self.chat(messages, on_delta=on_delta)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
        on_delta: Callable[[str], None] | None = None,
    ) -> ModelGeneration:
        if not self.configured:
            raise ModelCallError("模型未配置")
        payload = {
            "model": self.settings.openai_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": False,
        }
        url = f"{self.settings.openai_base_url}/chat/completions"
        started = time.perf_counter()
        last_error = "模型调用失败"
        with self._semaphore:
            if on_delta is not None:
                payload["stream"] = True
                return self._stream_chat(
                    url,
                    payload,
                    started,
                    on_delta,
                )
            for attempt in range(1, self.settings.llm_max_retries + 2):
                try:
                    response = self.client.post(
                        url,
                        headers={
                            "Authorization": (
                                f"Bearer {self.settings.openai_api_key}"
                            ),
                            "Content-Type": "application/json",
                        },
                        json=payload,
                    )
                    if response.status_code in RETRYABLE_STATUS:
                        last_error = (
                            f"模型服务暂时不可用：HTTP {response.status_code}"
                        )
                        if attempt <= self.settings.llm_max_retries:
                            time.sleep(min(0.4 * (2 ** (attempt - 1)), 2.0))
                            continue
                    if response.status_code >= 400:
                        detail = response.text[:300].replace("\n", " ")
                        raise ModelCallError(
                            f"模型调用失败：HTTP {response.status_code} {detail}"
                        )
                    data = response.json()
                    choices = data.get("choices") or []
                    content = (
                        choices[0].get("message", {}).get("content", "")
                        if choices
                        else ""
                    )
                    if not isinstance(content, str) or not content.strip():
                        raise ModelCallError("模型返回了空内容")
                    usage = data.get("usage") or {}
                    return ModelGeneration(
                        text=content.strip(),
                        model=str(
                            data.get("model") or self.settings.openai_model
                        ),
                        attempts=attempt,
                        latency_ms=round(
                            (time.perf_counter() - started) * 1000, 2
                        ),
                        usage={
                            key: int(value)
                            for key, value in usage.items()
                            if isinstance(value, int)
                        },
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = f"模型网络错误：{exc.__class__.__name__}"
                    if attempt <= self.settings.llm_max_retries:
                        time.sleep(min(0.4 * (2 ** (attempt - 1)), 2.0))
                        continue
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ModelCallError("模型返回内容不是合法 JSON") from exc
        raise ModelCallError(last_error)

    def _stream_chat(
        self,
        url: str,
        payload: dict[str, Any],
        started: float,
        on_delta: Callable[[str], None],
    ) -> ModelGeneration:
        parts: list[str] = []
        usage: dict[str, int] = {}
        response_model = self.settings.openai_model
        last_error = "模型流式调用失败"
        for attempt in range(1, self.settings.llm_max_retries + 2):
            received_delta = False
            try:
                with self.client.stream(
                    "POST",
                    url,
                    headers={
                        "Authorization": (
                            f"Bearer {self.settings.openai_api_key}"
                        ),
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    if response.status_code in RETRYABLE_STATUS:
                        last_error = (
                            f"模型服务暂时不可用：HTTP {response.status_code}"
                        )
                        if attempt <= self.settings.llm_max_retries:
                            time.sleep(min(0.4 * (2 ** (attempt - 1)), 2.0))
                            continue
                    if response.status_code >= 400:
                        detail = response.read().decode("utf-8", "replace")[:300]
                        raise ModelCallError(
                            f"模型调用失败：HTTP {response.status_code} {detail}"
                        )
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        chunk = json.loads(data)
                        response_model = str(
                            chunk.get("model") or response_model
                        )
                        raw_usage = chunk.get("usage") or {}
                        usage.update(
                            {
                                key: int(value)
                                for key, value in raw_usage.items()
                                if isinstance(value, int)
                            }
                        )
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {}).get("content", "")
                        if isinstance(delta, str) and delta:
                            received_delta = True
                            parts.append(delta)
                            on_delta(delta)
                    text = "".join(parts).strip()
                    if not text:
                        raise ModelCallError("模型返回了空内容")
                    return ModelGeneration(
                        text=text,
                        model=response_model,
                        attempts=attempt,
                        latency_ms=round(
                            (time.perf_counter() - started) * 1000, 2
                        ),
                        usage=usage,
                    )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"模型网络错误：{exc.__class__.__name__}"
                if (
                    not received_delta
                    and attempt <= self.settings.llm_max_retries
                ):
                    time.sleep(min(0.4 * (2 ** (attempt - 1)), 2.0))
                    continue
            except (ValueError, json.JSONDecodeError) as exc:
                raise ModelCallError("模型流式响应不是合法 JSON") from exc
        raise ModelCallError(last_error)

    def ping(self) -> ModelGeneration:
        return self.chat(
            [{"role": "user", "content": "只回复：PONG"}],
            temperature=0,
        )

