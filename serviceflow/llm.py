from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .config import Settings


@dataclass
class LLMResult:
    text: str
    mode: str


def polish_answer(
    query: str,
    evidence: list[dict[str, str]],
    settings: Settings,
) -> LLMResult | None:
    """Optional OpenAI-compatible generation grounded in retrieved evidence."""
    if not settings.llm_enabled or not settings.openai_api_key or not evidence:
        return None
    context = "\n\n".join(
        f"[{item['id']}] {item['title']}\n{item['text']}" for item in evidence
    )
    prompt = (
        "你是企业客服助手。只根据给定知识库证据回答，不得编造政策、时效或金额。"
        "用中文给出简洁、可执行的答复，并在相关句末标注 [引用编号]。"
        "如果证据不足以回答，明确说明将转交人工。\n\n"
        f"知识库证据：\n{context}\n\n用户问题：{query}"
    )
    data = json.dumps(
        {
            "model": settings.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{settings.openai_base_url}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {settings.openai_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        text = payload["choices"][0]["message"]["content"].strip()
        return LLMResult(text=text, mode="llm_grounded") if text else None
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError):
        # Local extractive mode remains available when the upstream model fails.
        return None

