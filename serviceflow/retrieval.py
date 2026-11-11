from __future__ import annotations

import re
from dataclasses import dataclass

from .knowledge import RetrievedChunk, tokenize


REWRITE_RULES: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("打车", "网约车", "出租车"), ("网约车", "出租车", "交通票据", "报销")),
    (("报销", "票据", "发票"), ("报销", "凭证", "审核")),
    (("到账", "退回", "退款"), ("退款", "到账", "时效")),
    (("多久", "几天", "什么时候"), ("时效", "工作日")),
    (("有人回复", "人工响应"), ("人工响应", "服务时间")),
    (("登录不上", "登不上", "无法登录"), ("登录失败", "验证码", "账号")),
    (("续费", "订阅"), ("自动续费", "取消订阅", "账单")),
    (("扣款", "扣钱"), ("重复扣款", "支付", "账单")),
)

CONTEXT_REFERENCES = (
    "那",
    "这个",
    "它",
    "上面",
    "前面",
    "还有",
    "另外",
    "超过",
    "怎么办",
    "然后",
)


@dataclass(frozen=True)
class RewriteResult:
    original_query: str
    rewritten_query: str
    expansions: tuple[str, ...]


def rewrite_query(query: str, max_expansions: int = 8) -> RewriteResult:
    """Apply conservative business-term expansion without changing intent."""
    original = re.sub(r"\s+", " ", query).strip()
    lowered = original.lower()
    expansions: list[str] = []
    for triggers, additions in REWRITE_RULES:
        if not any(trigger in lowered for trigger in triggers):
            continue
        for addition in additions:
            if addition not in lowered and addition not in expansions:
                expansions.append(addition)
            if len(expansions) >= max_expansions:
                break
        if len(expansions) >= max_expansions:
            break
    rewritten = original
    if expansions:
        rewritten = f"{original} {' '.join(expansions)}"
    return RewriteResult(
        original_query=original,
        rewritten_query=rewritten,
        expansions=tuple(expansions),
    )


def build_contextual_query(
    current_message: str,
    history: list[dict[str, object]] | None,
) -> tuple[str, bool]:
    """Attach the latest user turn when the current message depends on it."""
    if not history:
        return current_message, False
    previous_user_messages = [
        str(item.get("content", ""))
        for item in history
        if item.get("role") == "user" and item.get("content")
    ][-2:]
    if not previous_user_messages:
        return current_message, False
    needs_context = len(current_message) <= 20 or any(
        phrase in current_message[:20] for phrase in CONTEXT_REFERENCES
    )
    if not needs_context:
        return current_message, False
    context = "；".join(previous_user_messages)
    return f"{context}；{current_message}", True


def heuristic_rerank(
    query: str,
    matches: list[RetrievedChunk],
    limit: int = 4,
) -> list[RetrievedChunk]:
    """Rerank candidates using title relevance and evidence coverage."""
    query_terms = set(tokenize(query))
    if not query_terms:
        return matches[:limit]
    for item in matches:
        title_terms = set(tokenize(item.chunk.title))
        title_coverage = len(query_terms & title_terms) / max(len(query_terms), 1)
        score = (
            0.94 * item.hybrid_relevance
            + 0.04 * title_coverage
            + 0.02 * item.coverage
        )
        item.rerank_score = round(min(0.99, score), 4)
        item.relevance = item.rerank_score
    matches.sort(key=lambda item: item.rerank_score, reverse=True)
    return matches[:limit]
