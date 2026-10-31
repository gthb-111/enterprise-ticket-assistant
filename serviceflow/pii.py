from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field


@dataclass
class MaskResult:
    text: str
    found: list[str] = field(default_factory=list)


PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PHONE", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    (
        "ID_CARD",
        re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)|(?<!\d)\d{15}(?!\d)"),
    ),
    ("BANK_CARD", re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)")),
    (
        "ORDER_ID",
        re.compile(r"(?i)(?<![a-z0-9])(?:SF|JD|TB|ORDER)[-_]?[A-Z0-9]{6,18}(?![a-z0-9])"),
    ),
)


def mask_pii(text: str) -> MaskResult:
    """Mask common personal and payment identifiers before persistence/logging."""
    masked = text
    found: list[str] = []
    for label, pattern in PATTERNS:
        def replace(match: re.Match[str]) -> str:
            found.append(label)
            raw = match.group(0)
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:6]
            return f"[{label}_{digest}]"

        masked = pattern.sub(replace, masked)
    return MaskResult(text=masked, found=sorted(set(found)))

