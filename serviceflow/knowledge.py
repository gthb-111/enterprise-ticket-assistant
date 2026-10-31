from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path


FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+")
SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")

STOP_TERMS = {
    "请问",
    "一下",
    "怎么",
    "怎样",
    "如何",
    "可以",
    "是否",
    "我的",
    "我想",
    "现在",
    "这个",
    "那个",
    "咨询",
    "一个",
    "问题",
    "客服",
    "麻烦",
}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    category: str
    team: str
    text: str
    source: str


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    coverage: float

    @property
    def relevance(self) -> float:
        raw = self.score / (self.score + 5.0)
        return round(min(0.99, 0.58 * raw + 0.42 * self.coverage), 3)


@dataclass
class KnowledgeBase:
    chunks: list[Chunk] = field(default_factory=list)
    _term_frequency: list[dict[str, int]] = field(default_factory=list)
    _doc_frequency: dict[str, int] = field(default_factory=dict)
    _avg_length: float = 0.0

    @classmethod
    def load(cls, directory: Path) -> "KnowledgeBase":
        kb = cls()
        for path in sorted(directory.glob("*.md")):
            raw = path.read_text(encoding="utf-8")
            meta, body = _split_frontmatter(raw)
            title = meta.get("title") or _first_heading(body) or path.stem
            category = meta.get("category", "general")
            team = meta.get("team", "一线服务组")
            doc_id = meta.get("id", path.stem)
            for index, paragraph in enumerate(_paragraphs(body)):
                text = paragraph.strip()
                if not text or text.startswith("#"):
                    continue
                kb.chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}#c{index + 1}",
                        doc_id=doc_id,
                        title=title,
                        category=category,
                        team=team,
                        text=text,
                        source=str(path.relative_to(directory.parent)),
                    )
                )
        kb._build_index()
        return kb

    def _build_index(self) -> None:
        self._term_frequency = []
        self._doc_frequency = {}
        lengths: list[int] = []
        for chunk in self.chunks:
            tokens = tokenize(chunk.title + " " + chunk.text)
            tf: dict[str, int] = {}
            for token in tokens:
                tf[token] = tf.get(token, 0) + 1
            self._term_frequency.append(tf)
            lengths.append(len(tokens))
            for token in tf:
                self._doc_frequency[token] = self._doc_frequency.get(token, 0) + 1
        self._avg_length = sum(lengths) / max(len(lengths), 1)

    def search(
        self,
        query: str,
        category: str | None = None,
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        terms = unique_terms(tokenize(query))
        if not terms:
            return []
        total_docs = max(len(self.chunks), 1)
        scored: list[RetrievedChunk] = []
        for index, chunk in enumerate(self.chunks):
            tf = self._term_frequency[index]
            length = sum(tf.values()) or 1
            score = 0.0
            matched_weight = 0.0
            total_weight = sum(term_weight(term) for term in terms) or 1.0
            for term in terms:
                frequency = tf.get(term, 0)
                if not frequency:
                    continue
                df = self._doc_frequency.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denominator = frequency + 1.2 * (
                    0.25 + 0.75 * length / max(self._avg_length, 1)
                )
                score += idf * (frequency * 2.2) / denominator
                matched_weight += term_weight(term)
            if category and chunk.category == category:
                score *= 1.28
            if score > 0:
                coverage = min(1.0, matched_weight / total_weight)
                if coverage >= 0.12:
                    scored.append(
                        RetrievedChunk(chunk=chunk, score=score, coverage=coverage)
                    )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]

    def get(self, doc_id: str) -> list[Chunk]:
        return [chunk for chunk in self.chunks if chunk.doc_id == doc_id]


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"').strip("'")
    return meta, text[match.end() :]


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _paragraphs(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip()
    ]


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        if raw in STOP_TERMS:
            continue
        if _is_chinese(raw):
            clean = raw
            for stop_term in sorted(STOP_TERMS, key=len, reverse=True):
                clean = clean.replace(stop_term, "")
            if not clean:
                continue
            chars = list(clean)
            tokens.extend(chars)
            tokens.extend(clean[i : i + 2] for i in range(len(clean) - 1))
        else:
            tokens.append(raw)
    return tokens


def unique_terms(tokens: list[str]) -> list[str]:
    # Preserve order for deterministic scoring.
    return list(dict.fromkeys(tokens))


def term_weight(term: str) -> float:
    if _is_chinese(term):
        return 2.0 if len(term) >= 2 else 0.7
    return 1.5


def _is_chinese(value: str) -> bool:
    return bool(value) and all("\u4e00" <= char <= "\u9fff" for char in value)


def best_sentences(chunks: list[RetrievedChunk], query: str, limit: int = 3) -> list[str]:
    """Select short evidence sentences that overlap the query."""
    query_terms = set(tokenize(query))
    candidates: list[tuple[float, str]] = []
    for item in chunks:
        for sentence in SENTENCE_RE.split(item.chunk.text):
            sentence = re.sub(r"^[-*\d.、\s]+", "", sentence).strip()
            if not sentence:
                continue
            terms = set(tokenize(sentence))
            overlap = len(query_terms & terms)
            bonus = 1 if item is chunks[0] else 0
            candidates.append((overlap + bonus + min(len(sentence), 80) / 200, sentence))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    selected: list[str] = []
    for _, sentence in candidates:
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) >= limit:
            break
    return selected

