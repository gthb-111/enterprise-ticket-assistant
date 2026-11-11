from __future__ import annotations

import hashlib
import io
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from docx import Document
from pypdf import PdfReader

from .knowledge import Chunk


MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".pdf", ".docx"}
MAX_CHUNK_CHARS = 700
CHUNK_OVERLAP_CHARS = 80

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    title TEXT NOT NULL,
    category TEXT NOT NULL,
    team TEXT NOT NULL,
    source TEXT NOT NULL,
    media_type TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    checksum TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document
ON document_chunks(document_id, position);
"""


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    title: str
    category: str
    team: str
    media_type: str
    content: str
    chunks: list[str]


class DocumentStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create(self, document: ParsedDocument) -> dict[str, Any]:
        document_id = f"DOC-{uuid.uuid4().hex[:10].upper()}"
        created_at = datetime.now(timezone.utc).isoformat()
        source = f"upload://{document.filename}"
        checksum = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    id, filename, title, category, team, source, media_type,
                    size_bytes, checksum, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    document.filename,
                    document.title,
                    document.category,
                    document.team,
                    source,
                    document.media_type,
                    len(document.content.encode("utf-8")),
                    checksum,
                    created_at,
                ),
            )
            conn.executemany(
                """
                INSERT INTO document_chunks (
                    chunk_id, document_id, position, text
                ) VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        f"{document_id}#c{index + 1}",
                        document_id,
                        index,
                        chunk,
                    )
                    for index, chunk in enumerate(document.chunks)
                ],
            )
        return self.get(document_id) or {}

    def get(self, document_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
            if row is None:
                return None
            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM document_chunks WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
        item = dict(row)
        item["chunk_count"] = chunk_count
        return item

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT d.*, COUNT(c.chunk_id) AS chunk_count
                FROM documents d
                LEFT JOIN document_chunks c ON c.document_id = d.id
                GROUP BY d.id
                ORDER BY d.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self) -> list[Chunk]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.chunk_id, c.document_id, c.text,
                    d.title, d.category, d.team, d.source
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY d.created_at, c.position
                """
            ).fetchall()
        return [
            Chunk(
                chunk_id=row["chunk_id"],
                doc_id=row["document_id"],
                title=row["title"],
                category=row["category"],
                team=row["team"],
                text=row["text"],
                source=row["source"],
            )
            for row in rows
        ]


def parse_document(
    filename: str,
    raw: bytes,
    media_type: str,
    title: str = "",
    category: str = "general",
    team: str = "一线服务组",
) -> ParsedDocument:
    safe_filename = Path(filename or "document.txt").name
    suffix = Path(safe_filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = "、".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"当前仅支持 {supported} 文件")
    if not raw:
        raise ValueError("上传文件不能为空")
    if len(raw) > MAX_DOCUMENT_BYTES:
        raise ValueError("上传文件不能超过 2 MB")

    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(raw))
            content = "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            )
        except Exception as exc:
            raise ValueError("PDF 文件无法解析") from exc
    elif suffix == ".docx":
        try:
            document = Document(io.BytesIO(raw))
            blocks = [paragraph.text for paragraph in document.paragraphs]
            for table in document.tables:
                for row in table.rows:
                    blocks.append(" | ".join(cell.text for cell in row.cells))
            content = "\n\n".join(blocks)
        except Exception as exc:
            raise ValueError("DOCX 文件无法解析") from exc
    else:
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                content = raw.decode("gb18030")
            except UnicodeDecodeError as exc:
                raise ValueError("文件必须使用 UTF-8 或 GB18030 编码") from exc

    content = content.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
    content = re.sub(r"\n{4,}", "\n\n\n", content).strip()
    if not content:
        raise ValueError("文件中没有可索引的文本")

    clean_title = title.strip()[:100] or _first_heading(content) or Path(
        safe_filename
    ).stem
    clean_category = re.sub(r"[^a-z_]", "", category.lower()) or "general"
    clean_team = team.strip()[:50] or "一线服务组"
    chunks = split_text(content)
    if not chunks:
        raise ValueError("文件中没有可索引的文本")
    return ParsedDocument(
        filename=safe_filename,
        title=clean_title,
        category=clean_category,
        team=clean_team,
        media_type=media_type or "text/plain",
        content=content,
        chunks=chunks,
    )


def split_text(text: str) -> list[str]:
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n", text)
        if part.strip() and not part.strip().startswith("# ")
    ]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        pieces = (
            [paragraph]
            if len(paragraph) <= MAX_CHUNK_CHARS
            else _split_long_paragraph(paragraph)
        )
        for piece in pieces:
            if current and len(current) + len(piece) + 2 > MAX_CHUNK_CHARS:
                chunks.append(current)
                current = _overlap_tail(current) + piece
            else:
                current = f"{current}\n\n{piece}".strip()
    if current:
        chunks.append(current)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _split_long_paragraph(paragraph: str) -> list[str]:
    sentences = re.split(r"(?<=[。！？!?；;])\s*", paragraph)
    pieces: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if len(sentence) > MAX_CHUNK_CHARS:
            while len(sentence) > MAX_CHUNK_CHARS:
                pieces.append(sentence[:MAX_CHUNK_CHARS])
                sentence = sentence[MAX_CHUNK_CHARS - CHUNK_OVERLAP_CHARS :]
            current = sentence
            continue
        if current and len(current) + len(sentence) > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = _overlap_tail(current) + sentence
        else:
            current += sentence
    if current:
        pieces.append(current)
    return pieces


def _overlap_tail(text: str) -> str:
    tail = text[-CHUNK_OVERLAP_CHARS:]
    return tail + "\n\n"


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None
