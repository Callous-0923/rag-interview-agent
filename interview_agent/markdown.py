from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

from .models import Chunk, DocumentMeta


FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


def parse_markdown(path: Path, notes_dir: Path) -> tuple[DocumentMeta, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = FRONTMATTER_RE.match(text)
    meta: dict[str, Any] = {}
    body = text
    if match:
        meta = yaml.safe_load(match.group(1)) or {}
        body = text[match.end() :]
    rel = path.relative_to(notes_dir.parent).as_posix()
    title = str(meta.get("title") or meta.get("source_title") or path.stem)
    doc_id = str(meta.get("doc_id") or meta.get("xhs_note_id") or hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16])
    tags = [str(v) for v in meta.get("tags", [])] if isinstance(meta.get("tags"), list) else []
    raw_topics = meta.get("topic", meta.get("topics", []))
    topics = [str(v) for v in raw_topics] if isinstance(raw_topics, list) else []
    return (
        DocumentMeta(
            doc_id=doc_id,
            title=title,
            source_file=rel,
            source_url=str(meta.get("xhs_url") or meta.get("source_url") or ""),
            author=str(meta.get("author") or ""),
            publish_date=str(meta.get("publish_date") or meta.get("date") or meta.get("captured_at") or ""),
            tags=tags,
            topics=topics,
        ),
        clean_body(body),
    )


def clean_body(body: str) -> str:
    body = re.sub(r"!\[Image\]\([^)]+\)", "", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def split_chunks(meta: DocumentMeta, body: str, chunk_size: int, overlap: int) -> list[Chunk]:
    sections = _split_by_heading(body)
    chunks: list[Chunk] = []
    idx = 0
    for section in sections:
        start = 0
        while start < len(section):
            piece = section[start : start + chunk_size].strip()
            if piece:
                chunk_id = f"{meta.doc_id}:{idx}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=meta.doc_id,
                        title=meta.title,
                        source_file=meta.source_file,
                        source_url=meta.source_url,
                        topics=meta.topics,
                        text=piece,
                    )
                )
                idx += 1
            if start + chunk_size >= len(section):
                break
            start += max(1, chunk_size - overlap)
    return chunks


def _split_by_heading(body: str) -> list[str]:
    parts = re.split(r"(?=^#{1,3}\s+)", body, flags=re.MULTILINE)
    return [part.strip() for part in parts if part.strip()]
