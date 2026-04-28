from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path

from .models import Chunk, DocumentMeta


class KnowledgeStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    source_url TEXT,
                    author TEXT,
                    publish_date TEXT,
                    tags_json TEXT NOT NULL,
                    topics_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source_file TEXT NOT NULL,
                    source_url TEXT,
                    topics_json TEXT NOT NULL,
                    text TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    title,
                    text,
                    topics
                );
                """
            )

    def clear(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks_fts")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")

    def upsert_documents(self, docs: Iterable[DocumentMeta], chunks: Iterable[Chunk]) -> None:
        with self.connect() as conn:
            for doc in docs:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO documents
                    (doc_id, title, source_file, source_url, author, publish_date, tags_json, topics_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc.doc_id,
                        doc.title,
                        doc.source_file,
                        doc.source_url,
                        doc.author,
                        doc.publish_date,
                        json.dumps(doc.tags, ensure_ascii=False),
                        json.dumps(doc.topics, ensure_ascii=False),
                    ),
                )
            for chunk in chunks:
                topics = json.dumps(chunk.topics, ensure_ascii=False)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, doc_id, title, source_file, source_url, topics_json, text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.title,
                        chunk.source_file,
                        chunk.source_url,
                        topics,
                        chunk.text,
                    ),
                )
                conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, title, text, topics) VALUES (?, ?, ?, ?)",
                    (chunk.chunk_id, chunk.title, chunk.text, " ".join(chunk.topics)),
                )

    def search(self, query: str, limit: int = 8) -> list[sqlite3.Row]:
        terms = [term for term in _tokenize_query(query) if len(term) > 1]
        if not terms:
            return []
        fts_query = " OR ".join(terms[:12])
        try:
            return self._fts_search(fts_query, limit)
        except sqlite3.OperationalError:
            return self._like_search(terms, limit)

    def _fts_search(self, fts_query: str, limit: int) -> list[sqlite3.Row]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*, bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return rows

    def metadata_search(self, terms: list[str], limit: int = 40, source_prefix: str | None = None) -> list[sqlite3.Row]:
        terms = [term for term in terms if len(term) > 1][:16]
        if not terms:
            return []
        clauses = []
        params: list[str | int] = []
        for term in terms:
            clauses.append("(title LIKE ? OR source_file LIKE ? OR text LIKE ?)")
            params.extend([f"%{term}%", f"%{term}%", f"%{term}%"])
        where = " OR ".join(clauses)
        if source_prefix:
            where = f"source_file LIKE ? AND ({where})"
            params.insert(0, f"{source_prefix}%")
        params.append(limit)
        with self.connect() as conn:
            return conn.execute(f"SELECT *, 0 AS rank FROM chunks WHERE {where} LIMIT ?", params).fetchall()

    def _like_search(self, terms: list[str], limit: int) -> list[sqlite3.Row]:
        where = " OR ".join(["text LIKE ? OR title LIKE ?" for _ in terms])
        params: list[str | int] = []
        for term in terms:
            params.extend([f"%{term}%", f"%{term}%"])
        params.append(limit)
        with self.connect() as conn:
            rows = conn.execute(f"SELECT *, 0 AS rank FROM chunks WHERE {where} LIMIT ?", params).fetchall()
        return rows


def _tokenize_query(query: str) -> list[str]:
    import re

    chinese = re.findall(r"[\u4e00-\u9fff]{2,}", query)
    latin = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", query)
    tokens = chinese + latin
    extra: list[str] = []
    for token in chinese:
        if len(token) > 4:
            extra.extend(token[i : i + 2] for i in range(0, len(token) - 1))
    return tokens + extra
