from __future__ import annotations

import json
import math
import re
import hashlib
from collections.abc import Iterable

from .config import AgentConfig
from .models import Chunk


def sync_chroma(config: AgentConfig, chunks: Iterable[Chunk], reset: bool = True) -> dict[str, object]:
    if not config.chroma_dir:
        return {"enabled": False, "reason": "chroma_dir is not configured"}
    try:
        import chromadb
    except ImportError:
        return {"enabled": False, "reason": "chromadb is not installed"}

    config.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.chroma_dir))
    if reset:
        try:
            client.delete_collection(config.chroma_collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(config.chroma_collection)

    batch_ids: list[str] = []
    batch_docs: list[str] = []
    batch_meta: list[dict[str, str]] = []
    batch_embeddings: list[list[float]] = []
    count = 0
    for chunk in chunks:
        batch_ids.append(chunk.chunk_id)
        batch_docs.append(chunk.text)
        batch_embeddings.append(_hash_embedding(chunk.text))
        batch_meta.append(
            {
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "source_file": chunk.source_file,
                "source_url": chunk.source_url,
                "topics_json": json.dumps(chunk.topics, ensure_ascii=False),
            }
        )
        if len(batch_ids) >= 128:
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=batch_embeddings)
            count += len(batch_ids)
            batch_ids, batch_docs, batch_meta, batch_embeddings = [], [], [], []
    if batch_ids:
        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=batch_embeddings)
        count += len(batch_ids)

    return {
        "enabled": True,
        "chunks": count,
        "path": str(config.chroma_dir),
        "collection": config.chroma_collection,
    }


def search_chroma(config: AgentConfig, query: str, limit: int = 8) -> list[dict[str, object]]:
    if not config.chroma_dir or not config.chroma_dir.exists():
        return []
    try:
        import chromadb
    except ImportError:
        return []
    try:
        client = chromadb.PersistentClient(path=str(config.chroma_dir))
        collection = client.get_collection(config.chroma_collection)
        result = collection.query(query_embeddings=[_hash_embedding(query)], n_results=limit)
    except Exception:
        return []

    ids = (result.get("ids") or [[]])[0]
    docs = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    rows: list[dict[str, object]] = []
    for idx, chunk_id in enumerate(ids):
        meta = metadatas[idx] if idx < len(metadatas) else {}
        rows.append(
            {
                "chunk_id": chunk_id,
                "text": docs[idx] if idx < len(docs) else "",
                "title": meta.get("title", ""),
                "source_file": meta.get("source_file", ""),
                "source_url": meta.get("source_url", ""),
                "distance": distances[idx] if idx < len(distances) else 1.0,
            }
        )
    return rows


def _hash_embedding(text: str, dims: int = 384) -> list[float]:
    vector = [0.0] * dims
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]+", text.lower())
    for token in tokens:
        idx = _stable_index(token, dims)
        vector[idx] += 1.0
        if len(token) > 4:
            for i in range(0, len(token) - 1):
                vector[_stable_index(token[i : i + 2], dims)] += 0.3
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _stable_index(token: str, dims: int) -> int:
    return int(hashlib.sha1(token.encode("utf-8")).hexdigest()[:8], 16) % dims
