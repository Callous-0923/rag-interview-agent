from __future__ import annotations

from pathlib import Path

from .config import AgentConfig
from .markdown import parse_markdown, split_chunks
from .store import KnowledgeStore
from .vector_store import sync_chroma


def ingest_notes(config: AgentConfig, reset: bool = True) -> dict[str, int]:
    store = KnowledgeStore(config.db_path)
    store.init_schema()
    if reset:
        store.clear()

    docs = []
    chunks = []
    for notes_dir in config.notes_dirs:
        for path in sorted(notes_dir.rglob("*.md")):
            meta, body = parse_markdown(path, notes_dir)
            docs.append(meta)
            chunks.extend(split_chunks(meta, body, config.chunk_size, config.chunk_overlap))

    store.upsert_documents(docs, chunks)
    chroma = sync_chroma(config, chunks, reset=reset)
    return {"documents": len(docs), "chunks": len(chunks), "db": str(config.db_path), "chroma": chroma}


def ensure_workspace(config: AgentConfig) -> None:
    for path in [
        config.storage_dir,
        config.session_dir,
        config.memory_dir,
        config.vision_dir or (config.project_root / "vision"),
        (config.vision_dir or (config.project_root / "vision")) / "xhs_image_notes",
        (config.vision_dir or (config.project_root / "vision")) / "cache",
        config.skills_dir / "pending",
        config.skills_dir / "active",
        config.project_root / "profile",
        config.project_root / "jd",
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
    _ensure_file(config.memory_dir / "weakness_map.md", "# Weakness Map\n\n")
    _ensure_file(config.memory_dir / "user_profile.md", "# User Profile\n\n")
    _ensure_file(config.memory_dir / "topic_mastery.json", "{}\n")
    _ensure_file(config.memory_dir / "growth_metrics.json", "{}\n")
    _ensure_file(config.memory_dir / "review_schedule.json", "{}\n")
    _ensure_file(config.memory_dir / "learning_gaps.json", "{}\n")
    _ensure_file(config.memory_dir / "answer_history.jsonl", "")


def _ensure_file(path: Path, content: str) -> None:
    if not path.exists():
        path.write_text(content, encoding="utf-8", newline="\n")
