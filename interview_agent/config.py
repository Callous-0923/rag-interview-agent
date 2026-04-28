from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AgentConfig:
    project_root: Path
    knowledge_root: Path
    notes_dir: Path
    notes_dirs: list[Path]
    topic_index: Path
    topic_assignments: Path
    graph_json: Path
    graph_jsons: list[Path]
    storage_dir: Path
    session_dir: Path
    memory_dir: Path
    skills_dir: Path
    vision_dir: Path | None = None
    vision_enabled: bool = True
    vision_usage_log: Path | None = None
    chroma_dir: Path | None = None
    chroma_collection: str = "interview_agent_chunks"
    chunk_size: int = 900
    chunk_overlap: int = 120
    max_rewrites: int = 2
    top_k: int = 8
    llm: dict[str, Any] | None = None

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "interview_agent.sqlite"


def _path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def load_config(config_path: str | Path | None = None) -> AgentConfig:
    path = Path(config_path) if config_path else DEFAULT_ROOT / "config.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    notes_dir = _path(raw["notes_dir"])
    notes_dirs = [_path(value) for value in raw.get("notes_dirs", [notes_dir])]
    graph_json = _path(raw["graph_json"])
    graph_jsons = [_path(value) for value in raw.get("graph_jsons", [graph_json])]
    project_root = _path(raw.get("project_root", DEFAULT_ROOT))
    storage_dir = _path(raw.get("storage_dir", DEFAULT_ROOT / "storage"))
    vision_raw = raw.get("vision") or {}
    vision_dir = _path(raw.get("vision_dir", project_root / "vision"))
    vision_usage_log = _path(vision_raw.get("usage_log", vision_dir / "usage.jsonl"))
    return AgentConfig(
        project_root=project_root,
        knowledge_root=_path(raw.get("knowledge_root", DEFAULT_ROOT.parent)),
        notes_dir=notes_dir,
        notes_dirs=notes_dirs,
        topic_index=_path(raw["topic_index"]),
        topic_assignments=_path(raw["topic_assignments"]),
        graph_json=graph_json,
        graph_jsons=graph_jsons,
        storage_dir=storage_dir,
        session_dir=_path(raw.get("session_dir", DEFAULT_ROOT / "sessions")),
        memory_dir=_path(raw.get("memory_dir", DEFAULT_ROOT / "memory")),
        skills_dir=_path(raw.get("skills_dir", DEFAULT_ROOT / "skills")),
        vision_dir=vision_dir,
        vision_enabled=bool(raw.get("vision_enabled", vision_raw.get("enabled", True))),
        vision_usage_log=vision_usage_log,
        chroma_dir=_path(raw.get("chroma_dir", storage_dir / "chroma")),
        chroma_collection=str(raw.get("chroma_collection", "interview_agent_chunks")),
        chunk_size=int(raw.get("chunk_size", 900)),
        chunk_overlap=int(raw.get("chunk_overlap", 120)),
        max_rewrites=int(raw.get("max_rewrites", 2)),
        top_k=int(raw.get("top_k", 8)),
        llm=raw.get("llm") or {},
    )
