from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import AgentConfig


def new_session_id() -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


class SessionLog:
    def __init__(self, config: AgentConfig, session_id: str) -> None:
        self.config = config
        self.session_id = session_id
        self.path = config.session_dir / f"{session_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: str, payload: dict[str, Any]) -> None:
        row = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "event": event,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


def list_sessions(config: AgentConfig) -> list[Path]:
    if not config.session_dir.exists():
        return []
    return sorted(config.session_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
