from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .config import AgentConfig
from .models import EvaluationReport


class MemoryManager:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.config.memory_dir.mkdir(parents=True, exist_ok=True)
        self.weakness_path = self.config.memory_dir / "weakness_map.md"
        self.mastery_path = self.config.memory_dir / "topic_mastery.json"

    def load_topic_mastery(self) -> dict[str, dict]:
        if not self.mastery_path.exists():
            return {}
        return json.loads(self.mastery_path.read_text(encoding="utf-8") or "{}")

    def update_from_evaluation(self, topic: str, report: EvaluationReport) -> list[str]:
        missing = report.missing_points or ["表达结构和工程细节不足"]
        mastery = self.load_topic_mastery()
        item = mastery.setdefault(topic, {"attempts": 0, "avg_score": 0.0, "weaknesses": {}})
        attempts = int(item["attempts"]) + 1
        item["avg_score"] = round(((float(item["avg_score"]) * (attempts - 1)) + report.average) / attempts, 2)
        item["attempts"] = attempts
        weaknesses = Counter(item.get("weaknesses", {}))
        for weakness in missing:
            weaknesses[weakness] += 1
        item["weaknesses"] = dict(weaknesses)
        self.mastery_path.write_text(json.dumps(mastery, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
        self._append_weakness_markdown(topic, report)
        return [weakness for weakness, count in weaknesses.items() if count >= 3]

    def build_working_summary(self, questions: list[str], answers: list[str]) -> str:
        pairs = list(zip(questions[-3:], answers[-3:]))
        if not pairs:
            return ""
        lines = ["最近问答摘要："]
        for question, answer in pairs:
            lines.append(f"- 问：{question[:80]}；答：{answer[:120]}")
        return "\n".join(lines)

    def _append_weakness_markdown(self, topic: str, report: EvaluationReport) -> None:
        if not self.weakness_path.exists():
            self.weakness_path.write_text("# Weakness Map\n\n", encoding="utf-8")
        lines = [
            f"## {topic}",
            f"- average_score: {report.average}",
            f"- missing_points: {'；'.join(report.missing_points) if report.missing_points else '无'}",
            f"- next_tasks: {'；'.join(report.next_tasks) if report.next_tasks else '继续练习'}",
            "",
        ]
        with self.weakness_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines))
