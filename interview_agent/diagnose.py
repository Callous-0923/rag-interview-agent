from __future__ import annotations

from pathlib import Path

from .config import AgentConfig
from .retrieval import AgenticRetriever


DEFAULT_TOPICS = ["Agent", "RAG", "Harness", "大模型", "Python", "工程开发", "AI应用"]


def diagnose(config: AgentConfig, role: str, jd: Path | None = None, resume: Path | None = None) -> str:
    jd_text = jd.read_text(encoding="utf-8", errors="replace") if jd and jd.exists() else ""
    resume_text = resume.read_text(encoding="utf-8", errors="replace") if resume and resume.exists() else ""
    retriever = AgenticRetriever(config)
    topic_scores: list[tuple[str, int]] = []
    combined = f"{role}\n{jd_text}\n{resume_text}"
    for topic in DEFAULT_TOPICS:
        signal = int(topic.lower() in combined.lower() or topic in combined)
        evidence = retriever.retrieve(topic)
        topic_scores.append((topic, signal + min(len(evidence.evidence), 5)))
    topic_scores.sort(key=lambda item: item[1], reverse=True)
    weak = [topic for topic, _ in topic_scores[:4]]
    lines = [
        f"# 诊断报告：{role}",
        "",
        "## 高优先级主题",
    ]
    lines.extend(f"- {topic}" for topic in weak)
    lines.extend(
        [
            "",
            "## 建议训练路径",
            "1. 先用 `agent ask` 补齐概念和架构答案。",
            "2. 再用 `agent mock --topic <主题>` 做追问训练。",
            "3. 每轮 `agent review` 后把短板写入长期记忆。",
        ]
    )
    out = config.memory_dir / "last_diagnosis.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return "\n".join(lines)
