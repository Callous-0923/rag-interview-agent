from __future__ import annotations

import re
from pathlib import Path

from .config import AgentConfig


class SkillManager:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.pending_dir = config.skills_dir / "pending"
        self.active_dir = config.skills_dir / "active"
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        self.active_dir.mkdir(parents=True, exist_ok=True)

    def list_skills(self, status: str = "active") -> list[Path]:
        root = self.active_dir if status == "active" else self.pending_dir
        return sorted(root.glob("*/SKILL.md"))

    def show(self, name: str, status: str = "active") -> str:
        root = self.active_dir if status == "active" else self.pending_dir
        path = root / name / "SKILL.md"
        if not path.exists():
            raise FileNotFoundError(f"Skill not found: {path}")
        return path.read_text(encoding="utf-8")

    def select(self, topic: str, question: str, limit: int = 3) -> list[str]:
        haystack = f"{topic} {question}".lower()
        selected: list[str] = []
        for path in self.list_skills("active"):
            text = path.read_text(encoding="utf-8")
            name = path.parent.name
            if name.lower() in haystack or any(keyword.lower() in haystack for keyword in _extract_keywords(text)):
                selected.append(text)
            if len(selected) >= limit:
                break
        return selected

    def suggest_from_weaknesses(self, topic: str, weaknesses: list[str]) -> list[Path]:
        created: list[Path] = []
        for weakness in weaknesses:
            slug = _slugify(f"{topic}_{weakness}")[:60]
            skill_dir = self.pending_dir / slug
            path = skill_dir / "SKILL.md"
            if path.exists():
                continue
            skill_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(_skill_template(topic, weakness), encoding="utf-8", newline="\n")
            created.append(path)
        return created

    def activate(self, name: str) -> Path:
        src = self.pending_dir / name / "SKILL.md"
        if not src.exists():
            raise FileNotFoundError(f"Pending skill not found: {src}")
        dst_dir = self.active_dir / name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "SKILL.md"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        return dst


def _extract_keywords(text: str) -> list[str]:
    lines = [line for line in text.splitlines() if "触发" in line or "适用" in line]
    return re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]+", " ".join(lines))


def _slugify(value: str) -> str:
    value = re.sub(r"\s+", "_", value.strip().lower())
    value = re.sub(r"[^\w\u4e00-\u9fff_-]+", "", value)
    return value or "skill"


def _skill_template(topic: str, weakness: str) -> str:
    return f"""# {topic}：{weakness}

## 触发条件
- 面试题主题包含 `{topic}`
- 复盘短板命中：{weakness}

## 适用题型
- 概念解释
- 系统设计
- 项目深挖
- 追问防御

## 答题结构
1. 先给一句直接结论。
2. 按“背景 -> 方案 -> 取舍 -> 指标 -> 失败兜底”展开。
3. 主动引用本地知识库证据，不凭空编造。
4. 最后补一个可被追问的工程细节。

## 反例
- 只背概念，没有说明为什么这样设计。
- 没有指标、边界和失败处理。
- 没有结合项目或知识库证据。

## 可复用 Prompt
请把我的回答改写成面试表达，重点修复“{weakness}”，并补充工程指标、取舍和失败兜底。

## 示例答案
这个问题我会先拆成目标、架构、取舍和评估四部分回答。核心结论是：方案不能只追求模型回答流畅，而要通过证据检索、状态记忆、评估指标和失败兜底保证可复现。
"""
