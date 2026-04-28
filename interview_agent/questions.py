from __future__ import annotations

from .models import EvidencePack


QUESTION_TEMPLATES = {
    "RAG": [
        "请完整讲一下 RAG 的知识库构建流程，你会如何做 chunk、索引和更新？",
        "传统 RAG 和 Agentic RAG 的区别是什么？什么时候需要 query rewrite？",
        "如果 RAG 回答错了，你怎么判断是检索问题还是生成问题？",
        "GraphRAG 解决了什么问题？它在增量更新时有什么难点？",
        "RAG 系统的评估体系怎么做？哪些指标最关键？",
    ],
    "Agent": [
        "Agent 和普通 LLM App 的本质区别是什么？",
        "你会如何设计 Agent 的记忆系统？",
        "Tool Use、Planning、Memory、RAG 在 Agent 架构里如何协作？",
        "如何避免 Agent 上下文膨胀和历史污染？",
        "多 Agent 什么时候值得用，什么时候应该避免？",
    ],
    "Hermes": [
        "Hermes Agent 和 OpenClaw 的核心区别是什么？",
        "Hermes 风格的自动技能沉淀如何避免错误经验污染？",
        "如何设计 skill 失效检测和自我修复机制？",
    ],
}


def generate_question(topic: str, round_index: int, evidence_pack: EvidencePack | None = None) -> str:
    key = _normalize_topic(topic)
    templates = QUESTION_TEMPLATES.get(key, QUESTION_TEMPLATES["Agent"])
    question = templates[round_index % len(templates)]
    if evidence_pack and evidence_pack.topics:
        return f"{question} 请结合 {', '.join(evidence_pack.topics[:3])} 相关资料回答。"
    return question


def _normalize_topic(topic: str) -> str:
    lowered = topic.lower()
    if "rag" in lowered or "检索" in topic:
        return "RAG"
    if "hermes" in lowered:
        return "Hermes"
    return "Agent"
