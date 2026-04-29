from __future__ import annotations

from .memory import normalize_topic
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
    "Harness": [
        "什么是 Agent Harness？它和普通 Agent 应用的区别是什么？",
        "JIT Context 为什么适合代码库问答和故障排查？",
        "你会如何设计一个可观测、可恢复的 Harness 执行轨迹？",
        "Harness 中工具编排、上下文装配和失败恢复如何协作？",
    ],
    "大模型": [
        "请解释 Transformer Encoder 的核心结构和 Attention 的计算流程。",
        "SFT、RLHF、DPO 的目标和适用场景分别是什么？",
        "长上下文模型在推理成本和效果上有哪些主要挑战？",
        "多模态模型做图文理解时，你会如何评估 OCR 和视觉推理能力？",
    ],
    "Python": [
        "Python 中 list、dict、set 的底层特点和常见复杂度是什么？",
        "你会如何用 Python 设计一个可测试的 CLI 工具？",
        "asyncio、线程、进程分别适合什么场景？",
        "请说明 Python 文件读写时如何处理编码、异常和大文件流式读取。",
    ],
}


def generate_question(topic: str, round_index: int, evidence_pack: EvidencePack | None = None) -> str:
    key = normalize_topic(topic)
    templates = QUESTION_TEMPLATES.get(key, QUESTION_TEMPLATES["Agent"])
    question = templates[round_index % len(templates)]
    if evidence_pack and evidence_pack.topics:
        return f"{question} 请结合 {', '.join(evidence_pack.topics[:3])} 相关资料回答。"
    return question
