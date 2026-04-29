from __future__ import annotations

from .llm import LLMClient
from .models import EvaluationReport, EvidencePack


def evaluate_answer(
    question: str,
    answer: str,
    evidence_pack: EvidencePack,
    llm: LLMClient | None = None,
) -> EvaluationReport:
    if llm and llm.enabled:
        report = _evaluate_with_llm(question, answer, evidence_pack, llm)
        if report:
            return report
    return _evaluate_offline(question, answer, evidence_pack)


def _evaluate_offline(question: str, answer: str, evidence_pack: EvidencePack) -> EvaluationReport:
    answer = answer.strip()
    lower = answer.lower()
    length_score = 1 if len(answer) < 60 else 3 if len(answer) < 180 else 4
    structure = 4 if any(marker in answer for marker in ["1.", "首先", "其次", "最后", "：", ":"]) else 2
    engineering = 4 if any(word in lower for word in ["指标", "延迟", "召回", "评估", "数据库", "状态", "工程", "trade-off", "取舍"]) else 2
    tradeoff = 4 if any(word in lower for word in ["取舍", "trade", "优点", "缺点", "风险", "代价"]) else 2
    grounding = 4 if evidence_pack.evidence and any(ev.title[:4] in answer for ev in evidence_pack.evidence[:3]) else 2
    anti_followup = 4 if any(word in lower for word in ["失败", "兜底", "降级", "边界", "鲁棒"]) else 2
    correctness = min(5, max(1, round((length_score + engineering + grounding) / 3)))
    missing = []
    if structure < 4:
        missing.append("回答缺少结构化框架")
    if engineering < 4:
        missing.append("缺少工程细节和指标")
    if tradeoff < 4:
        missing.append("缺少取舍分析")
    if grounding < 4:
        missing.append("缺少知识库证据引用")
    if anti_followup < 4:
        missing.append("缺少失败模式或追问防御")
    return EvaluationReport(
        correctness=correctness,
        structure=structure,
        engineering_depth=engineering,
        tradeoff_quality=tradeoff,
        source_grounding=grounding,
        anti_followup=anti_followup,
        missing_points=missing,
        better_answer=build_better_answer(question, evidence_pack),
        next_tasks=[f"围绕「{point}」补一版 2 分钟回答" for point in missing] or ["保持当前题型练习"],
    )


def _evaluate_with_llm(
    question: str,
    answer: str,
    evidence_pack: EvidencePack,
    llm: LLMClient,
) -> EvaluationReport | None:
    payload = llm.complete_json(
        system=(
            "你是严苛的 AI Agent/RAG 面试官。请只输出 JSON，不要 Markdown。"
            "所有分数必须是 1 到 5 的整数。"
        ),
        user=(
            f"面试问题：{question}\n\n"
            f"候选人回答：{answer}\n\n"
            f"可用证据：\n{_format_evidence(evidence_pack)}\n\n"
            "请按这个 JSON schema 输出：\n"
            "{\n"
            '  "correctness": 1,\n'
            '  "structure": 1,\n'
            '  "engineering_depth": 1,\n'
            '  "tradeoff_quality": 1,\n'
            '  "source_grounding": 1,\n'
            '  "anti_followup": 1,\n'
            '  "missing_points": ["..."],\n'
            '  "better_answer": "...",\n'
            '  "next_tasks": ["..."]\n'
            "}\n\n"
            "字段要求：\n"
            "- missing_points 和 next_tasks 必须是短句数组。\n"
            "- better_answer 必须是适合写入 .md 的多行文本，使用 1. 2. 3. 编号分段。\n"
            "- better_answer 不要写成一整段，必须覆盖结论、关键机制、工程取舍和评估/兜底。"
        ),
        max_output_tokens=4096,
    )
    if not payload:
        return None
    try:
        return EvaluationReport(
            correctness=_clamp_score(payload.get("correctness")),
            structure=_clamp_score(payload.get("structure")),
            engineering_depth=_clamp_score(payload.get("engineering_depth")),
            tradeoff_quality=_clamp_score(payload.get("tradeoff_quality")),
            source_grounding=_clamp_score(payload.get("source_grounding")),
            anti_followup=_clamp_score(payload.get("anti_followup")),
            missing_points=[str(v) for v in payload.get("missing_points", [])][:8],
            better_answer=str(payload.get("better_answer") or build_better_answer(question, evidence_pack)),
            next_tasks=[str(v) for v in payload.get("next_tasks", [])][:8] or ["保持当前题型练习"],
        )
    except Exception:
        return None


def build_better_answer(question: str, evidence_pack: EvidencePack) -> str:
    sources = "、".join(ev.title for ev in evidence_pack.evidence[:3]) or "本地知识库"
    return (
        f"1. 先给直接结论：围绕「{question}」明确系统目标、适用边界和核心判断。\n"
        f"2. 展开架构或流程：引用 {sources} 中的证据，说明关键模块、数据流和执行链路。\n"
        "3. 补充工程取舍：说明成本、延迟、准确率、可维护性和失败兜底方案。\n"
        "4. 给出评估方式：覆盖离线指标、线上观测、错误归因和下一步迭代动作。"
    )


def _format_evidence(evidence_pack: EvidencePack, limit: int = 5) -> str:
    lines = []
    for idx, ev in enumerate(evidence_pack.evidence[:limit], start=1):
        lines.append(f"[{idx}] {ev.title} | {ev.source_file} | {ev.snippet}")
    return "\n".join(lines)


def _clamp_score(value: object) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, min(5, score))
