from __future__ import annotations

from typing import Any, TypedDict

from .config import AgentConfig
from .llm import LLMClient
from .memory import MemoryManager, infer_question_type, normalize_difficulty, normalize_topic
from .questions import generate_question
from .retrieval import AgenticRetriever
from .scoring import evaluate_answer
from .sessions import SessionLog
from .skills import SkillManager


class GraphState(TypedDict, total=False):
    session_id: str
    topic: str
    round_index: int
    question: str
    answer: str
    evidence_pack: dict[str, Any]
    report: dict[str, Any]
    created_skills: list[str]
    selected_skills: list[str]
    working_summary: str
    knowledge_point: str
    question_type: str
    difficulty: str
    is_review: bool
    review_key: str
    memory_update: dict[str, Any]


def run_review_turn(config: AgentConfig, state: GraphState) -> GraphState:
    graph = build_graph(config)
    try:
        return graph.invoke(state)
    except AttributeError:
        return _run_review_turn_plain(config, state)


def build_graph(config: AgentConfig):
    try:
        from langgraph.graph import END, StateGraph
    except Exception:
        return PlainGraph(config)

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", lambda state: _node_retrieve(config, state))
    graph.add_node("evaluate", lambda state: _node_evaluate(config, state))
    graph.add_node("update_memory", lambda state: _node_update_memory(config, state))
    graph.add_node("suggest_skill", lambda state: _node_suggest_skill(config, state))
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "evaluate")
    graph.add_edge("evaluate", "update_memory")
    graph.add_edge("update_memory", "suggest_skill")
    graph.add_edge("suggest_skill", END)
    return graph.compile()


class PlainGraph:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def invoke(self, state: GraphState) -> GraphState:
        return _run_review_turn_plain(self.config, state)


def _run_review_turn_plain(config: AgentConfig, state: GraphState) -> GraphState:
    for node in [_node_retrieve, _node_evaluate, _node_update_memory, _node_suggest_skill]:
        state = node(config, state)
    return state


def _node_retrieve(config: AgentConfig, state: GraphState) -> GraphState:
    question = state.get("question") or generate_question(state.get("topic", "Agent"), state.get("round_index", 0))
    pack = AgenticRetriever(config).retrieve(question)
    state["question"] = question
    state["evidence_pack"] = pack.model_dump()
    return state


def _node_evaluate(config: AgentConfig, state: GraphState) -> GraphState:
    from .models import EvidencePack

    pack = EvidencePack.model_validate(state["evidence_pack"])
    report = evaluate_answer(state["question"], state.get("answer", ""), pack, LLMClient.from_config(config))
    state["report"] = report.model_dump()
    return state


def _node_update_memory(config: AgentConfig, state: GraphState) -> GraphState:
    from .models import EvaluationReport

    report = EvaluationReport.model_validate(state["report"])
    repeated = MemoryManager(config).update_from_evaluation(state.get("topic", "Agent"), report)
    state["repeated_weaknesses"] = repeated
    return state


def _node_suggest_skill(config: AgentConfig, state: GraphState) -> GraphState:
    repeated = state.get("repeated_weaknesses", [])
    created = SkillManager(config).suggest_from_weaknesses(state.get("topic", "Agent"), repeated)
    state["created_skills"] = [str(path) for path in created]
    return state


def run_mock_session(config: AgentConfig, session_id: str, topic: str, rounds: int) -> str:
    retriever = AgenticRetriever(config)
    llm = LLMClient.from_config(config)
    memory = MemoryManager(config)
    skills = SkillManager(config)
    log = SessionLog(config, session_id)
    questions: list[str] = []
    answers: list[str] = []
    log.append("session_start", {"topic": topic, "rounds": rounds})
    for idx in range(rounds):
        seed_pack = retriever.retrieve(topic)
        question = _llm_interview_question(topic, idx, seed_pack, llm) or generate_question(topic, idx, seed_pack)
        selected_skills = skills.select(topic, question)
        answer = _llm_answer_hint(question, seed_pack, selected_skills, llm) or _offline_answer_hint(
            question, seed_pack, selected_skills
        )
        report = evaluate_answer(question, answer, seed_pack, llm)
        repeated = memory.update_from_evaluation(topic, report)
        created = skills.suggest_from_weaknesses(topic, repeated)
        questions.append(question)
        answers.append(answer)
        log.append(
            "turn",
            {
                "round": idx + 1,
                "question": question,
                "answer": answer,
                "evidence": seed_pack.model_dump(),
                "selected_skills": selected_skills,
                "report": report.model_dump(),
                "created_skills": [str(path) for path in created],
                "working_summary": memory.build_working_summary(questions, answers),
            },
        )
    log.append("session_end", {"topic": topic, "rounds": rounds})
    return session_id


def prepare_interview_question(
    config: AgentConfig,
    topic: str,
    round_index: int,
    difficulty: str = "medium",
    knowledge_point: str = "auto",
    review_first: bool = True,
    exclude_points: list[str] | None = None,
) -> GraphState:
    retriever = AgenticRetriever(config)
    llm = LLMClient.from_config(config)
    skills = SkillManager(config)
    memory = MemoryManager(config)
    topic = normalize_topic(topic)
    difficulty = normalize_difficulty(difficulty)
    selection = memory.select_knowledge_point(topic, difficulty, knowledge_point or "auto", review_first, exclude_points)
    selected_kp = selection["knowledge_point"]
    seed_pack = retriever.retrieve(f"{topic} {selected_kp}")
    question = _llm_interview_question(topic, round_index, seed_pack, llm, difficulty, selected_kp, bool(selection["is_review"]))
    if not question:
        question = f"围绕知识点「{selected_kp}」，请用 {difficulty} 难度回答：{generate_question(topic, round_index, seed_pack)}"
    question_type = infer_question_type(question)
    selected_skills = skills.select(topic, question)
    return {
        "topic": topic,
        "round_index": round_index,
        "question": question,
        "evidence_pack": seed_pack.model_dump(),
        "selected_skills": selected_skills,
        "knowledge_point": selected_kp,
        "question_type": question_type,
        "difficulty": difficulty,
        "is_review": bool(selection["is_review"]),
        "review_key": str(selection.get("review_key") or ""),
    }


def run_interactive_turn(
    config: AgentConfig,
    session_id: str,
    topic: str,
    round_index: int,
    question_state: GraphState,
    answer: str,
) -> GraphState:
    from .models import EvaluationReport

    state: GraphState = {
        "session_id": session_id,
        "topic": normalize_topic(topic),
        "round_index": round_index,
        "question": str(question_state["question"]),
        "answer": answer,
        "evidence_pack": question_state["evidence_pack"],
        "selected_skills": question_state.get("selected_skills", []),
        "knowledge_point": str(question_state.get("knowledge_point") or ""),
        "question_type": str(question_state.get("question_type") or infer_question_type(str(question_state["question"]))),
        "difficulty": normalize_difficulty(str(question_state.get("difficulty") or "medium")),
        "is_review": bool(question_state.get("is_review", False)),
        "review_key": str(question_state.get("review_key") or ""),
    }
    state = _node_evaluate(config, state)
    memory = MemoryManager(config)
    report = EvaluationReport.model_validate(state["report"])
    repeated = memory.update_from_evaluation(normalize_topic(topic), report)
    memory_update = memory.record_user_answer(
        session_id=session_id,
        topic=normalize_topic(topic),
        knowledge_point=state["knowledge_point"],
        question_type=state["question_type"],
        difficulty=state["difficulty"],
        question=state["question"],
        answer=answer,
        report=report,
        evidence=state["evidence_pack"],
        is_review=state["is_review"],
        review_key=state["review_key"],
    )
    created = SkillManager(config).suggest_from_weaknesses(topic, repeated)
    state["repeated_weaknesses"] = repeated
    state["created_skills"] = [str(path) for path in created]
    state["memory_update"] = memory_update
    return state


def _llm_interview_question(
    topic: str,
    round_index: int,
    pack,
    llm: LLMClient,
    difficulty: str = "medium",
    knowledge_point: str = "",
    is_review: bool = False,
) -> str:
    if not llm.enabled:
        return ""
    evidence = "\n".join(f"- {ev.title}: {ev.snippet}" for ev in pack.evidence[:4])
    difficulty_hint = {
        "easy": "概念、流程、关键区别，适合基础巩固。",
        "medium": "工程方案、指标、取舍，适合常规面试。",
        "hard": "复杂系统设计、故障场景、多约束权衡和追问空间。",
    }.get(difficulty, "工程方案、指标、取舍，适合常规面试。")
    review_hint = "这是到期复习题，请围绕同一知识点生成变体题，不要重复原题。" if is_review else "这是新练习题。"
    text = llm.complete(
        system="你是 AI Agent/RAG 方向面试官。只输出一个面试问题，不要解释。",
        user=(
            f"主题：{topic}\n知识点：{knowledge_point}\n难度：{difficulty} - {difficulty_hint}\n"
            f"轮次：{round_index + 1}\n复习策略：{review_hint}\n证据：\n{evidence}\n\n"
            "请生成一个紧扣该知识点、能考察工程理解、取舍和追问空间的问题。"
        ),
        max_output_tokens=1200,
    )
    if not text or text.startswith("[LLM_ERROR]"):
        return ""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return lines[0].strip(" -")


def _llm_answer_hint(question: str, pack, selected_skills: list[str], llm: LLMClient) -> str:
    if not llm.enabled:
        return ""
    evidence = "\n".join(f"[{idx}] {ev.title}: {ev.snippet}" for idx, ev in enumerate(pack.evidence[:5], start=1))
    skills_text = "\n\n".join(selected_skills[:3])
    text = llm.complete(
        system=(
            "你是面试训练助手。请生成一版候选人示范回答，要求结构化、工程化、能抗追问，"
            "并基于证据回答。"
        ),
        user=(
            f"面试问题：{question}\n\n"
            f"证据：\n{evidence}\n\n"
            f"可用技能：\n{skills_text}\n\n"
            "请输出 1-2 分钟口头回答。"
        ),
        max_output_tokens=3000,
    )
    if not text or text.startswith("[LLM_ERROR]"):
        return ""
    return text.strip()


def _offline_answer_hint(question: str, pack, selected_skills: list[str]) -> str:
    sources = "、".join(ev.title for ev in pack.evidence[:2]) or "本地知识库"
    skill_hint = " 已参考 active skill。" if selected_skills else ""
    return (
        f"这是离线模拟回答草稿。针对问题「{question}」，先给结论，再结合 {sources} 的证据说明流程、"
        f"工程取舍、评估指标和失败兜底。{skill_hint}"
    )
