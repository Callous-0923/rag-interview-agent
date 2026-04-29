from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from .config import AgentConfig
from .models import EvaluationReport


REVIEW_INTERVAL_DAYS = [1, 2, 4, 7, 15, 30]
DIFFICULTIES = {"easy", "medium", "hard"}

KNOWLEDGE_POINTS: dict[str, list[str]] = {
    "RAG": [
        "知识库构建",
        "Chunk策略",
        "Embedding与向量库",
        "混合检索",
        "Rerank",
        "Query Rewrite",
        "GraphRAG",
        "Agentic RAG",
        "RAG评估",
        "幻觉归因",
        "多模态RAG",
        "工程化与监控",
    ],
    "Agent": [
        "Agent架构",
        "Tool Use",
        "Planning",
        "Memory",
        "Context Engineering",
        "多Agent",
        "LangGraph工作流",
        "失败恢复",
        "评估体系",
    ],
    "Harness": [
        "Harness架构",
        "JIT Context",
        "上下文装配",
        "工具编排",
        "代码库检索",
        "执行轨迹",
        "评估与观测",
    ],
    "大模型": [
        "Transformer基础",
        "Attention机制",
        "模型微调",
        "对齐方法",
        "长上下文",
        "多模态模型",
        "推理优化",
        "模型评估",
    ],
    "Python": [
        "基础语法",
        "数据结构",
        "面向对象",
        "异步与并发",
        "文件与IO",
        "测试与调试",
        "工程化",
        "算法实现",
    ],
}

TRAINING_TOPICS = tuple(KNOWLEDGE_POINTS.keys())

SYLLABUS_GAPS: dict[str, list[str]] = {
    "知识库构建": ["数据清洗", "元数据设计", "增量更新", "权限过滤"],
    "Chunk策略": ["标题层级切分", "语义分块", "chunk overlap", "父子 chunk"],
    "Embedding与向量库": ["embedding 模型选择", "向量库索引", "召回延迟", "更新策略"],
    "混合检索": ["BM25", "Dense 召回", "多路召回融合", "Top-K 截断"],
    "Rerank": ["cross-encoder rerank", "重排特征", "rerank 评估", "延迟成本"],
    "Query Rewrite": ["意图识别", "多查询改写", "HyDE", "改写失败兜底"],
    "GraphRAG": ["实体关系抽取", "多跳推理", "图谱增量更新", "图谱召回评估"],
    "Agentic RAG": ["证据充足性判断", "循环检索", "工具调用", "失败恢复"],
    "RAG评估": ["Hit Rate", "NDCG", "faithfulness", "answer correctness", "失败归因实验"],
    "幻觉归因": ["检索缺失", "证据冲突", "生成越界", "引用覆盖率"],
    "多模态RAG": ["图片 OCR", "多模态 embedding", "图文对齐", "视觉证据引用"],
    "工程化与监控": ["延迟", "成本", "缓存", "观测指标", "灰度评估"],
    "Agent架构": ["状态机", "工具层", "记忆层", "执行闭环"],
    "Tool Use": ["工具 schema", "参数校验", "权限", "错误重试"],
    "Planning": ["任务分解", "计划修正", "执行检查点", "停止条件"],
    "Memory": ["短期记忆", "长期记忆", "工作摘要", "记忆检索"],
    "Context Engineering": ["上下文选择", "JIT Context", "上下文压缩", "污染控制"],
    "多Agent": ["角色拆分", "通信协议", "冲突协调", "成本控制"],
    "LangGraph工作流": ["节点状态", "分支恢复", "checkpoint", "工具节点"],
    "失败恢复": ["重试", "fallback", "人工确认", "回滚"],
    "评估体系": ["任务成功率", "工具调用准确率", "忠实度", "端到端稳定性"],
    "Harness架构": ["任务目标", "上下文管理", "工具层", "执行循环"],
    "JIT Context": ["按需探索", "上下文预算", "工具调用策略", "停止条件"],
    "上下文装配": ["证据选择", "上下文压缩", "引用保留", "污染控制"],
    "工具编排": ["工具路由", "参数构造", "异常恢复", "权限边界"],
    "代码库检索": ["符号搜索", "调用链", "依赖关系", "增量索引"],
    "执行轨迹": ["中间日志", "可恢复状态", "回放", "审计"],
    "评估与观测": ["任务成功率", "延迟", "成本", "工具错误率"],
    "Transformer基础": ["Encoder/Decoder", "位置编码", "残差连接", "LayerNorm"],
    "Attention机制": ["QKV", "缩放点积注意力", "多头注意力", "mask"],
    "模型微调": ["SFT", "数据构造", "LoRA", "过拟合控制"],
    "对齐方法": ["RLHF", "DPO", "PPO", "偏好数据"],
    "长上下文": ["RoPE", "上下文压缩", "注意力复杂度", "检索增强"],
    "多模态模型": ["视觉编码", "图文对齐", "OCR", "多模态评估"],
    "推理优化": ["KV Cache", "量化", "批处理", "延迟优化"],
    "模型评估": ["基准集", "人工评估", "鲁棒性", "安全性"],
    "基础语法": ["变量类型", "控制流", "函数", "异常处理"],
    "数据结构": ["list/dict/set", "复杂度", "堆栈队列", "排序"],
    "面向对象": ["类与实例", "继承", "协议", "数据模型"],
    "异步与并发": ["asyncio", "线程", "进程", "锁与队列"],
    "文件与IO": ["路径处理", "编码", "JSON/CSV", "流式读写"],
    "测试与调试": ["unittest", "pytest", "断点调试", "日志"],
    "工程化": ["包管理", "配置", "CLI", "代码风格"],
    "算法实现": ["双指针", "动态规划", "图算法", "复杂度分析"],
}


class MemoryManager:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.config.memory_dir.mkdir(parents=True, exist_ok=True)
        self.weakness_path = self.config.memory_dir / "weakness_map.md"
        self.mastery_path = self.config.memory_dir / "topic_mastery.json"
        self.answer_history_path = self.config.memory_dir / "answer_history.jsonl"
        self.growth_path = self.config.memory_dir / "growth_metrics.json"
        self.review_path = self.config.memory_dir / "review_schedule.json"
        self.learning_gaps_path = self.config.memory_dir / "learning_gaps.json"

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

    def record_user_answer(
        self,
        *,
        session_id: str,
        topic: str,
        knowledge_point: str,
        question_type: str,
        difficulty: str,
        question: str,
        answer: str,
        report: EvaluationReport,
        evidence: dict[str, Any],
        is_review: bool = False,
        review_key: str = "",
    ) -> dict[str, Any]:
        topic = normalize_topic(topic)
        difficulty = normalize_difficulty(difficulty)
        knowledge_point = normalize_knowledge_point(topic, knowledge_point)
        question_id = _stable_id([session_id, topic, knowledge_point, question, datetime.now().isoformat()])
        record = {
            "question_id": question_id,
            "session_id": session_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "topic": topic,
            "knowledge_point": knowledge_point,
            "question_type": question_type,
            "difficulty": difficulty,
            "question": question,
            "answer": answer,
            "score": report.average,
            "scores": {
                "correctness": report.correctness,
                "structure": report.structure,
                "engineering_depth": report.engineering_depth,
                "tradeoff_quality": report.tradeoff_quality,
                "source_grounding": report.source_grounding,
                "anti_followup": report.anti_followup,
            },
            "missing_points": report.missing_points,
            "better_answer": report.better_answer,
            "next_tasks": report.next_tasks,
            "evidence_sources": [
                {"title": item.get("title", ""), "source_file": item.get("source_file", "")}
                for item in evidence.get("evidence", [])[:6]
            ],
            "is_review": is_review,
            "review_key": review_key,
        }
        with self.answer_history_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        growth_item = self.update_growth_metrics(record, report)
        review_item = self.schedule_review(record, report)
        gaps = self.update_learning_gaps(record, report)
        return {"answer_record": record, "growth": growth_item, "review": review_item, "gaps": gaps}

    def update_growth_metrics(self, record: dict[str, Any], report: EvaluationReport) -> dict[str, Any]:
        metrics = self._load_json(self.growth_path)
        key = metric_key(record["topic"], record["knowledge_point"], record["question_type"], record["difficulty"])
        item = metrics.setdefault(
            key,
            {
                "topic": record["topic"],
                "knowledge_point": record["knowledge_point"],
                "question_type": record["question_type"],
                "difficulty": record["difficulty"],
                "attempts": 0,
                "avg_score": 0.0,
                "recent_scores": [],
                "recent_score": 0.0,
                "best_score": 0.0,
                "weakness_trend": {},
                "last_answered_at": "",
                "status": "unseen",
            },
        )
        attempts = int(item["attempts"]) + 1
        item["avg_score"] = round(((float(item["avg_score"]) * (attempts - 1)) + report.average) / attempts, 2)
        item["attempts"] = attempts
        recent = [float(value) for value in item.get("recent_scores", [])][-9:] + [report.average]
        item["recent_scores"] = recent
        item["recent_score"] = round(sum(recent[-5:]) / min(len(recent), 5), 2)
        item["best_score"] = max(float(item.get("best_score", 0.0)), report.average)
        item["last_answered_at"] = record["ts"]
        weaknesses = Counter(item.get("weakness_trend", {}))
        for weakness in report.missing_points or ["表达结构和工程细节不足"]:
            weaknesses[weakness] += 1
        item["weakness_trend"] = dict(weaknesses)
        item["status"] = mastery_status(item)
        self._write_json(self.growth_path, metrics)
        return item

    def schedule_review(self, record: dict[str, Any], report: EvaluationReport) -> dict[str, Any]:
        schedule = self._load_json(self.review_path)
        key = metric_key(record["topic"], record["knowledge_point"], record["question_type"], record["difficulty"])
        current = schedule.get(key, {})
        stage = int(current.get("review_stage", 0))
        if report.average < 3:
            next_stage = stage
            interval = 1
        else:
            next_stage = min(stage + 1, len(REVIEW_INTERVAL_DAYS) - 1)
            interval = REVIEW_INTERVAL_DAYS[next_stage]
        mastered = bool(report.average >= 4.5 and stage >= 1)
        if mastered:
            next_stage = len(REVIEW_INTERVAL_DAYS) - 1
            interval = REVIEW_INTERVAL_DAYS[-1]
        item = {
            "review_key": key,
            "topic": record["topic"],
            "knowledge_point": record["knowledge_point"],
            "question_type": record["question_type"],
            "difficulty": record["difficulty"],
            "next_due_at": (datetime.now() + timedelta(days=interval)).date().isoformat(),
            "review_stage": next_stage,
            "last_score": report.average,
            "source_question_id": record["question_id"],
            "mastered": mastered,
        }
        schedule[key] = item
        self._write_json(self.review_path, schedule)
        return item

    def update_learning_gaps(self, record: dict[str, Any], report: EvaluationReport) -> dict[str, Any]:
        gaps = self._load_json(self.learning_gaps_path)
        topic = record["topic"]
        kp = record["knowledge_point"]
        topic_item = gaps.setdefault(topic, {})
        suggestions = set(topic_item.get(kp, {}).get("suggestions", []))
        for item in SYLLABUS_GAPS.get(kp, []):
            suggestions.add(item)
        for missing in report.missing_points:
            suggestions.update(_map_missing_to_suggestions(missing, kp))
        entry = {
            "topic": topic,
            "knowledge_point": kp,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "last_score": report.average,
            "suggestions": sorted(suggestions)[:8],
            "next_practice": _next_practice_hint(topic, kp, report),
        }
        topic_item[kp] = entry
        self._write_json(self.learning_gaps_path, gaps)
        return entry

    def get_due_reviews(self, topic: str | None = None, difficulty: str | None = None) -> list[dict[str, Any]]:
        schedule = self._load_json(self.review_path)
        today = datetime.now().date().isoformat()
        topic_norm = normalize_topic(topic or "") if topic else None
        difficulty_norm = normalize_difficulty(difficulty or "") if difficulty else None
        items = []
        for item in schedule.values():
            if topic_norm and item.get("topic") != topic_norm:
                continue
            if difficulty_norm and item.get("difficulty") != difficulty_norm:
                continue
            if str(item.get("next_due_at", "")) <= today:
                items.append(item)
        return sorted(items, key=lambda value: (value.get("next_due_at", ""), value.get("last_score", 0)))

    def get_progress_summary(self, topic: str | None = None) -> dict[str, Any]:
        metrics = self._load_json(self.growth_path)
        gaps = self._load_json(self.learning_gaps_path)
        schedule = self._load_json(self.review_path)
        topics = [normalize_topic(topic)] if topic else sorted(KNOWLEDGE_POINTS)
        result: dict[str, Any] = {"topics": {}, "due_reviews": len(self.get_due_reviews(topic))}
        for topic_name in topics:
            kp_rows = []
            for kp in knowledge_points_for_topic(topic_name):
                rows = [item for item in metrics.values() if item.get("topic") == topic_name and item.get("knowledge_point") == kp]
                attempts = sum(int(item.get("attempts", 0)) for item in rows)
                recent_scores = [score for item in rows for score in item.get("recent_scores", [])][-10:]
                recent_score = round(sum(recent_scores[-5:]) / min(len(recent_scores), 5), 2) if recent_scores else 0.0
                status = _aggregate_status(rows)
                due = [
                    item
                    for item in schedule.values()
                    if item.get("topic") == topic_name and item.get("knowledge_point") == kp
                ]
                next_due = min([item.get("next_due_at", "") for item in due if item.get("next_due_at")] or [""])
                kp_rows.append(
                    {
                        "knowledge_point": kp,
                        "attempts": attempts,
                        "recent_score": recent_score,
                        "status": status,
                        "recent_scores": recent_scores[-10:],
                        "next_due_at": next_due,
                        "gaps": gaps.get(topic_name, {}).get(kp, {}).get("suggestions", SYLLABUS_GAPS.get(kp, []))[:5],
                    }
                )
            result["topics"][topic_name] = {
                "knowledge_points": kp_rows,
                "attempts": sum(item["attempts"] for item in kp_rows),
                "avg_recent_score": _avg([item["recent_score"] for item in kp_rows if item["attempts"] > 0]),
            }
        return result

    def get_learning_gaps(self, topic: str) -> dict[str, Any]:
        topic = normalize_topic(topic)
        gaps = self._load_json(self.learning_gaps_path).get(topic, {})
        points = []
        for kp in knowledge_points_for_topic(topic):
            entry = gaps.get(kp, {})
            points.append(
                {
                    "knowledge_point": kp,
                    "suggestions": entry.get("suggestions", SYLLABUS_GAPS.get(kp, []))[:6],
                    "next_practice": entry.get("next_practice", f"围绕{kp}做一道中等难度工程题。"),
                    "last_score": entry.get("last_score"),
                }
            )
        return {"topic": topic, "items": points}

    def select_knowledge_point(
        self,
        topic: str,
        difficulty: str = "medium",
        requested: str = "auto",
        review_first: bool = True,
        exclude_points: list[str] | None = None,
    ) -> dict[str, Any]:
        topic = normalize_topic(topic)
        difficulty = normalize_difficulty(difficulty)
        excluded = set(exclude_points or [])
        if requested and requested != "auto":
            kp = normalize_knowledge_point(topic, requested)
            return {"knowledge_point": kp, "is_review": False, "review_key": ""}
        if review_first:
            due = [item for item in self.get_due_reviews(topic, difficulty) if item.get("knowledge_point") not in excluded]
            if due:
                item = due[0]
                return {"knowledge_point": item["knowledge_point"], "is_review": True, "review_key": item["review_key"]}
        summary = self.get_progress_summary(topic)["topics"][topic]["knowledge_points"]
        available = [item for item in summary if item["knowledge_point"] not in excluded]
        if not available:
            available = summary
        weak = [item for item in available if item["status"] in {"weak", "improving"} and item["attempts"] > 0]
        if weak:
            item = sorted(weak, key=lambda value: (value["recent_score"], value["attempts"]))[0]
            return {"knowledge_point": item["knowledge_point"], "is_review": False, "review_key": ""}
        unseen = [item for item in available if item["attempts"] == 0]
        if unseen:
            return {"knowledge_point": unseen[0]["knowledge_point"], "is_review": False, "review_key": ""}
        item = sorted(available, key=lambda value: value["recent_score"])[0]
        return {"knowledge_point": item["knowledge_point"], "is_review": False, "review_key": ""}

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

    def _load_json(self, path) -> dict[str, Any]:
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8") or "{}")

    def _write_json(self, path, data: dict[str, Any]) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")


def normalize_topic(topic: str) -> str:
    lowered = topic.lower()
    if "rag" in lowered or "检索" in topic:
        return "RAG"
    if "harness" in lowered or "上下文工程" in topic or "jit" in lowered:
        return "Harness"
    if "大模型" in topic or "llm" in lowered or "模型" in topic:
        return "大模型"
    if "python" in lowered or "py" == lowered:
        return "Python"
    return "Agent"


def normalize_difficulty(difficulty: str) -> str:
    value = (difficulty or "medium").lower()
    return value if value in DIFFICULTIES else "medium"


def knowledge_points_for_topic(topic: str) -> list[str]:
    return KNOWLEDGE_POINTS.get(normalize_topic(topic), KNOWLEDGE_POINTS["Agent"])


def normalize_knowledge_point(topic: str, knowledge_point: str) -> str:
    points = knowledge_points_for_topic(topic)
    if knowledge_point in points:
        return knowledge_point
    lowered = knowledge_point.lower()
    for point in points:
        if lowered and (lowered in point.lower() or point.lower() in lowered):
            return point
    return points[0]


def infer_question_type(question: str) -> str:
    lowered = question.lower()
    if any(term in question for term in ["评估", "指标", "体系"]) or "metric" in lowered:
        return "evaluation"
    if any(term in question for term in ["系统", "设计", "架构", "方案"]):
        return "system_design"
    if any(term in question for term in ["项目", "落地", "工程"]):
        return "project"
    if any(term in question for term in ["排查", "错误", "失败", "Bug", "错了", "幻觉"]):
        return "troubleshooting"
    if any(term in question for term in ["追问", "如果"]) or "follow" in lowered:
        return "followup"
    return "concept"


def metric_key(topic: str, knowledge_point: str, question_type: str, difficulty: str) -> str:
    return "|".join([normalize_topic(topic), knowledge_point, question_type, normalize_difficulty(difficulty)])


def mastery_status(item: dict[str, Any]) -> str:
    attempts = int(item.get("attempts", 0))
    if attempts <= 0:
        return "unseen"
    recent = float(item.get("recent_score", 0.0))
    scores = [float(value) for value in item.get("recent_scores", [])]
    if len(scores) >= 2 and scores[-1] >= 4.5 and scores[-2] >= 4.5:
        return "mastered"
    if recent < 3:
        return "weak"
    return "improving"


def _aggregate_status(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "unseen"
    if any(item.get("status") == "weak" for item in rows):
        return "weak"
    if rows and all(item.get("status") == "mastered" for item in rows):
        return "mastered"
    return "improving"


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _stable_id(parts: list[str]) -> str:
    import hashlib

    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _map_missing_to_suggestions(missing: str, knowledge_point: str) -> set[str]:
    text = missing.lower()
    suggestions = set()
    mapping = {
        "rerank": "Rerank",
        "重排": "Rerank",
        "rewrite": "Query Rewrite",
        "改写": "Query Rewrite",
        "评估": "RAG评估",
        "指标": "RAG评估",
        "graph": "GraphRAG",
        "图谱": "GraphRAG",
        "幻觉": "幻觉归因",
        "chunk": "Chunk策略",
        "分块": "Chunk策略",
        "监控": "工程化与监控",
        "成本": "工程化与监控",
    }
    for keyword, target in mapping.items():
        if keyword in text:
            suggestions.update(SYLLABUS_GAPS.get(target, [target]))
    suggestions.update(SYLLABUS_GAPS.get(knowledge_point, [])[:3])
    return suggestions


def _next_practice_hint(topic: str, knowledge_point: str, report: EvaluationReport) -> str:
    if report.average < 3:
        return f"先做一道 {knowledge_point} 的 easy/medium 基础题，补齐核心概念和流程。"
    return f"继续做一道 {knowledge_point} 的 medium/hard 场景题，重点覆盖工程取舍和评估指标。"
