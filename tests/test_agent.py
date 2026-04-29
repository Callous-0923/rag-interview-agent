from __future__ import annotations

import tempfile
import unittest
import json
from datetime import datetime, timedelta
from pathlib import Path

from interview_agent.config import AgentConfig
from interview_agent.ingest import ensure_workspace, ingest_notes
from interview_agent.memory import MemoryManager
from interview_agent.models import EvaluationReport
from interview_agent.retrieval import AgenticRetriever
from interview_agent.skills import SkillManager
from interview_agent.workflow import prepare_interview_question, run_interactive_turn, run_mock_session


ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / "01_XHS_Notes" / "notes"
GRAPH = ROOT / "01_XHS_Notes" / "graphify-out" / "graph.json"
INTERVIEW_NOTES = ROOT / "00_Inbox" / "实习面经" / "01_notes"
INTERVIEW_GRAPH = ROOT / "00_Inbox" / "实习面经" / "graphify-out" / "graph.json"
TOPIC_INDEX = ROOT / "01_XHS_Notes" / "indexes" / "topic_index.md"
TOPIC_ASSIGNMENTS = ROOT / "01_XHS_Notes" / "indexes" / "topic_assignments.csv"


def make_config(tmp: Path) -> AgentConfig:
    return AgentConfig(
        project_root=tmp,
        knowledge_root=ROOT,
        notes_dir=NOTES,
        notes_dirs=[NOTES, INTERVIEW_NOTES],
        topic_index=TOPIC_INDEX,
        topic_assignments=TOPIC_ASSIGNMENTS,
        graph_json=GRAPH,
        graph_jsons=[GRAPH, INTERVIEW_GRAPH],
        storage_dir=tmp / "storage",
        session_dir=tmp / "sessions",
        memory_dir=tmp / "memory",
        skills_dir=tmp / "skills",
        chunk_size=900,
        chunk_overlap=120,
        max_rewrites=2,
        top_k=8,
        llm={"provider": "offline"},
    )


class AgentTests(unittest.TestCase):
    def test_ingest_and_rag_sources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = make_config(Path(td))
            ensure_workspace(cfg)
            result = ingest_notes(cfg)
            self.assertGreaterEqual(result["documents"], 100)
            pack = AgenticRetriever(cfg).retrieve("Agent记忆系统怎么设计")
            self.assertGreaterEqual(len(pack.evidence), 2)
            self.assertTrue(all(ev.source_file for ev in pack.evidence[:2]))

    def test_mock_session_writes_transcript(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = make_config(Path(td))
            ensure_workspace(cfg)
            ingest_notes(cfg)
            run_mock_session(cfg, "test_session", "RAG", 3)
            text = (cfg.session_dir / "test_session.jsonl").read_text(encoding="utf-8")
            self.assertIn('"event": "turn"', text)
            self.assertIn("working_summary", text)

    def test_interactive_turn_scores_user_answer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = make_config(Path(td))
            ensure_workspace(cfg)
            ingest_notes(cfg)
            question_state = prepare_interview_question(cfg, "RAG", 0)
            result = run_interactive_turn(
                cfg,
                "interactive_test",
                "RAG",
                0,
                question_state,
                "我会用混合检索、rerank、证据评估和失败兜底来设计RAG系统。",
            )
            self.assertIn("question", result)
            self.assertIn("report", result)
            self.assertIn("memory_update", result)
            self.assertGreaterEqual(result["report"]["correctness"], 1)
            history = (cfg.memory_dir / "answer_history.jsonl").read_text(encoding="utf-8")
            self.assertIn("混合检索", history)
            self.assertTrue((cfg.memory_dir / "growth_metrics.json").exists())
            self.assertTrue((cfg.memory_dir / "review_schedule.json").exists())

    def test_knowledge_point_progress_and_review_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = make_config(Path(td))
            ensure_workspace(cfg)
            memory = MemoryManager(cfg)
            report = EvaluationReport(
                correctness=2,
                structure=2,
                engineering_depth=2,
                tradeoff_quality=2,
                source_grounding=2,
                anti_followup=2,
                missing_points=["缺少rerank和评估指标"],
                better_answer="better",
                next_tasks=["补充RAG评估"],
            )
            memory.record_user_answer(
                session_id="s1",
                topic="RAG",
                knowledge_point="Rerank",
                question_type="evaluation",
                difficulty="hard",
                question="如何评估Rerank效果？",
                answer="answer",
                report=report,
                evidence={"evidence": []},
            )
            progress = memory.get_progress_summary("RAG")
            row = [item for item in progress["topics"]["RAG"]["knowledge_points"] if item["knowledge_point"] == "Rerank"][0]
            self.assertEqual(row["status"], "weak")
            schedule = json.loads((cfg.memory_dir / "review_schedule.json").read_text(encoding="utf-8"))
            next_due = list(schedule.values())[0]["next_due_at"]
            self.assertEqual(next_due, (datetime.now() + timedelta(days=1)).date().isoformat())
            gaps = memory.get_learning_gaps("RAG")
            rerank = [item for item in gaps["items"] if item["knowledge_point"] == "Rerank"][0]
            self.assertTrue(rerank["suggestions"])

    def test_auto_questions_cover_different_knowledge_points_in_round(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = make_config(Path(td))
            ensure_workspace(cfg)
            ingest_notes(cfg)
            used: list[str] = []
            for idx in range(3):
                state = prepare_interview_question(cfg, "RAG", idx, "medium", "auto", True, used)
                used.append(str(state["knowledge_point"]))
            self.assertEqual(len(used), len(set(used)))

    def test_repeated_weakness_creates_pending_skill(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cfg = make_config(Path(td))
            ensure_workspace(cfg)
            report = EvaluationReport(
                correctness=2,
                structure=2,
                engineering_depth=2,
                tradeoff_quality=2,
                source_grounding=2,
                anti_followup=2,
                missing_points=["缺少工程细节和指标"],
                better_answer="better",
                next_tasks=["task"],
            )
            memory = MemoryManager(cfg)
            repeated = []
            for _ in range(3):
                repeated = memory.update_from_evaluation("RAG", report)
            created = SkillManager(cfg).suggest_from_weaknesses("RAG", repeated)
            self.assertTrue(created)
            self.assertTrue(created[0].exists())


if __name__ == "__main__":
    unittest.main()
