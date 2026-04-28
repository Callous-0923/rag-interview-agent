from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    doc_id: str
    title: str
    source_file: str
    source_url: str = ""
    author: str = ""
    publish_date: str = ""
    tags: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source_file: str
    source_url: str = ""
    topics: list[str] = Field(default_factory=list)
    text: str


class Evidence(BaseModel):
    title: str
    source_file: str
    source_url: str = ""
    snippet: str
    score: float
    reason: str


class EvidencePack(BaseModel):
    query: str
    rewritten_queries: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    graph_neighbors: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)


class EvaluationReport(BaseModel):
    correctness: int
    structure: int
    engineering_depth: int
    tradeoff_quality: int
    source_grounding: int
    anti_followup: int
    missing_points: list[str] = Field(default_factory=list)
    better_answer: str = ""
    next_tasks: list[str] = Field(default_factory=list)

    @property
    def average(self) -> float:
        scores = [
            self.correctness,
            self.structure,
            self.engineering_depth,
            self.tradeoff_quality,
            self.source_grounding,
            self.anti_followup,
        ]
        return round(sum(scores) / len(scores), 2)


@dataclass
class InterviewState:
    session_id: str
    target_role: str = "AI Agent实习"
    jd_summary: str = ""
    resume_summary: str = ""
    current_topic: str = ""
    question_history: list[str] = field(default_factory=list)
    answer_history: list[str] = field(default_factory=list)
    retrieved_evidence: list[dict[str, Any]] = field(default_factory=list)
    weakness_map: dict[str, int] = field(default_factory=dict)
    score_history: list[dict[str, Any]] = field(default_factory=list)
    short_memory: list[str] = field(default_factory=list)
    working_summary: str = ""
    selected_skills: list[str] = field(default_factory=list)
