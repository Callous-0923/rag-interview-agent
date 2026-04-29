from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import typer

from .config import load_config
from .diagnose import diagnose as diagnose_agent
from .ingest import ensure_workspace, ingest_notes
from .memory import MemoryManager
from .retrieval import AgenticRetriever
from .sessions import SessionLog, list_sessions, new_session_id
from .skills import SkillManager
from .vision import run_vision_ingest
from .web import run_web_app
from .workflow import prepare_interview_question, run_interactive_turn, run_mock_session, run_review_turn

app = typer.Typer(help="Local interview learning Agent.")
skills_app = typer.Typer(help="Skill commands.")
vision_app = typer.Typer(help="Vision OCR ingestion commands.")
app.add_typer(skills_app, name="skills")
app.add_typer(vision_app, name="vision")

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


@app.command()
def init(config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml")) -> None:
    cfg = load_config(config)
    ensure_workspace(cfg)
    typer.echo(f"Initialized workspace: {cfg.project_root}")


@app.command()
def ingest(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
    reset: bool = typer.Option(True, help="Clear previous index first."),
) -> None:
    cfg = load_config(config)
    ensure_workspace(cfg)
    result = ingest_notes(cfg, reset=reset)
    typer.echo(f"Ingested {result['documents']} documents, {result['chunks']} chunks")
    typer.echo(f"SQLite index: {result['db']}")
    chroma = result.get("chroma") or {}
    if chroma.get("enabled"):
        typer.echo(f"Chroma index: {chroma.get('path')} ({chroma.get('collection')}, {chroma.get('chunks')} chunks)")
    else:
        typer.echo(f"Chroma skipped: {chroma.get('reason')}")


@app.command()
def ask(query: str, config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml")) -> None:
    cfg = load_config(config)
    typer.echo(AgenticRetriever(cfg).answer(query))


@app.command()
def diagnose(
    role: str = typer.Option(..., "--role", help="Target role"),
    jd: Optional[Path] = typer.Option(None, "--jd", help="JD markdown file"),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Resume markdown file"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    ensure_workspace(cfg)
    typer.echo(diagnose_agent(cfg, role, jd, resume))


@app.command()
def mock(
    topic: str = typer.Option("RAG", "--topic", help="Training topic: RAG, Agent, Harness, 大模型, or Python"),
    rounds: int = typer.Option(3, "--rounds", min=1, max=20, help="Number of turns"),
    session: Optional[str] = typer.Option(None, "--session", help="Resume/write to session id"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    ensure_workspace(cfg)
    session_id = session or new_session_id()
    run_mock_session(cfg, session_id, topic, rounds)
    typer.echo(f"Mock session saved: {cfg.session_dir / (session_id + '.jsonl')}")


@app.command()
def interview(
    topic: str = typer.Option("RAG", "--topic", help="Training topic: RAG, Agent, Harness, 大模型, or Python"),
    rounds: int = typer.Option(5, "--rounds", min=1, max=20, help="Number of turns"),
    difficulty: str = typer.Option("medium", "--difficulty", help="easy, medium, or hard"),
    knowledge_point: str = typer.Option("auto", "--knowledge-point", help="Knowledge point or auto"),
    review_first: bool = typer.Option(True, "--review-first/--no-review-first", help="Prefer due review items first."),
    session: Optional[str] = typer.Option(None, "--session", help="Resume/write to session id"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    ensure_workspace(cfg)
    session_id = session or new_session_id()
    log = SessionLog(cfg, session_id)
    previous_turns = [event for event in log.read_events() if event.get("event") == "turn"]
    questions = [str(event["payload"].get("question", "")) for event in previous_turns]
    answers = [str(event["payload"].get("answer", "")) for event in previous_turns]
    used_points = [str(event["payload"].get("knowledge_point", "")) for event in previous_turns if event["payload"].get("knowledge_point")]
    start_round = len(previous_turns)
    if start_round == 0:
        log.append(
            "session_start",
            {
                "topic": topic,
                "rounds": rounds,
                "mode": "interactive",
                "difficulty": difficulty,
                "knowledge_point": knowledge_point,
                "review_first": review_first,
            },
        )
    typer.echo(f"Interactive interview session: {session_id}")
    typer.echo("Commands: :quit exit, :hint show evidence, :skip skip current question")

    for idx in range(start_round, rounds):
        typer.echo("")
        typer.echo(f"Round {idx + 1}/{rounds}")
        exclude_points = used_points if knowledge_point == "auto" else []
        question_state = prepare_interview_question(cfg, topic, idx, difficulty, knowledge_point, review_first, exclude_points)
        question = str(question_state["question"])
        marker = "review" if question_state.get("is_review") else "new"
        typer.echo(
            f"Interviewer [{question_state.get('knowledge_point')} · {question_state.get('difficulty')} · {marker}]: {question}"
        )
        answer = _read_multiline_answer()
        if answer == ":quit":
            log.append("session_pause", {"topic": topic, "round": idx + 1})
            typer.echo(f"Paused. Resume with: python -m interview_agent.cli interview --topic {topic} --rounds {rounds} --session {session_id}")
            return
        if answer == ":hint":
            _print_evidence_hint(question_state)
            answer = _read_multiline_answer()
            if answer == ":quit":
                log.append("session_pause", {"topic": topic, "round": idx + 1})
                typer.echo(f"Paused. Resume with: python -m interview_agent.cli interview --topic {topic} --rounds {rounds} --session {session_id}")
                return
        if answer == ":skip":
            log.append("skip", {"round": idx + 1, "question": question})
            typer.echo("Skipped.")
            continue
        if not answer.strip():
            typer.echo("Empty answer skipped.")
            continue

        result = run_interactive_turn(cfg, session_id, topic, idx, question_state, answer)
        if result.get("knowledge_point"):
            used_points.append(str(result["knowledge_point"]))
        questions.append(question)
        answers.append(answer)
        report = result["report"]
        log.append(
            "turn",
            {
                "round": idx + 1,
                "question": question,
                "answer": answer,
                "evidence": result["evidence_pack"],
                "selected_skills": result.get("selected_skills", []),
                "knowledge_point": result.get("knowledge_point", ""),
                "question_type": result.get("question_type", ""),
                "difficulty": result.get("difficulty", difficulty),
                "is_review": result.get("is_review", False),
                "report": report,
                "created_skills": result.get("created_skills", []),
                "memory_update": result.get("memory_update", {}),
                "working_summary": _build_working_summary(cfg, questions, answers),
            },
        )
        _print_report(report)
        if result.get("created_skills"):
            typer.echo("Created pending skills:")
            for item in result["created_skills"]:
                typer.echo(f"- {item}")

    log.append("session_end", {"topic": topic, "rounds": rounds, "mode": "interactive"})
    typer.echo("")
    typer.echo(f"Session saved: {cfg.session_dir / (session_id + '.jsonl')}")


@app.command()
def review(
    session: str = typer.Option(..., "--session", help="Session id or jsonl path"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    path = Path(session)
    session_id = path.stem if path.suffix == ".jsonl" else session
    log = SessionLog(cfg, session_id)
    events = [event for event in log.read_events() if event.get("event") == "turn"]
    if not events:
        raise typer.BadParameter(f"No turn events found for session: {session}")
    last = events[-1]["payload"]
    result = run_review_turn(
        cfg,
        {
            "session_id": session_id,
            "topic": last.get("evidence", {}).get("topics", ["Agent"])[0] if last.get("evidence") else "Agent",
            "question": last["question"],
            "answer": last["answer"],
        },
    )
    log.append("review", result)
    typer.echo("Review complete.")
    typer.echo(result.get("report", {}))
    if result.get("created_skills"):
        typer.echo("Created pending skills:")
        for item in result["created_skills"]:
            typer.echo(f"- {item}")


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8765, "--port", min=1024, max=65535, help="Bind port"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    run_web_app(cfg, host=host, port=port)


def _read_multiline_answer() -> str:
    typer.echo("Your answer. Finish with an empty line.")
    first = input("> ")
    command = first.strip()
    if command in {":quit", ":hint", ":skip"}:
        return command
    lines = [first]
    while True:
        line = input("> ")
        if not line:
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _print_evidence_hint(question_state: dict) -> None:
    evidence = (question_state.get("evidence_pack") or {}).get("evidence", [])
    if not evidence:
        typer.echo("No evidence found.")
        return
    typer.echo("Evidence hint:")
    for idx, item in enumerate(evidence[:3], start=1):
        typer.echo(f"{idx}. {item.get('title', '')} - {item.get('source_file', '')}")
        snippet = str(item.get("snippet", "")).replace("\n", " ")
        typer.echo(f"   {snippet[:220]}")


def _print_report(report: dict) -> None:
    scores = [
        report.get("correctness", 0),
        report.get("structure", 0),
        report.get("engineering_depth", 0),
        report.get("tradeoff_quality", 0),
        report.get("source_grounding", 0),
        report.get("anti_followup", 0),
    ]
    avg = round(sum(int(v) for v in scores) / len(scores), 2) if scores else 0
    typer.echo("")
    typer.echo(f"Score: {avg}/5")
    missing = report.get("missing_points") or []
    if missing:
        typer.echo("Missing points:")
        for item in missing[:5]:
            typer.echo(f"- {item}")
    better = str(report.get("better_answer") or "").strip()
    if better:
        typer.echo("Better answer:")
        typer.echo(better)
    tasks = report.get("next_tasks") or []
    if tasks:
        typer.echo("Next tasks:")
        for item in tasks[:5]:
            typer.echo(f"- {item}")


@app.command()
def progress(
    topic: Optional[str] = typer.Option(None, "--topic", help="Filter by training topic"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    summary = MemoryManager(cfg).get_progress_summary(topic)
    for topic_name, item in summary["topics"].items():
        typer.echo(f"{topic_name}: attempts={item['attempts']} recent_avg={item['avg_recent_score']}")
        for kp in item["knowledge_points"]:
            typer.echo(
                f"- {kp['knowledge_point']}: {kp['status']} attempts={kp['attempts']} "
                f"recent={kp['recent_score']} next_due={kp['next_due_at'] or '-'}"
            )


@app.command()
def reviews(
    topic: Optional[str] = typer.Option(None, "--topic", help="Filter by training topic"),
    difficulty: Optional[str] = typer.Option(None, "--difficulty", help="Filter by difficulty"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    due = MemoryManager(cfg).get_due_reviews(topic, difficulty)
    if not due:
        typer.echo("No due reviews.")
        return
    for item in due:
        typer.echo(
            f"{item['next_due_at']} · {item['topic']} · {item['knowledge_point']} · "
            f"{item['difficulty']} · score={item['last_score']}"
        )


@app.command()
def gaps(
    topic: str = typer.Option("RAG", "--topic", help="Training topic"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    data = MemoryManager(cfg).get_learning_gaps(topic)
    typer.echo(f"Learning gaps for {data['topic']}:")
    for item in data["items"]:
        typer.echo(f"- {item['knowledge_point']}: {'; '.join(item['suggestions'])}")
        typer.echo(f"  next: {item['next_practice']}")


def _build_working_summary(cfg, questions: list[str], answers: list[str]) -> str:
    from .memory import MemoryManager

    return MemoryManager(cfg).build_working_summary(questions, answers)


@app.command("sessions")
def sessions_cmd(config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml")) -> None:
    cfg = load_config(config)
    for path in list_sessions(cfg):
        typer.echo(path.stem)


@vision_app.command("ingest")
def vision_ingest(
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
    limit: Optional[int] = typer.Option(None, "--limit", min=1, help="Max images to process."),
    force: bool = typer.Option(False, "--force", help="Reprocess images even if generated Markdown exists."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only count pending images."),
    update_index: bool = typer.Option(False, "--update-index", help="Run agent ingest after writing vision Markdown."),
) -> None:
    cfg = load_config(config)
    ensure_workspace(cfg)
    result = run_vision_ingest(cfg, limit=limit, force=force, dry_run=dry_run)
    typer.echo(
        f"Vision scanned={result['scanned']} selected={result['selected']} "
        f"processed={result['processed']} pending={result['pending']}"
    )
    typer.echo(f"Output: {result['output_dir']}")
    for error in result.get("errors", []):
        typer.echo(f"ERROR: {error}", err=True)
    if update_index and not dry_run:
        ingest_result = ingest_notes(cfg, reset=True)
        typer.echo(f"Re-indexed {ingest_result['documents']} documents, {ingest_result['chunks']} chunks")
        chroma = ingest_result.get("chroma") or {}
        if chroma.get("enabled"):
            typer.echo(f"Chroma index: {chroma.get('path')} ({chroma.get('collection')}, {chroma.get('chunks')} chunks)")
        else:
            typer.echo(f"Chroma skipped: {chroma.get('reason')}")


@skills_app.command("list")
def skills_list(
    status: str = typer.Option("active", "--status", help="active or pending"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    manager = SkillManager(cfg)
    for path in manager.list_skills(status):
        typer.echo(path.parent.name)


@skills_app.command("show")
def skills_show(
    name: str,
    status: str = typer.Option("active", "--status", help="active or pending"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    typer.echo(SkillManager(cfg).show(name, status))


@skills_app.command("run")
def skills_run(
    name: str,
    status: str = typer.Option("active", "--status", help="active or pending"),
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    typer.echo(SkillManager(cfg).show(name, status))


@skills_app.command("activate")
def skills_activate(
    name: str,
    config: Optional[Path] = typer.Option(None, "--config", help="Path to config.yaml"),
) -> None:
    cfg = load_config(config)
    path = SkillManager(cfg).activate(name)
    typer.echo(f"Activated: {path}")


if __name__ == "__main__":
    app()
