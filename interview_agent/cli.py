from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import typer

from .config import load_config
from .diagnose import diagnose as diagnose_agent
from .ingest import ensure_workspace, ingest_notes
from .retrieval import AgenticRetriever
from .sessions import SessionLog, list_sessions, new_session_id
from .skills import SkillManager
from .vision import run_vision_ingest
from .workflow import run_mock_session, run_review_turn

app = typer.Typer(help="Local interview learning Agent.")
skills_app = typer.Typer(help="Hermes-style skill commands.")
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
    topic: str = typer.Option("RAG", "--topic", help="Interview topic"),
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
