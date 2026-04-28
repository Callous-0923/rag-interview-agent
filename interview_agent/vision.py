from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import AgentConfig
from .llm import LLMClient
from .markdown import FRONTMATTER_RE


IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageTask:
    note_path: Path
    note_rel: str
    note_id: str
    title: str
    image_path: Path
    image_ref: str
    image_index: int
    source_url: str
    author: str
    publish_date: str
    tags: list[str]
    topics: list[str]

    @property
    def task_id(self) -> str:
        raw = f"{self.note_rel}|{self.image_ref}|{self.image_index}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def run_vision_ingest(
    config: AgentConfig,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, object]:
    vision_dir = config.vision_dir or (config.project_root / "vision")
    output_dir = vision_dir / "xhs_image_notes"
    cache_dir = vision_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = list(find_xhs_image_tasks(config.notes_dir))
    selected = [task for task in tasks if force or not output_path(output_dir, task).exists()]
    if limit is not None:
        selected = selected[: max(0, limit)]
    if dry_run:
        return {
            "scanned": len(tasks),
            "pending": len([task for task in tasks if not output_path(output_dir, task).exists()]),
            "selected": len(selected),
            "processed": 0,
            "skipped": len(tasks) - len(selected),
            "errors": [],
            "output_dir": str(output_dir),
        }
    if not config.vision_enabled:
        return {
            "scanned": len(tasks),
            "pending": len(selected),
            "selected": len(selected),
            "processed": 0,
            "skipped": len(tasks),
            "errors": ["Vision ingest is disabled by config: set vision.enabled=true to process images."],
            "output_dir": str(output_dir),
        }

    llm = LLMClient.from_config(config)
    if not llm.enabled:
        return {
            "scanned": len(tasks),
            "pending": len(selected),
            "selected": len(selected),
            "processed": 0,
            "skipped": len(tasks) - len(selected),
            "errors": [f"LLM disabled: set {llm.api_key_env} and llm config first."],
            "output_dir": str(output_dir),
        }

    processed = 0
    errors: list[str] = []
    for task in selected:
        try:
            payload = load_or_extract_image_text(llm, task, cache_dir, force=force, usage_log=config.vision_usage_log)
            output_path(output_dir, task).write_text(render_vision_markdown(task, payload), encoding="utf-8", newline="\n")
            processed += 1
        except Exception as exc:
            errors.append(f"{task.image_path}: {type(exc).__name__}: {exc}")

    return {
        "scanned": len(tasks),
        "pending": len([task for task in tasks if not output_path(output_dir, task).exists()]),
        "selected": len(selected),
        "processed": processed,
        "skipped": len(tasks) - len(selected),
        "errors": errors,
        "output_dir": str(output_dir),
    }


def find_xhs_image_tasks(notes_dir: Path) -> list[ImageTask]:
    tasks: list[ImageTask] = []
    for note_path in sorted(notes_dir.glob("*.md")):
        text = note_path.read_text(encoding="utf-8", errors="replace")
        meta, body = _split_meta(text)
        note_rel = note_path.relative_to(notes_dir.parent).as_posix()
        title = str(meta.get("title") or note_path.stem)
        note_id = str(meta.get("xhs_note_id") or note_path.stem)
        refs = _image_refs(meta, body)
        for idx, ref in enumerate(refs):
            image_path = (note_path.parent / ref).resolve()
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES or not image_path.exists():
                continue
            tasks.append(
                ImageTask(
                    note_path=note_path,
                    note_rel=note_rel,
                    note_id=note_id,
                    title=title,
                    image_path=image_path,
                    image_ref=ref,
                    image_index=idx,
                    source_url=str(meta.get("xhs_url") or ""),
                    author=str(meta.get("author") or ""),
                    publish_date=str(meta.get("publish_date") or ""),
                    tags=_as_list(meta.get("tags")),
                    topics=_as_list(meta.get("topic") or meta.get("topics")),
                )
            )
    return tasks


def load_or_extract_image_text(
    llm: LLMClient,
    task: ImageTask,
    cache_dir: Path,
    force: bool = False,
    usage_log: Path | None = None,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{task.task_id}.json"
    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    payload, metadata = llm.complete_vision_json_with_metadata(
        task.image_path,
        prompt=_vision_prompt(task),
        max_output_tokens=6000,
    )
    append_usage_log(usage_log, task, metadata)
    if payload is None:
        raise RuntimeError("vision model returned no parseable text")
    normalized = normalize_payload(payload)
    cache_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    return normalized


def append_usage_log(path: Path | None, task: ImageTask, metadata: dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    usage = metadata.get("usage") or {}
    record = {
        "task_id": task.task_id,
        "title": task.title,
        "image_path": str(task.image_path),
        "image_ref": task.image_ref,
        "model": metadata.get("model"),
        "status": metadata.get("status"),
        "incomplete_details": metadata.get("incomplete_details"),
        "usage": usage,
    }
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    text = str(payload.get("text") or payload.get("ocr_text") or "").strip()
    key_points = payload.get("key_points") or []
    uncertain = payload.get("uncertain_parts") or []
    if isinstance(key_points, str):
        key_points = [key_points]
    if isinstance(uncertain, str):
        uncertain = [uncertain]
    return {
        "text": text,
        "key_points": [str(item).strip() for item in key_points if str(item).strip()],
        "uncertain_parts": [str(item).strip() for item in uncertain if str(item).strip()],
    }


def render_vision_markdown(task: ImageTask, payload: dict[str, Any]) -> str:
    title = f"图片OCR：{task.title} #{task.image_index + 1}"
    metadata = {
        "doc_id": f"xhs_vision_{task.task_id}",
        "source": "xiaohongshu",
        "source_type": "vision_ocr",
        "title": title,
        "source_title": task.title,
        "source_note_id": task.note_id,
        "source_note_file": task.note_rel,
        "source_image": task.image_ref,
        "source_image_path": str(task.image_path),
        "xhs_url": task.source_url,
        "author": task.author,
        "publish_date": task.publish_date,
        "tags": _dedupe(task.tags + ["vision_ocr", "图片OCR"]),
        "topic": _dedupe(task.topics + ["图片资料"]),
        "generated_by": "agent vision ingest",
    }
    lines = ["---", yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False).strip(), "---", ""]
    lines.extend(
        [
            f"# {title}",
            "",
            f"- 原笔记：{task.title}",
            f"- 原文件：{task.note_rel}",
            f"- 图片：{task.image_ref}",
            "",
            "## 图片文字",
            "",
            str(payload.get("text") or "").strip() or "[未识别到清晰文字]",
            "",
            "## 关键点",
            "",
        ]
    )
    key_points = payload.get("key_points") or []
    lines.extend([f"- {point}" for point in key_points] or ["- [无]"])
    lines.extend(["", "## 不确定内容", ""])
    uncertain = payload.get("uncertain_parts") or []
    lines.extend([f"- {item}" for item in uncertain] or ["- [无]"])
    lines.append("")
    return "\n".join(lines)


def output_path(output_dir: Path, task: ImageTask) -> Path:
    safe_stem = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", task.note_path.stem)[:80]
    return output_dir / f"{safe_stem}_img{task.image_index + 1}_{task.task_id}.md"


def _vision_prompt(task: ImageTask) -> str:
    return (
        "你是知识库入库前的图片文字识别器。请识别图片中的文字，并尽量保留标题、编号、层级和换行。\n"
        "不要编造看不清的内容；看不清的位置用[不清晰]标注。\n"
        "如果图片包含架构图、流程图、表格或代码，也要把可读文字和结构关系转成文本。\n"
        "只输出 JSON，不要输出 Markdown，不要加解释。\n"
        "JSON schema:\n"
        "{\n"
        '  "text": "完整OCR文本，保留层级和换行",\n'
        '  "key_points": ["可用于RAG检索的关键知识点"],\n'
        '  "uncertain_parts": ["不清晰或不确定的位置"]\n'
        "}\n\n"
        f"原笔记标题：{task.title}\n"
        f"原图序号：{task.image_index + 1}\n"
    )


def _split_meta(text: str) -> tuple[dict[str, Any], str]:
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    return yaml.safe_load(match.group(1)) or {}, text[match.end() :]


def _image_refs(meta: dict[str, Any], body: str) -> list[str]:
    refs = _as_list(meta.get("assets"))
    refs.extend(match.group(1).strip() for match in IMAGE_RE.finditer(body))
    return _dedupe(refs)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out
