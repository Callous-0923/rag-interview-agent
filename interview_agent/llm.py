from __future__ import annotations

import json
import os
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AgentConfig


@dataclass
class LLMClient:
    provider: str
    api_base: str
    api_key_env: str
    model: str
    timeout: float = 60.0

    @classmethod
    def from_config(cls, config: AgentConfig) -> "LLMClient":
        raw = config.llm or {}
        return cls(
            provider=str(raw.get("provider") or "offline"),
            api_base=str(raw.get("api_base") or ""),
            api_key_env=str(raw.get("api_key_env") or "ARK_API_KEY"),
            model=str(raw.get("model") or ""),
        )

    @property
    def enabled(self) -> bool:
        return (
            self.provider != "offline"
            and bool(self.api_base)
            and bool(self.model)
            and bool(os.getenv(self.api_key_env))
        )

    def complete(self, system: str, user: str, max_output_tokens: int = 2000) -> str:
        if not self.enabled:
            return ""
        try:
            from openai import OpenAI
        except ImportError:
            return ""

        client = OpenAI(
            base_url=self.api_base,
            api_key=os.getenv(self.api_key_env),
            timeout=self.timeout,
        )
        try:
            response = client.responses.create(
                model=self.model,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user}],
                    },
                ],
                max_output_tokens=max_output_tokens,
            )
            text = getattr(response, "output_text", "")
            if text:
                return str(text).strip()
            return _extract_response_text(response)
        except Exception as exc:
            return f"[LLM_ERROR] {type(exc).__name__}: {exc}"

    def complete_json(self, system: str, user: str, max_output_tokens: int = 1000) -> dict[str, Any] | None:
        text = self.complete(system, user, max_output_tokens=max_output_tokens)
        if not text or text.startswith("[LLM_ERROR]"):
            return None
        cleaned = _strip_code_fence(text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1])
                except json.JSONDecodeError:
                    return None
        return None

    def complete_vision_json(
        self,
        image_path: Path,
        prompt: str,
        max_output_tokens: int = 6000,
        model: str | None = None,
    ) -> dict[str, Any] | None:
        payload, _metadata = self.complete_vision_json_with_metadata(
            image_path,
            prompt,
            max_output_tokens=max_output_tokens,
            model=model,
        )
        return payload

    def complete_vision_json_with_metadata(
        self,
        image_path: Path,
        prompt: str,
        max_output_tokens: int = 6000,
        model: str | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        text, metadata = self.complete_vision_with_metadata(
            image_path,
            prompt,
            max_output_tokens=max_output_tokens,
            model=model,
        )
        if not text or text.startswith("[LLM_ERROR]"):
            return None, metadata
        cleaned = _strip_code_fence(text)
        try:
            return json.loads(cleaned), metadata
        except json.JSONDecodeError:
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(cleaned[start : end + 1]), metadata
                except json.JSONDecodeError:
                    return None, metadata
        return {"text": cleaned, "key_points": [], "uncertain_parts": []}, metadata

    def complete_vision(
        self,
        image_path: Path,
        prompt: str,
        max_output_tokens: int = 6000,
        model: str | None = None,
    ) -> str:
        text, _metadata = self.complete_vision_with_metadata(
            image_path,
            prompt,
            max_output_tokens=max_output_tokens,
            model=model,
        )
        return text

    def complete_vision_with_metadata(
        self,
        image_path: Path,
        prompt: str,
        max_output_tokens: int = 6000,
        model: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if not self.enabled:
            return "", {}
        try:
            from openai import OpenAI
        except ImportError:
            return "", {}

        client = OpenAI(
            base_url=self.api_base,
            api_key=os.getenv(self.api_key_env),
            timeout=max(self.timeout, 120.0),
        )
        try:
            response = client.responses.create(
                model=model or self.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": _image_to_data_url(image_path)},
                            {"type": "input_text", "text": prompt},
                        ],
                    }
                ],
                max_output_tokens=max_output_tokens,
            )
            text = getattr(response, "output_text", "")
            metadata = _response_metadata(response)
            if text:
                return str(text).strip(), metadata
            return _extract_response_text(response), metadata
        except Exception as exc:
            return f"[LLM_ERROR] {type(exc).__name__}: {exc}", {"error": f"{type(exc).__name__}: {exc}"}


def _extract_response_text(response: Any) -> str:
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", "")
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp" if suffix == ".webp" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def _response_metadata(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    incomplete = getattr(response, "incomplete_details", None)
    return {
        "id": getattr(response, "id", ""),
        "model": getattr(response, "model", ""),
        "status": getattr(response, "status", ""),
        "incomplete_details": _to_plain(incomplete),
        "usage": _to_plain(usage),
    }


def _to_plain(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {str(k): _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
