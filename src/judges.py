"""OpenRouter free-judge model discovery and caching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
DEFAULT_CACHE_PATH = Path.home() / ".eval-harness" / "judges.json"


class JudgeModel(BaseModel):
    """A judge model description discovered from OpenRouter."""

    id: str
    name: str
    context_length: int = 0
    free: bool = True


DEFAULT_JUDGE_MODELS: list[JudgeModel] = [
    JudgeModel(
        id="meta-llama/llama-3.1-8b-instruct:free", name="Llama 3.1 8B", context_length=128000
    ),
    JudgeModel(
        id="google/gemini-flash-1.5:free", name="Gemini Flash 1.5", context_length=1_000_000
    ),
    JudgeModel(id="mistralai/mistral-7b-instruct:free", name="Mistral 7B", context_length=32768),
    JudgeModel(
        id="nousresearch/hermes-3-llama-3.1-405b:free", name="Hermes 3 405B", context_length=128000
    ),
]


def _is_free(model: dict[str, Any]) -> bool:
    """Return True when both prompt and completion prices are zero."""
    pricing = model.get("pricing") or {}
    try:
        return float(pricing.get("prompt", 0)) == 0 and float(pricing.get("completion", 0)) == 0
    except (TypeError, ValueError):
        return False


class JudgeRegistry:
    """Caches and queries the OpenRouter free-model catalog."""

    def __init__(self, cache_path: str | Path | None = None) -> None:
        """Initialize the registry.

        Args:
            cache_path: Optional path for the on-disk judges cache file.
        """
        self.cache_path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH

    def list(self) -> list[JudgeModel]:
        """Return the cached judge list, or built-in defaults if no cache exists.

        Falls back to built-in defaults when the cache file is missing,
        corrupted, or contains no usable entries.
        """
        if not self.cache_path.exists():
            return list(DEFAULT_JUDGE_MODELS)
        try:
            data = json.loads(self.cache_path.read_text())
            models = [JudgeModel(**m) for m in data.get("models", [])]
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return list(DEFAULT_JUDGE_MODELS)
        return models or list(DEFAULT_JUDGE_MODELS)

    def fetch(self, refresh: bool = False, api_key: str | None = None) -> list[JudgeModel]:  # type: ignore[valid-type]
        """Fetch and cache the latest free model list.

        Args:
            refresh: When True, always fetch from the network and overwrite the cache.
            api_key: Optional API key for the OpenRouter request.
        """
        if not refresh and self.cache_path.exists():
            return self.list()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = httpx.get(OPENROUTER_MODELS_URL, headers=headers, timeout=30.0)
        resp.raise_for_status()
        body = resp.json()
        models: list[JudgeModel] = []
        for m in body.get("data", []):
            if not _is_free(m):
                continue
            models.append(
                JudgeModel(
                    id=m.get("id", ""),
                    name=m.get("name", m.get("id", "")),
                    context_length=int(m.get("context_length") or 0),
                    free=True,
                )
            )
        models.sort(key=lambda m: m.context_length, reverse=True)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps({"models": [m.model_dump() for m in models]}, indent=2)
        )
        return models

    def to_json(self) -> str:
        """Return the current judge list as a JSON array string."""
        return json.dumps([m.model_dump() for m in self.list()], indent=2)
