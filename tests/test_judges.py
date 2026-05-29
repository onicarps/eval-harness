"""Tests for src/judges.py."""

from __future__ import annotations

import json
from pathlib import Path

from pytest_httpx import HTTPXMock

from src.judges import (
    DEFAULT_JUDGE_MODELS,
    JudgeModel,
    JudgeRegistry,
)


def _payload() -> dict:
    return {
        "data": [
            {
                "id": "meta/free-1",
                "name": "Free One",
                "context_length": 32000,
                "pricing": {"prompt": "0", "completion": "0"},
            },
            {
                "id": "paid/model",
                "name": "Paid",
                "context_length": 8192,
                "pricing": {"prompt": "0.001", "completion": "0.001"},
            },
            {
                "id": "google/free-2",
                "name": "Free Two",
                "context_length": 128000,
                "pricing": {"prompt": "0", "completion": "0"},
            },
        ]
    }


def test_default_models_nonempty() -> None:
    assert len(DEFAULT_JUDGE_MODELS) > 0


def test_fetch_filters_free_and_sorts(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://openrouter.ai/api/v1/models",
        method="GET",
        json=_payload(),
    )
    registry = JudgeRegistry(cache_path=tmp_path / "judges.json")
    models = registry.fetch(refresh=True)
    ids = [m.id for m in models]
    assert "paid/model" not in ids
    assert ids[0] == "google/free-2"


def test_fetch_uses_cache(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    cache = tmp_path / "judges.json"
    cache.write_text(
        json.dumps({"models": [{"id": "a/free", "name": "A", "context_length": 100, "free": True}]})
    )
    registry = JudgeRegistry(cache_path=cache)
    models = registry.list()
    assert models[0].id == "a/free"


def test_fetch_refresh_overwrites_cache(tmp_path: Path, httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url="https://openrouter.ai/api/v1/models",
        method="GET",
        json=_payload(),
    )
    cache = tmp_path / "judges.json"
    cache.write_text(
        json.dumps({"models": [{"id": "old", "name": "Old", "context_length": 1, "free": True}]})
    )
    registry = JudgeRegistry(cache_path=cache)
    models = registry.fetch(refresh=True)
    ids = [m.id for m in models]
    assert "old" not in ids


def test_judge_model_serialization() -> None:
    m = JudgeModel(id="x", name="X", context_length=100, free=True)
    d = m.model_dump()
    assert d["id"] == "x"
    assert d["free"] is True


def test_get_default_when_no_cache(tmp_path: Path) -> None:
    registry = JudgeRegistry(cache_path=tmp_path / "missing.json")
    models = registry.list()
    assert len(models) > 0


def test_to_json_lines(tmp_path: Path) -> None:
    registry = JudgeRegistry(cache_path=tmp_path / "j.json")
    out = registry.to_json()
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert all("id" in m for m in parsed)
