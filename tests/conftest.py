"""Pytest fixtures shared across the eval-harness test suite."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from src.db import Database
from src.models import EvalRecord, EvalRun


@pytest.fixture()
def temp_db(tmp_path: Path) -> Iterator[Database]:
    """Provide a fresh sqlite database for each test."""
    db = Database(tmp_path / "eval.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def sample_records() -> list[EvalRecord]:
    """Return a small set of EvalRecords for general use."""
    return [
        EvalRecord(input_text="What is 2+2?", output_text="4"),
        EvalRecord(
            input_text="Capital of France?",
            output_text="Paris",
            reference_text="Paris",
        ),
        EvalRecord(input_text="Name a color.", output_text="Octopus"),
    ]


@pytest.fixture()
def sample_run() -> EvalRun:
    """Return a fresh in-memory EvalRun model."""
    return EvalRun(config={"judge": "test"}, judge_model="judge/test")


@pytest.fixture()
def mock_judge_response() -> dict[str, object]:
    """Return a canonical mock judge JSON response."""
    return {
        "faithfulness": 0.9,
        "task_completion": 0.85,
        "reasoning": "good answer",
        "faithfulness_reasoning": "matches reference",
        "task_completion_reasoning": "completes the task",
    }


@pytest.fixture()
def openrouter_payload(mock_judge_response: dict[str, object]) -> dict[str, object]:
    """Return a mock OpenRouter chat completion payload wrapping the judge JSON."""
    return {
        "id": "chatcmpl-test",
        "model": "test/judge",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(mock_judge_response),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


@pytest.fixture()
def jsonl_records_file(tmp_path: Path) -> Path:
    """Write a tiny JSONL input file for ingestion tests."""
    p = tmp_path / "records.jsonl"
    p.write_text('{"input": "i1", "output": "o1"}\n{"input": "i2", "output": "o2"}\n')
    return p
