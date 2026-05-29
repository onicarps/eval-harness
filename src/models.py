"""Pydantic data models for eval-harness."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PassFail(StrEnum):
    """Pass/fail result of evaluation."""

    PASS = "pass"
    FAIL = "fail"


class RunStatus(StrEnum):
    """Lifecycle status of an evaluation run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _new_id() -> str:
    """Return a fresh UUID4 string."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(UTC)


class EvalRecord(BaseModel):
    """A single input/output pair to be evaluated.

    Attributes:
        record_id: Unique record identifier.
        run_id: Parent run identifier (assigned at insert time).
        input_text: Raw user input prompt.
        output_text: Model response text.
        reference_text: Optional ground-truth reference.
        source_file: Optional source filename for traceability.
        metadata: Arbitrary user metadata dictionary.
        created_at: UTC creation time.
    """

    model_config = ConfigDict(extra="ignore")

    record_id: str = Field(default_factory=_new_id)
    run_id: str | None = None
    input_text: str
    output_text: str
    reference_text: str | None = None
    source_file: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)


class EvalResult(BaseModel):
    """Judge-produced evaluation result for a single record."""

    model_config = ConfigDict(extra="ignore")

    result_id: str = Field(default_factory=_new_id)
    record_id: str
    run_id: str
    rubric_id: str = "faithfulness-v1"
    rubric_version: str = "1.0"
    faithfulness: float
    task_completion: float
    combined_score: float
    pass_fail: PassFail
    reasoning: str = ""
    faithfulness_reasoning: str = ""
    task_completion_reasoning: str = ""
    judge_model: str
    judge_fallbacks: int = 0
    judge_tried: list[str] = Field(default_factory=list)
    tokens_estimated: int | None = None
    evaluated_at: datetime = Field(default_factory=_utcnow)
    error: str | None = None

    @field_validator("faithfulness", "task_completion", "combined_score")
    @classmethod
    def _score_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"score must be in [0, 1], got {v}")
        return v


class EvalRun(BaseModel):
    """A single evaluation run, grouping a set of records and results."""

    model_config = ConfigDict(extra="ignore")

    run_id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_utcnow)
    config: dict[str, Any] = Field(default_factory=dict)
    record_count: int = 0
    rubric_id: str = "faithfulness-v1"
    judge_model: str | None = None
    status: RunStatus = RunStatus.RUNNING
    completed_at: datetime | None = None
    mean_score: float | None = None
    pass_rate: float | None = None
    eval_time_seconds: float | None = None


class JudgeCacheEntry(BaseModel):
    """Cached judge response keyed by (model_id, rubric_version, input/output hash)."""

    cache_key: str
    model_id: str
    rubric_version: str
    response: dict[str, Any]
    created_at: datetime = Field(default_factory=_utcnow)
    hits: int = 1


class RubricTemplate(BaseModel):
    """Rubric prompt template used to instruct the judge."""

    rubric_id: str
    version: str
    prompt_template: str
    description: str = ""


BUILTIN_RUBRIC_V1 = RubricTemplate(
    rubric_id="faithfulness-v1",
    version="1.0",
    description="Dual-dimension rubric: faithfulness + task completion.",
    prompt_template=(
        "You are an impartial evaluator. Score the assistant's output on two "
        "dimensions: FAITHFULNESS (does it stay grounded in the input/reference "
        "without hallucination?) and TASK_COMPLETION (does it satisfy what was "
        "asked?). Each dimension is a float in [0.0, 1.0].\n\n"
        "Return STRICT JSON only, with the following keys: "
        '{"faithfulness": float, "task_completion": float, '
        '"faithfulness_reasoning": str, "task_completion_reasoning": str, '
        '"reasoning": str}.\n\n'
        "INPUT:\n{input}\n\nOUTPUT:\n{output}\n\nREFERENCE:\n{reference}\n"
    ),
)


class EvalSummary(BaseModel):
    """Summary statistics for a completed run."""

    run_id: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    mean_faithfulness: float
    mean_task_completion: float
    mean_combined: float
    eval_time_seconds: float
    judge_usage: dict[str, int] = Field(default_factory=dict)
    errors: int = 0
