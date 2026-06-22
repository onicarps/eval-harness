"""Tests for src/models.py."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from src.models import (
    BUILTIN_RUBRIC_V1,
    EvalRecord,
    EvalResult,
    EvalRun,
    EvalSummary,
    JudgeCacheEntry,
    PassFail,
    RubricTemplate,
    RunStatus,
)


def test_eval_record_minimum_valid():
    rec = EvalRecord(input_text="hello", output_text="world")
    assert rec.record_id
    assert rec.input_text == "hello"
    assert rec.output_text == "world"
    assert rec.reference_text is None


def test_eval_record_with_metadata():
    rec = EvalRecord(
        input_text="i",
        output_text="o",
        reference_text="r",
        source_file="x.jsonl",
        metadata={"k": "v"},
    )
    assert rec.metadata == {"k": "v"}
    assert rec.source_file == "x.jsonl"


def test_eval_result_score_bounds():
    res = EvalResult(
        record_id="r1",
        run_id="run1",
        faithfulness=0.8,
        task_completion=0.6,
        combined_score=0.7,
        pass_fail=PassFail.PASS,
        judge_model="m",
    )
    assert res.combined_score == 0.7
    assert res.pass_fail == PassFail.PASS


def test_eval_result_invalid_score():
    with pytest.raises(ValidationError):
        EvalResult(
            record_id="r1",
            run_id="run1",
            faithfulness=1.5,
            task_completion=0.5,
            combined_score=0.5,
            pass_fail=PassFail.PASS,
            judge_model="m",
        )


def test_eval_result_negative_score():
    with pytest.raises(ValidationError):
        EvalResult(
            record_id="r1",
            run_id="run1",
            faithfulness=-0.1,
            task_completion=0.5,
            combined_score=0.5,
            pass_fail=PassFail.PASS,
            judge_model="m",
        )


def test_eval_run_defaults():
    run = EvalRun(config={})
    assert run.run_id
    assert run.status == RunStatus.RUNNING
    assert run.record_count == 0
    assert run.rubric_id == "faithfulness-v1"


def test_eval_run_completes():
    run = EvalRun(config={})
    run.status = RunStatus.COMPLETED
    run.completed_at = datetime.now(UTC)
    run.mean_score = 0.7
    run.pass_rate = 1.0
    assert run.status == RunStatus.COMPLETED


def test_judge_cache_entry():
    e = JudgeCacheEntry(
        cache_key="k",
        model_id="m",
        rubric_version="1.0",
        response={"faithfulness": 0.9, "task_completion": 0.8},
    )
    assert e.hits == 1
    assert e.response["faithfulness"] == 0.9


def test_rubric_template_builtin():
    assert BUILTIN_RUBRIC_V1.rubric_id == "faithfulness-v1"
    assert BUILTIN_RUBRIC_V1.version == "1.0"
    assert "faithfulness" in BUILTIN_RUBRIC_V1.prompt_template.lower()


def test_rubric_template_custom():
    r = RubricTemplate(
        rubric_id="x",
        version="1.0",
        prompt_template="Evaluate {input} vs {output} with ref {reference}",
    )
    assert r.rubric_id == "x"


def test_rubric_template_missing_placeholders():
    with pytest.raises(ValueError, match="missing required placeholders"):
        RubricTemplate(rubric_id="x", version="1.0", prompt_template="no placeholders here")


def test_rubric_template_partial_placeholders():
    with pytest.raises(ValueError, match="missing required placeholders"):
        RubricTemplate(rubric_id="x", version="1.0", prompt_template="only {input} present")


def test_pass_fail_enum_values():
    assert PassFail.PASS.value == "pass"
    assert PassFail.FAIL.value == "fail"


def test_run_status_values():
    assert RunStatus.RUNNING.value == "running"
    assert RunStatus.COMPLETED.value == "completed"
    assert RunStatus.FAILED.value == "failed"


def test_eval_summary():
    s = EvalSummary(
        run_id="r",
        total=10,
        passed=8,
        failed=2,
        pass_rate=0.8,
        mean_faithfulness=0.85,
        mean_task_completion=0.75,
        mean_combined=0.8,
        eval_time_seconds=12.5,
        judge_usage={"m1": 7, "m2": 3},
        errors=0,
    )
    assert s.total == 10
    assert s.passed + s.failed == s.total


def test_combined_score_validation_within_tolerance():
    res = EvalResult(
        record_id="r1",
        run_id="run1",
        faithfulness=0.8,
        task_completion=0.6,
        combined_score=0.7,
        pass_fail=PassFail.PASS,
        judge_model="m",
    )
    assert abs(res.combined_score - 0.7) < 1e-9
