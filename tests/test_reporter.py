"""Tests for src/reporter.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.models import EvalResult, EvalRun, PassFail
from src.reporter import (
    ascii_histogram,
    build_summary,
    export_results,
    judge_usage,
    render_table,
)


def _results(run_id: str) -> list[EvalResult]:
    return [
        EvalResult(
            record_id="r1",
            run_id=run_id,
            faithfulness=0.9,
            task_completion=0.8,
            combined_score=0.85,
            pass_fail=PassFail.PASS,
            judge_model="m1",
        ),
        EvalResult(
            record_id="r2",
            run_id=run_id,
            faithfulness=0.4,
            task_completion=0.5,
            combined_score=0.45,
            pass_fail=PassFail.FAIL,
            judge_model="m2",
        ),
        EvalResult(
            record_id="r3",
            run_id=run_id,
            faithfulness=0.0,
            task_completion=0.0,
            combined_score=0.0,
            pass_fail=PassFail.FAIL,
            judge_model="m1",
            error="boom",
        ),
    ]


def test_build_summary_counts() -> None:
    run = EvalRun(config={})
    run.eval_time_seconds = 5.0
    s = build_summary(run, _results(run.run_id))
    assert s.total == 3
    assert s.passed == 1
    assert s.failed == 2
    assert s.errors == 1
    assert 0 <= s.pass_rate <= 1


def test_judge_usage_counts() -> None:
    run = EvalRun(config={})
    usage = judge_usage(_results(run.run_id))
    assert usage["m1"] == 2
    assert usage["m2"] == 1


def test_ascii_histogram_buckets() -> None:
    out = ascii_histogram([0.0, 0.1, 0.2, 0.9, 1.0], buckets=5)
    assert isinstance(out, str)
    assert "[" in out


def test_ascii_histogram_empty() -> None:
    out = ascii_histogram([], buckets=5)
    assert isinstance(out, str)


def test_render_table_returns_string() -> None:
    run = EvalRun(config={})
    s = build_summary(run, _results(run.run_id))
    s.judge_usage = judge_usage(_results(run.run_id))
    table = render_table(s)
    assert "Summary" in table or "pass" in table.lower()


def test_export_results_json(tmp_path: Path) -> None:
    run = EvalRun(config={})
    results = _results(run.run_id)
    out = tmp_path / "x.json"
    export_results(run, results, out, fmt="json")
    data = json.loads(out.read_text())
    assert data["run_id"] == run.run_id
    assert len(data["results"]) == 3


def test_export_results_csv(tmp_path: Path) -> None:
    run = EvalRun(config={})
    results = _results(run.run_id)
    out = tmp_path / "x.csv"
    export_results(run, results, out, fmt="csv")
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 3
    assert "faithfulness" in rows[0]


def test_summary_zero_results() -> None:
    run = EvalRun(config={})
    s = build_summary(run, [])
    assert s.total == 0
    assert s.pass_rate == 0.0
    assert s.mean_combined == 0.0
