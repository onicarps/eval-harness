"""Tests for the calibrate module."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from src.calibrate import (
    CalibrationResult,
    CalibrationRunner,
    CalibrationSummary,
    compute_agreement_metrics,
)
from src.db import Database
from src.models import EvalRecord, EvalResult, EvalRun, PassFail, RubricTemplate


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_records() -> list[EvalRecord]:
    """Three simple records for calibration testing."""
    return [
        EvalRecord(input_text="What is 2+2?", output_text="4"),
        EvalRecord(input_text="What is the capital of France?", output_text="Paris"),
        EvalRecord(input_text="Write a poem about cats.", output_text="Meow meow meow."),
    ]


@pytest.fixture
def sample_results_multi_judge() -> list[EvalResult]:
    """Multiple results per record from different judges.
    
    Record 0: high agreement (scores close together).
    Record 1: medium disagreement.
    Record 2: high disagreement (scores spread wide).
    """
    results: list[EvalResult] = []
    # Record 0: high agreement
    for judge, faith, task in [
        ("judge-a", 0.95, 0.9),
        ("judge-b", 0.9, 0.9),
        ("judge-c", 0.9, 0.85),
    ]:
        combined = 0.5 * faith + 0.5 * task
        results.append(
            EvalResult(
                record_id="rec-0",
                run_id="test-run",
                rubric_id="faithfulness-v1",
                rubric_version="1.0",
                faithfulness=faith,
                task_completion=task,
                combined_score=combined,
                pass_fail=PassFail.PASS if combined >= 0.7 else PassFail.FAIL,
                judge_model=judge,
                judge_tried=[judge],
                reasoning="test",
                faithfulness_reasoning="test",
                task_completion_reasoning="test",
            )
        )
    # Record 1: medium disagreement
    for judge, faith, task in [
        ("judge-a", 0.85, 0.85),
        ("judge-b", 0.7, 0.7),
        ("judge-c", 0.75, 0.75),
    ]:
        combined = 0.5 * faith + 0.5 * task
        results.append(
            EvalResult(
                record_id="rec-1",
                run_id="test-run",
                rubric_id="faithfulness-v1",
                rubric_version="1.0",
                faithfulness=faith,
                task_completion=task,
                combined_score=combined,
                pass_fail=PassFail.PASS if combined >= 0.7 else PassFail.FAIL,
                judge_model=judge,
                judge_tried=[judge],
                reasoning="test",
                faithfulness_reasoning="test",
                task_completion_reasoning="test",
            )
        )
    # Record 2: high disagreement
    for judge, faith, task in [
        ("judge-a", 1.0, 1.0),
        ("judge-b", 0.5, 0.4),
        ("judge-c", 0.6, 0.5),
    ]:
        combined = 0.5 * faith + 0.5 * task
        results.append(
            EvalResult(
                record_id="rec-2",
                run_id="test-run",
                rubric_id="faithfulness-v1",
                rubric_version="1.0",
                faithfulness=faith,
                task_completion=task,
                combined_score=combined,
                pass_fail=PassFail.PASS if combined >= 0.7 else PassFail.FAIL,
                judge_model=judge,
                judge_tried=[judge],
                reasoning="test",
                faithfulness_reasoning="test",
                task_completion_reasoning="test",
            )
        )
    return results


@pytest.fixture
def db(tmp_path: Path) -> Database:
    """Temporary database for calibration tests."""
    return Database(tmp_path / "test.db")


# ── CalibrationResult ────────────────────────────────────────────────────────

class TestCalibrationResult:
    def test_from_results_single_record(self, sample_results_multi_judge):
        """CalibrationResult groups multi-judge results for one record."""
        rec_id = sample_results_multi_judge[0].record_id
        cr = CalibrationResult.from_results(sample_results_multi_judge, rec_id, "test-run")
        assert cr.record_id == rec_id
        assert cr.run_id == "test-run"
        assert len(cr.scores) == 3
        # Check that we have scores from all 3 judges
        judges = {s.judge for s in cr.scores}
        assert judges == {"judge-a", "judge-b", "judge-c"}

    def test_from_results_no_results(self):
        """Empty results produce empty scores."""
        cr = CalibrationResult.from_results([], "some-record", "test-run")
        assert cr.record_id == "some-record"
        assert cr.run_id == "test-run"
        assert cr.scores == []

    def test_from_results_preserves_record_id(self):
        """All scores should share the same record_id."""
        rec_id = "unique-record-id"
        results = [
            EvalResult(
                record_id=rec_id,
                run_id="test-run",
                rubric_id="faithfulness-v1",
                rubric_version="1.0",
                faithfulness=0.8,
                task_completion=0.9,
                combined_score=0.85,
                pass_fail=PassFail.PASS,
                judge_model="judge-a",
                judge_tried=["judge-a"],
            ),
        ]
        cr = CalibrationResult.from_results(results, rec_id, "test-run")
        assert cr.record_id == rec_id
        assert all(s.record_id == rec_id for s in cr.scores)


# ── compute_agreement_metrics ────────────────────────────────────────────────

class TestComputeAgreementMetrics:
    def test_perfect_agreement(self):
        """All judges agree: std_dev should be 0."""
        scores = [0.8, 0.8, 0.8]
        result = compute_agreement_metrics(scores)
        assert result["std_dev"] == pytest.approx(0.0)
        assert result["mean_score"] == pytest.approx(0.8)
        assert result["min_score"] == pytest.approx(0.8)
        assert result["max_score"] == pytest.approx(0.8)

    def test_disagreement(self):
        """Judges disagree: std_dev > 0."""
        scores = [1.0, 0.5, 0.3]
        result = compute_agreement_metrics(scores)
        assert result["std_dev"] > 0.2  # should be at least 0.2
        assert result["mean_score"] == pytest.approx(0.6)
        assert result["min_score"] == pytest.approx(0.3)
        assert result["max_score"] == pytest.approx(1.0)

    def test_two_judges(self):
        """Two judges with slight disagreement."""
        scores = [0.9, 0.8]
        result = compute_agreement_metrics(scores)
        # sample std_dev of [0.9, 0.8] = sqrt(0.005) ≈ 0.0707
        assert result["std_dev"] == pytest.approx(0.0707, abs=0.001)

    def test_single_judge(self):
        """Single judge: std_dev should be 0."""
        scores = [0.75]
        result = compute_agreement_metrics(scores)
        assert result["std_dev"] == pytest.approx(0.0)

    def test_empty_scores(self):
        """Empty scores return defaults."""
        result = compute_agreement_metrics([])
        assert result["std_dev"] == 0.0
        assert result["mean_score"] == 0.0
        assert result["min_score"] == 0.0
        assert result["max_score"] == 0.0

    def test_pass_fail_agreement(self):
        """Pass/fail agreement: all above threshold."""
        scores = [0.8, 0.9, 0.85]
        result = compute_agreement_metrics(scores, threshold=0.7)
        assert result["pass_agreement"] is True
        assert result["pass_count"] == 3
        assert result["fail_count"] == 0

    def test_pass_fail_disagreement(self):
        """Pass/fail disagreement: mixed results."""
        scores = [0.9, 0.5, 0.8]
        result = compute_agreement_metrics(scores, threshold=0.7)
        assert result["pass_agreement"] is False
        assert result["pass_count"] == 2
        assert result["fail_count"] == 1

    def test_pass_fail_all_fail(self):
        """All below threshold."""
        scores = [0.3, 0.4, 0.2]
        result = compute_agreement_metrics(scores, threshold=0.7)
        assert result["pass_agreement"] is True  # all agree on FAIL
        assert result["pass_count"] == 0
        assert result["fail_count"] == 3


# ── CalibrationSummary ───────────────────────────────────────────────────────

class TestCalibrationSummary:
    def test_from_results_basic(self, sample_records, sample_results_multi_judge):
        """Summary aggregates across all records."""
        results = sample_results_multi_judge
        summary = CalibrationSummary.from_results(results, "test-run", ["judge-a", "judge-b", "judge-c"])
        assert summary.run_id == "test-run"
        assert summary.total_records == 3
        assert summary.total_judges == 3
        assert summary.mean_std_dev > 0
        # max should be >= mean (with varying disagreement levels)
        assert summary.max_std_dev >= summary.mean_std_dev

    def test_from_results_empty(self):
        """Empty results produce zeroed summary."""
        summary = CalibrationSummary.from_results([], "test-run", ["judge-a"])
        assert summary.total_records == 0
        assert summary.mean_std_dev == 0.0
        assert summary.pass_agreement_rate == 1.0  # vacuously true

    def test_from_results_perfect_agreement(self):
        """All judges agree on all records."""
        results: list[EvalResult] = []
        for i in range(3):
            for judge in ["judge-a", "judge-b", "judge-c"]:
                results.append(
                    EvalResult(
                        record_id=f"rec-{i}",
                        run_id="test-run",
                        rubric_id="faithfulness-v1",
                        rubric_version="1.0",
                        faithfulness=0.8,
                        task_completion=0.8,
                        combined_score=0.8,
                        pass_fail=PassFail.PASS,
                        judge_model=judge,
                        judge_tried=[judge],
                    )
                )
        summary = CalibrationSummary.from_results(results, "test-run", ["judge-a", "judge-b", "judge-c"])
        assert summary.mean_std_dev == pytest.approx(0.0)
        assert summary.pass_agreement_rate == pytest.approx(1.0)

    def test_disagreement_list(self, sample_results_multi_judge):
        """Disagreement list should contain records with std_dev > threshold."""
        summary = CalibrationSummary.from_results(
            sample_results_multi_judge, "test-run",
            ["judge-a", "judge-b", "judge-c"],
            disagreement_threshold=0.05,
        )
        # Record 1 and 2 should have std_dev >= 0.05
        assert len(summary.disagreements) >= 2
        for d in summary.disagreements:
            assert d["std_dev"] >= 0.05


# ── CalibrationRunner ────────────────────────────────────────────────────────

class TestCalibrationRunner:
    def test_init(self, db):
        """Runner initializes with DB and config."""
        runner = CalibrationRunner(
            db=db,
            api_key="test-key",
            judges=["judge-a", "judge-b"],
        )
        assert runner.judges == ["judge-a", "judge-b"]

    def test_init_empty_judges(self, db):
        """Runner rejects empty judge list."""
        with pytest.raises(ValueError, match="at least one judge"):
            CalibrationRunner(db=db, api_key="test-key", judges=[])

    def test_compute_agreement(self, sample_results_multi_judge, tmp_path: Path):
        """Runner can compute agreement from raw results."""
        runner = CalibrationRunner(
            db=Database(tmp_path / "test.db"),
            api_key="test-key",
            judges=["judge-a", "judge-b"],
        )
        summary = CalibrationSummary.from_results(
            sample_results_multi_judge, "test-run",
            ["judge-a", "judge-b", "judge-c"],
        )
        assert summary.total_records == 3

    def test_results_sorted_by_std_dev(self, sample_results_multi_judge):
        """Disagreements should be sorted by std_dev descending."""
        summary = CalibrationSummary.from_results(
            sample_results_multi_judge, "test-run",
            ["judge-a", "judge-b", "judge-c"],
            disagreement_threshold=0.0,
        )
        std_devs = [d["std_dev"] for d in summary.disagreements]
        assert std_devs == sorted(std_devs, reverse=True)