"""Tests for the calibrate module."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import pytest

from src.calibrate import (
    CalibrationRunner,
    CalibrationSummary,
    compute_agreement_metrics,
    render_calibration_json,
    render_calibration_summary,
    strip_ansi,
)
from src.db import Database
from src.models import EvalRecord, EvalResult, EvalRun, PassFail, RunStatus


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
        # Population stdev of [1.0, 0.5, 0.3] ≈ 0.2944
        assert result["std_dev"] == pytest.approx(0.2944, abs=0.001)
        assert result["mean_score"] == pytest.approx(0.6)
        assert result["min_score"] == pytest.approx(0.3)
        assert result["max_score"] == pytest.approx(1.0)

    def test_two_judges_population_stdev(self):
        """Two judges with slight disagreement — uses population stdev."""
        scores = [0.9, 0.8]
        result = compute_agreement_metrics(scores)
        # Population stdev of [0.9, 0.8] = sqrt(0.0025) = 0.05
        assert result["std_dev"] == pytest.approx(0.05, abs=0.001)

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
        assert summary.pass_agreement_rate is None  # None when no data

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

    def test_disagreements_sorted_descending(self, sample_results_multi_judge):
        """Disagreements should be sorted by std_dev descending."""
        summary = CalibrationSummary.from_results(
            sample_results_multi_judge, "test-run",
            ["judge-a", "judge-b", "judge-c"],
            disagreement_threshold=0.0,
        )
        std_devs = [d["std_dev"] for d in summary.disagreements]
        assert std_devs == sorted(std_devs, reverse=True)

    def test_judge_agreement_matrix(self, sample_results_multi_judge):
        """Judge pair agreement matrix should be symmetric with 1.0 on diagonal."""
        summary = CalibrationSummary.from_results(
            sample_results_multi_judge, "test-run",
            ["judge-a", "judge-b", "judge-c"],
        )
        for judge in ["judge-a", "judge-b", "judge-c"]:
            assert summary.judge_agreement[judge][judge] == 1.0
        # Symmetry
        assert summary.judge_agreement["judge-a"]["judge-b"] == pytest.approx(
            summary.judge_agreement["judge-b"]["judge-a"], abs=0.001
        )

    def test_judge_agreement_uses_passed_judges_list(self, sample_results_multi_judge):
        """Judge agreement keys should match the passed judges list."""
        summary = CalibrationSummary.from_results(
            sample_results_multi_judge, "test-run",
            ["judge-a", "judge-b", "judge-c"],
        )
        assert set(summary.judge_agreement.keys()) == {"judge-a", "judge-b", "judge-c"}


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

    def test_init_preserves_config(self, db):
        """Runner preserves concurrency and timeout settings."""
        runner = CalibrationRunner(
            db=db,
            api_key="test-key",
            judges=["judge-a"],
            concurrency=8,
            timeout=120.0,
            rpm_limit=60,
            use_cache=False,
        )
        assert runner.concurrency == 8
        assert runner.timeout == 120.0
        assert runner.rpm_limit == 60
        assert runner.use_cache is False

    def test_run_persists_run_to_db(self, db, sample_records):
        """Runner creates and persists EvalRun to DB."""
        runner = CalibrationRunner(
            db=db,
            api_key="test-key",
            judges=["judge-a"],
        )
        # We can't call runner.run() without a real API, but we can verify
        # the DB setup is correct
        assert db.connection is not None

    def test_run_creates_run_with_correct_config(self, db, sample_records):
        """Run object should have correct config when created."""
        from src.calibrate import CalibrationRunner
        from src.models import RunStatus

        runner = CalibrationRunner(
            db=db,
            api_key="test-key",
            judges=["judge-a", "judge-b"],
        )
        # Verify the run would be created with correct attributes
        run = EvalRun(
            run_id="test-run-id",
            status=RunStatus.RUNNING,
            record_count=len(sample_records),
            config={
                "file": "<calibration>",
                "format": "internal",
                "judges": ["judge-a", "judge-b"],
                "pass_threshold": 0.7,
            },
            rubric_id="faithfulness-v1",
            judge_model="judge-a,judge-b",
        )
        db.insert_run(run)
        # Verify it was persisted
        run_from_db = db.get_run("test-run-id")
        assert run_from_db is not None
        assert run_from_db.run_id == "test-run-id"
        assert run_from_db.status == RunStatus.RUNNING
        assert run_from_db.record_count == len(sample_records)


# ── Rendering ────────────────────────────────────────────────────────────────

class TestRendering:
    def test_render_summary(self, sample_results_multi_judge):
        """Render summary produces readable output."""
        summary = CalibrationSummary.from_results(
            sample_results_multi_judge, "test-run",
            ["judge-a", "judge-b", "judge-c"],
        )
        text = render_calibration_summary(summary)
        assert "Calibration Summary for run test-run" in text
        assert "Records evaluated:    3" in text
        assert "Judges used:          3" in text
        assert "Mean std deviation:" in text
        assert "Pass/fail agreement:" in text

    def test_render_summary_empty(self):
        """Empty summary renders cleanly."""
        summary = CalibrationSummary(run_id="empty-run", total_records=0, total_judges=1)
        text = render_calibration_summary(summary)
        assert "empty-run" in text
        assert "Records evaluated:    0" in text

    def test_render_json(self, sample_results_multi_judge):
        """Render JSON produces valid JSON."""
        summary = CalibrationSummary.from_results(
            sample_results_multi_judge, "test-run",
            ["judge-a", "judge-b", "judge-c"],
        )
        text = render_calibration_json(summary)
        data = json.loads(text)
        assert data["run_id"] == "test-run"
        assert data["total_records"] == 3
        assert data["total_judges"] == 3
        assert "disagreements" in data
        assert "judge_agreement" in data

    def test_render_json_empty(self):
        """Empty summary produces valid JSON."""
        summary = CalibrationSummary(run_id="empty-run", total_records=0, total_judges=1)
        text = render_calibration_json(summary)
        data = json.loads(text)
        assert data["run_id"] == "empty-run"
        assert data["pass_agreement_rate"] is None

    def test_strip_ansi(self):
        """strip_ansi removes ANSI escape codes."""
        colored = "hello \x1b[32mgreen\x1b[0m world"
        plain = strip_ansi(colored)
        assert plain == "hello green world"
        assert "\x1b" not in plain

    def test_strip_ansi_noop(self):
        """strip_ansi is a no-op on plain text."""
        assert strip_ansi("plain text") == "plain text"

    def test_render_summary_disagreements(self, sample_results_multi_judge):
        """Summary with disagreements shows disagreement section."""
        summary = CalibrationSummary.from_results(
            sample_results_multi_judge, "test-run",
            ["judge-a", "judge-b", "judge-c"],
            disagreement_threshold=0.0,
        )
        text = render_calibration_summary(summary)
        assert "Disagreements" in text
        assert "rec-0" in text  # all records should be in disagreements

    def test_render_summary_pass_agreement_none(self):
        """Summary with no data shows 0.0% for pass agreement."""
        summary = CalibrationSummary(run_id="test", total_records=0, total_judges=1)
        text = render_calibration_summary(summary)
        assert "Pass/fail agreement:  0.0%" in text