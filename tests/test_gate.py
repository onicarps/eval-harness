"""Tests for the gate (CI/CD quality gate) module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import typer

from src.cli import app
from src.db import Database
from src.gate import CheckGateResult, GateRunner
from src.models import EvalRecord, EvalResult, EvalRun, PassFail, RunStatus

# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "gate_test.db")


@pytest.fixture
def completed_run(db: Database) -> str:
    """Create a completed run with 10 records, 8 pass (80%)."""
    run_id = "gate-run-001"
    run = EvalRun(
        run_id=run_id,
        status=RunStatus.COMPLETED,
        record_count=10,
        mean_score=0.75,
        pass_rate=0.8,
        eval_time_seconds=12.5,
        config={"file": "test.jsonl", "format": "jsonl", "judges": ["judge-a"], "pass_threshold": 0.7},
        judge_model="judge-a",
    )
    db.insert_run(run)
    # Insert records first (FK constraint: eval_results → eval_records)
    for i in range(10):
        record = EvalRecord(
            record_id=f"rec-{i}",
            run_id=run_id,
            input_text=f"Question {i}",
            output_text=f"Answer {i}",
        )
        db.insert_record(record)
    # Then insert results
    for i in range(10):
        result = EvalResult(
            record_id=f"rec-{i}",
            run_id=run_id,
            rubric_id="faithfulness-v1",
            rubric_version="1.0",
            faithfulness=0.8 if i < 8 else 0.5,
            task_completion=0.8 if i < 8 else 0.4,
            combined_score=0.8 if i < 8 else 0.45,
            pass_fail=PassFail.PASS if i < 8 else PassFail.FAIL,
            judge_model="judge-a",
            judge_tried=["judge-a"],
            reasoning="test",
            faithfulness_reasoning="test",
            task_completion_reasoning="test",
        )
        db.insert_result(result)
    return run_id


@pytest.fixture
def failed_run(db: Database) -> str:
    """Create a failed run with 10 records, 3 pass (30%)."""
    run_id = "gate-run-002"
    run = EvalRun(
        run_id=run_id,
        status=RunStatus.COMPLETED,
        record_count=10,
        mean_score=0.55,
        pass_rate=0.3,
        eval_time_seconds=10.0,
        config={"file": "test.jsonl", "format": "jsonl", "judges": ["judge-a"], "pass_threshold": 0.7},
        judge_model="judge-a",
    )
    db.insert_run(run)
    for i in range(10):
        record = EvalRecord(
            record_id=f"rec-{i}",
            run_id=run_id,
            input_text=f"Question {i}",
            output_text=f"Answer {i}",
        )
        db.insert_record(record)
    for i in range(10):
        result = EvalResult(
            record_id=f"rec-{i}",
            run_id=run_id,
            rubric_id="faithfulness-v1",
            rubric_version="1.0",
            faithfulness=0.7 if i < 3 else 0.4,
            task_completion=0.7 if i < 3 else 0.3,
            combined_score=0.7 if i < 3 else 0.35,
            pass_fail=PassFail.PASS if i < 3 else PassFail.FAIL,
            judge_model="judge-a",
            judge_tried=["judge-a"],
            reasoning="test",
            faithfulness_reasoning="test",
            task_completion_reasoning="test",
        )
        db.insert_result(result)
    return run_id


@pytest.fixture
def multiple_runs(db: Database) -> list[str]:
    """Create 3 runs with varying pass rates."""
    run_ids = []
    for run_id, pass_rate in [("run-a", 0.9), ("run-b", 0.7), ("run-c", 0.4)]:
        run = EvalRun(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            record_count=10,
            mean_score=pass_rate,
            pass_rate=pass_rate,
            config={"file": "test.jsonl", "format": "jsonl"},
            judge_model="judge-a",
        )
        db.insert_run(run)
        for i in range(3):
            record = EvalRecord(
                record_id=f"{run_id}-rec-{i}",
                run_id=run_id,
                input_text="Q",
                output_text="A",
            )
            db.insert_record(record)
        run_ids.append(run_id)
    return run_ids


# ── CheckGateResult ──────────────────────────────────────────────────────────

class TestCheckGateResult:
    def test_pass_above_threshold(self):
        """Run with pass rate above threshold passes."""
        result = CheckGateResult(run_id="test-1", pass_rate=0.85, threshold=0.7, passed=True,
                                 mean_score=0.80, record_count=20)
        assert result.passed is True
        assert result.exit_code == 0

    def test_fail_below_threshold(self):
        """Run with pass rate below threshold fails."""
        result = CheckGateResult(run_id="test-2", pass_rate=0.5, threshold=0.7, passed=False,
                                 mean_score=0.55, record_count=20)
        assert result.passed is False
        assert result.exit_code == 1

    def test_json_output(self):
        """JSON output contains expected fields."""
        result = CheckGateResult(run_id="test-3", pass_rate=0.75, threshold=0.8, passed=False,
                                 mean_score=0.70, record_count=15)
        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["run_id"] == "test-3"
        assert data["pass_rate"] == 0.75
        assert data["threshold"] == 0.8
        assert data["passed"] is False
        assert data["exit_code"] == 1

    def test_human_readable(self):
        """Human-readable output contains key info."""
        result = CheckGateResult(run_id="test-4", pass_rate=0.8, threshold=0.7, passed=True,
                                 mean_score=0.78, record_count=25)
        text = result.to_text()
        assert "test-4" in text
        assert "80.0%" in text
        assert "70.0%" in text


class TestCheckGateResultEdgeCases:
    def test_zero_pass_rate(self):
        """Zero pass rate handled correctly."""
        result = CheckGateResult(run_id="test-zero", pass_rate=0.0, threshold=0.7, passed=False,
                                 mean_score=0.0, record_count=10)
        assert result.passed is False
        assert result.exit_code == 1

    def test_perfect_pass_rate(self):
        """Perfect pass rate handled correctly."""
        result = CheckGateResult(run_id="test-perfect", pass_rate=1.0, threshold=0.7, passed=True,
                                 mean_score=1.0, record_count=10)
        assert result.passed is True
        assert result.exit_code == 0


# ── GateRunner ───────────────────────────────────────────────────────────────

class TestGateRunner:
    def test_check_run_passes(self, db: Database, completed_run: str):
        """Runner correctly identifies a run that passes the threshold."""
        runner = GateRunner(db)
        result = runner.check(completed_run, threshold=0.7)
        assert result.passed is True
        assert result.exit_code == 0
        assert result.pass_rate == 0.8

    def test_check_run_fails(self, db: Database, failed_run: str):
        """Runner correctly identifies a run that fails the threshold."""
        runner = GateRunner(db)
        result = runner.check(failed_run, threshold=0.7)
        assert result.passed is False
        assert result.exit_code == 1

    def test_check_run_not_found(self, db: Database):
        """Runner raises error for nonexistent run."""
        runner = GateRunner(db)
        with pytest.raises(ValueError, match="not found"):
            runner.check("nonexistent-run", threshold=0.7)

    def test_check_custom_threshold(self, db: Database, completed_run: str):
        """Runner respects custom threshold."""
        runner = GateRunner(db)
        # 80% pass rate, but 90% threshold → fail
        result = runner.check(completed_run, threshold=0.9)
        assert result.passed is False
        assert result.exit_code == 1

    def test_suggest_baseline(self, db: Database, multiple_runs: list[str]):
        """Suggest baseline from run history."""
        runner = GateRunner(db)
        suggestion = runner.suggest_baseline()
        # Should return a reasonable baseline between 0.4 and 0.9
        assert suggestion is not None
        recommended = suggestion["recommended_baseline"]
        assert 0.0 <= recommended <= 1.0
        assert "recommended_baseline" in suggestion
        assert "runs_analyzed" in suggestion
        assert suggestion["runs_analyzed"] == 3

    def test_suggest_baseline_empty(self, db: Database):
        """Suggest baseline with no runs returns None."""
        runner = GateRunner(db)
        suggestion = runner.suggest_baseline()
        assert suggestion is None

    def test_suggest_baseline_only_failed_runs(self, db: Database):
        """Suggest baseline only considers completed runs, ignores failed ones."""
        # Insert a completed run
        run = EvalRun(
            run_id="good-run",
            status=RunStatus.COMPLETED,
            record_count=10,
            mean_score=0.7,
            pass_rate=0.7,
            config={"file": "test.jsonl"},
            judge_model="judge-a",
        )
        db.insert_run(run)
        # Insert a failed run
        run2 = EvalRun(
            run_id="failed-run",
            status=RunStatus.FAILED,
            record_count=5,
            config={"file": "test.jsonl"},
        )
        db.insert_run(run2)
        # Should only suggest from completed runs
        runner = GateRunner(db)
        suggestion = runner.suggest_baseline()
        assert suggestion is not None
        assert suggestion["runs_analyzed"] == 1

    def test_check_missing_mean_score(self, db: Database):
        """Handle runs without mean_score gracefully."""
        run_id = "no-mean-run"
        run = EvalRun(
            run_id=run_id,
            status=RunStatus.COMPLETED,
            record_count=5,
            mean_score=None,  # missing mean score
            pass_rate=0.8,
            config={"file": "test.jsonl"},
            judge_model="judge-a",
        )
        db.insert_run(run)
        runner = GateRunner(db)
        result = runner.check(run_id, threshold=0.7)
        assert result.passed is True
        assert result.mean_score is None  # preserved as None


# ── CLI Command ──────────────────────────────────────────────────────────────

class TestGateCLI:
    def test_gate_run_passes(self, db: Database, completed_run: str):
        """CLI exits 0 when run passes threshold."""
        exit_code = None
        def mock_exit(self, code=0, *args, **kwargs):
            nonlocal exit_code
            exit_code = code
            self.exit_code = code
        with patch.object(typer.Exit, "__init__", mock_exit):
            try:
                app(["gate", "--run-id", completed_run, "--db", str(db.path)])
            except SystemExit:
                pass
        assert exit_code == 0

    def test_gate_run_fails(self, db: Database, failed_run: str):
        """CLI exits 1 when run fails threshold."""
        exit_code = None
        def mock_exit(self, code=0, *args, **kwargs):
            nonlocal exit_code
            exit_code = code
            self.exit_code = code
        with patch.object(typer.Exit, "__init__", mock_exit):
            try:
                app(["gate", "--run-id", failed_run, "--db", str(db.path)])
            except SystemExit:
                pass
        assert exit_code == 1

    def test_gate_json_output(self, db: Database, completed_run: str, tmp_path: Path):
        """CLI outputs valid JSON with --json flag."""
        output_file = tmp_path / "gate_result.json"
        exit_code = None
        def mock_exit(self, code=0, *args, **kwargs):
            nonlocal exit_code
            exit_code = code
            self.exit_code = code
        with patch.object(typer.Exit, "__init__", mock_exit):
            try:
                app(["gate", "--run-id", completed_run, "--db", str(db.path),
                     "--json", "--output-file", str(output_file)])
            except SystemExit:
                pass
        assert exit_code == 0
        data = json.loads(output_file.read_text())
        assert data["run_id"] == completed_run
        assert data["pass_rate"] == 0.8
        assert data["passed"] is True

    def test_gate_suggest_baseline(self, db: Database, multiple_runs: list[str]):
        """CLI suggest-baseline returns a suggestion from history."""
        exit_code = None
        def mock_exit(self, code=0, *args, **kwargs):
            nonlocal exit_code
            exit_code = code
            self.exit_code = code
        with patch.object(typer.Exit, "__init__", mock_exit):
            try:
                app(["gate", "--suggest-baseline", "--db", str(db.path)])
            except SystemExit:
                pass
        assert exit_code == 0

    def test_gate_missing_run(self, db: Database):
        """CLI exits 2 when run is not found."""
        exit_code = None
        def mock_exit(self, code=0, *args, **kwargs):
            nonlocal exit_code
            exit_code = code
            self.exit_code = code
        with patch.object(typer.Exit, "__init__", mock_exit):
            try:
                app(["gate", "--run-id", "nonexistent", "--db", str(db.path)])
            except SystemExit:
                pass
        assert exit_code == 2
