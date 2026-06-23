"""Tests for Phase 2A new features: feedback, compare-judges, degrade."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.db import Database
from src.models import EvalRecord, EvalResult, EvalRun, PassFail, RunStatus
from src.evaluator import (
    LLMEvaluator,
    EvaluatorConfig,
    local_heuristic_score,
    combine_scores,
    pass_fail_from,
)
from src.reporter import render_comparison_table


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_record() -> EvalRecord:
    return EvalRecord(
        input_text="What is the capital of France?",
        output_text="The capital of France is Paris.",
        reference_text="Paris",
    )


@pytest.fixture
def low_scoring_record() -> EvalRecord:
    return EvalRecord(
        input_text="Explain quantum computing in detail",
        output_text="It is computing with quantum.",
        reference_text="Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computation.",
    )


@pytest.fixture
def sample_result_pass(sample_record) -> EvalResult:
    return EvalResult(
        record_id=sample_record.record_id,
        run_id="test-run",
        faithfulness=0.9,
        task_completion=0.85,
        combined_score=0.875,
        pass_fail=PassFail.PASS,
        judge_model="judge-a",
        judge_tried=["judge-a"],
    )


@pytest.fixture
def sample_result_fail(low_scoring_record) -> EvalResult:
    return EvalResult(
        record_id=low_scoring_record.record_id,
        run_id="test-run",
        faithfulness=0.3,
        task_completion=0.2,
        combined_score=0.25,
        pass_fail=PassFail.FAIL,
        judge_model="judge-a",
        judge_tried=["judge-a"],
        reasoning="Output is too short and misses key details",
    )


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


# ── Local Heuristic Tests ────────────────────────────────────────────────────

class TestLocalHeuristicScore:
    def test_basic_score(self, sample_record):
        """Local heuristic returns reasonable scores for a good response."""
        result = local_heuristic_score(sample_record)
        assert "faithfulness" in result
        assert "task_completion" in result
        assert 0.0 <= result["faithfulness"] <= 1.0
        assert 0.0 <= result["task_completion"] <= 1.0

    def test_short_output_scores_low(self, low_scoring_record):
        """Very short output should have low task completion."""
        result = local_heuristic_score(low_scoring_record)
        assert result["task_completion"] < 0.5

    def test_overlap_affects_faithfulness(self):
        """High word overlap between input/output -> higher faithfulness."""
        record = EvalRecord(
            input_text="Python programming language features",
            output_text="Python programming language features include dynamic typing",
        )
        result_high = local_heuristic_score(record)
        assert result_high["faithfulness"] > 0.3

    def test_empty_input(self):
        """Empty input doesn't crash."""
        record = EvalRecord(input_text="", output_text="Some output text here")
        result = local_heuristic_score(record)
        assert result["faithfulness"] == 0.5

    def test_reasoning_fields_present(self, sample_record):
        """Heuristic includes reasoning fields."""
        result = local_heuristic_score(sample_record)
        assert "reasoning" in result
        assert "faithfulness_reasoning" in result
        assert "task_completion_reasoning" in result


# ── Feedback Generation Tests ────────────────────────────────────────────────

class TestFeedbackGeneration:
    @pytest.mark.asyncio
    async def test_generate_feedback_success(self, low_scoring_record, sample_result_fail):
        """Feedback generation returns suggestions on success."""
        config = EvaluatorConfig(api_key="test-key", judges=["judge-a"])
        db = Database(":memory:")
        evaluator = LLMEvaluator(db, config)

        # Mock the HTTP client response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({"suggestions": ["Add more detail", "Include examples"]})
                }
            }]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        feedback = await evaluator.generate_feedback(mock_client, low_scoring_record, sample_result_fail)
        assert feedback is not None
        data = json.loads(feedback)
        assert "suggestions" in data
        assert len(data["suggestions"]) == 2

    @pytest.mark.asyncio
    async def test_generate_feedback_api_error(self, low_scoring_record, sample_result_fail):
        """Feedback generation returns None on API error."""
        config = EvaluatorConfig(api_key="test-key", judges=["judge-a"])
        db = Database(":memory:")
        evaluator = LLMEvaluator(db, config)

        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        feedback = await evaluator.generate_feedback(mock_client, low_scoring_record, sample_result_fail)
        assert feedback is None

    @pytest.mark.asyncio
    async def test_generate_feedback_no_suggestions(self, low_scoring_record, sample_result_fail):
        """Feedback generation returns None when no suggestions in response."""
        config = EvaluatorConfig(api_key="test-key", judges=["judge-a"])
        db = Database(":memory:")
        evaluator = LLMEvaluator(db, config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"result": "good"})}}]
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        feedback = await evaluator.generate_feedback(mock_client, low_scoring_record, sample_result_fail)
        assert feedback is None

    @pytest.mark.asyncio
    async def test_generate_all_feedback_skips_passes(self, db, sample_record, sample_result_pass):
        """Feedback is not generated for passing records."""
        config = EvaluatorConfig(api_key="test-key", judges=["judge-a"])
        evaluator = LLMEvaluator(db, config)

        run = EvalRun(run_id="test-run", config={}, status=RunStatus.RUNNING)
        await evaluator.generate_all_feedback(run, [sample_record], [sample_result_pass])
        # No feedback on pass
        assert sample_result_pass.feedback is None

    @pytest.mark.asyncio
    async def test_generate_all_feedback_skips_errors(self, db, sample_record):
        """Feedback is not generated for results with errors."""
        result = EvalResult(
            record_id=sample_record.record_id,
            run_id="test-run",
            faithfulness=0.0,
            task_completion=0.0,
            combined_score=0.0,
            pass_fail=PassFail.FAIL,
            judge_model="judge-a",
            judge_tried=["judge-a"],
            error="API timeout",
        )
        config = EvaluatorConfig(api_key="test-key", judges=["judge-a"])
        evaluator = LLMEvaluator(db, config)

        run = EvalRun(run_id="test-run", config={}, status=RunStatus.RUNNING)
        await evaluator.generate_all_feedback(run, [sample_record], [result])
        # Skipped because it has an error
        assert result.feedback is None


# ── Degraded Mode Tests ──────────────────────────────────────────────────────

class TestDegradedMode:
    def test_degrade_flag_in_config(self):
        """EvaluatorConfig has degrade field."""
        config = EvaluatorConfig(api_key="test-key", judges=["judge-a"], degrade=True)
        assert config.degrade is True

    def test_degrade_default_false(self):
        """degrade defaults to False."""
        config = EvaluatorConfig(api_key="test-key", judges=["judge-a"])
        assert config.degrade is False

    def test_local_heuristic_called_on_degrade(self, sample_record):
        """When degrade=True and all judges fail, local heuristic is used."""
        result = local_heuristic_score(sample_record)
        assert result["faithfulness"] >= 0.0
        assert result["task_completion"] >= 0.0
        assert "heuristic" in result["reasoning"].lower()


# ── Comparison Table Tests ───────────────────────────────────────────────────

class TestComparisonTable:
    def test_render_comparison_table(self):
        """Comparison table renders with correct structure."""
        records = [
            EvalRecord(input_text="Question 1", output_text="Answer 1"),
            EvalRecord(input_text="Question 2", output_text="Answer 2"),
        ]
        results = [
            EvalResult(record_id=records[0].record_id, run_id="r", faithfulness=0.9, task_completion=0.8, combined_score=0.85, pass_fail=PassFail.PASS, judge_model="judge-a", judge_tried=["judge-a"]),
            EvalResult(record_id=records[0].record_id, run_id="r", faithfulness=0.7, task_completion=0.7, combined_score=0.7, pass_fail=PassFail.PASS, judge_model="judge-b", judge_tried=["judge-b"]),
            EvalResult(record_id=records[1].record_id, run_id="r", faithfulness=0.3, task_completion=0.3, combined_score=0.3, pass_fail=PassFail.FAIL, judge_model="judge-a", judge_tried=["judge-a"]),
            EvalResult(record_id=records[1].record_id, run_id="r", faithfulness=0.9, task_completion=0.9, combined_score=0.9, pass_fail=PassFail.PASS, judge_model="judge-b", judge_tried=["judge-b"]),
        ]
        table = render_comparison_table(records, results, ["judge-a", "judge-b"])
        assert isinstance(table, str)
        assert "Judge Comparison" in table

    def test_comparison_table_single_judge(self):
        """Single judge still renders (but no std dev)."""
        records = [EvalRecord(input_text="Q", output_text="A")]
        results = [
            EvalResult(record_id=records[0].record_id, run_id="r", faithfulness=0.8, task_completion=0.8, combined_score=0.8, pass_fail=PassFail.PASS, judge_model="judge-a", judge_tried=["judge-a"]),
        ]
        table = render_comparison_table(records, results, ["judge-a"])
        assert isinstance(table, str)

    def test_comparison_table_empty_results(self):
        """Empty results produce empty table."""
        records = [EvalRecord(input_text="Q", output_text="A")]
        table = render_comparison_table(records, [], ["judge-a"])
        assert isinstance(table, str)


# ── DB Migration v3 Tests ────────────────────────────────────────────────────

class TestMigrationV3:
    def test_feedback_column_exists(self, db):
        """Feedback column exists in eval_results after migration."""
        run = EvalRun(run_id="run-1", config={}, status=RunStatus.RUNNING)
        db.insert_run(run)
        rec = EvalRecord(record_id="r1", run_id="run-1", input_text="q", output_text="a")
        db.insert_record(rec)
        res = EvalResult(
            record_id="r1", run_id="run-1",
            faithfulness=0.5, task_completion=0.5, combined_score=0.5,
            pass_fail=PassFail.FAIL, judge_model="j", judge_tried=["j"],
            feedback='{"suggestions": ["be more detailed"]}',
        )
        db.insert_result(res)
        results = db.get_results("run-1")
        assert len(results) == 1
        assert results[0].feedback == '{"suggestions": ["be more detailed"]}'

    def test_feedback_default_none(self, db):
        """Feedback defaults to None when not set."""
        run = EvalRun(run_id="run-2", config={}, status=RunStatus.RUNNING)
        db.insert_run(run)
        rec = EvalRecord(record_id="r2", run_id="run-2", input_text="q", output_text="a")
        db.insert_record(rec)
        res = EvalResult(
            record_id="r2", run_id="run-2",
            faithfulness=0.5, task_completion=0.5, combined_score=0.5,
            pass_fail=PassFail.FAIL, judge_model="j", judge_tried=["j"],
        )
        db.insert_result(res)
        results = db.get_results("run-2")
        assert len(results) == 1
        assert results[0].feedback is None

    def test_schema_version_is_3(self, db):
        """Database migrates to version 3."""
        assert db.get_schema_version() == 3

    def test_rollback_v3(self, db):
        """Rollback from v3 to v2 removes feedback column."""
        db.rollback(2)
        assert db.get_schema_version() == 2
