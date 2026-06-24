"""Tests for coverage gaps across all modules.

Targets untested functions, edge cases, error paths, and integration
scenarios identified during test audit.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pytest_httpx import HTTPXMock
from typer.testing import CliRunner

from src.calibrate import (
    CalibrationSummary,
    _compute_pair_agreement,
    asyncio_run,
)
from src.cli import _get_api_key, _judge_list, app
from src.db import CURRENT_SCHEMA_VERSION, Database
from src.evaluator import (
    EvaluatorConfig,
    LLMEvaluator,
    _is_permanent_http_error,
    _render_prompt,
)
from src.gate import CheckGateResult, GateRunner
from src.ingest import (
    IngestOptions,
    _apply_post_filters,
    _make_record,
    _parse_since,
    _row_after_since,
    ingest_csv,
    ingest_jsonl,
    ingest_stdin,
)
from src.judges import (
    JudgeRegistry,
    _is_free,
)
from src.models import (
    BUILTIN_RUBRIC_V1,
    EvalRecord,
    EvalResult,
    EvalRun,
    JudgeCacheEntry,
    PassFail,
    RubricTemplate,
    RunStatus,
)
from src.reporter import (
    ascii_histogram,
    build_summary,
    export_results,
    judge_usage,
    render_table,
)
from src.trend import (
    compute_trends,
)

runner = CliRunner()


# ── conftest shared fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "test.db")


@pytest.fixture()
def openrouter_payload() -> dict:
    return {
        "id": "chatcmpl-test",
        "model": "test/judge",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({
                        "faithfulness": 0.9,
                        "task_completion": 0.85,
                        "reasoning": "good answer",
                        "faithfulness_reasoning": "matches reference",
                        "task_completion_reasoning": "completes the task",
                    }),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


# ── _is_permanent_http_error ────────────────────────────────────────────────


class TestIsPermanentHttpError:
    """The _is_permanent_http_error function is untested directly."""

    def test_400_is_permanent(self):
        assert _is_permanent_http_error(400) is True

    def test_401_is_permanent(self):
        assert _is_permanent_http_error(401) is True

    def test_402_is_permanent(self):
        assert _is_permanent_http_error(402) is True

    def test_403_is_permanent(self):
        assert _is_permanent_http_error(403) is True

    def test_404_is_permanent(self):
        assert _is_permanent_http_error(404) is True

    def test_429_is_transient(self):
        assert _is_permanent_http_error(429) is False

    def test_500_is_transient(self):
        assert _is_permanent_http_error(500) is False

    def test_502_is_transient(self):
        assert _is_permanent_http_error(502) is False

    def test_503_is_transient(self):
        assert _is_permanent_http_error(503) is False

    def test_200_is_transient(self):
        assert _is_permanent_http_error(200) is False


# ── _render_prompt ───────────────────────────────────────────────────────────


class TestRenderPrompt:
    """_render_prompt is untested."""

    def test_replaces_all_placeholders(self):
        rubric = RubricTemplate(
            rubric_id="test",
            version="1.0",
            prompt_template="Input: {input}\nOutput: {output}\nRef: {reference}",
        )
        record = EvalRecord(
            input_text="hello",
            output_text="world",
            reference_text="truth",
        )
        result = _render_prompt(rubric, record)
        assert "Input: hello" in result
        assert "Output: world" in result
        assert "Ref: truth" in result

    def test_none_reference_replaced_with_none_string(self):
        rubric = RubricTemplate(
            rubric_id="test",
            version="1.0",
            prompt_template="Input: {input}\nOutput: {output}\nRef: {reference}",
        )
        record = EvalRecord(input_text="hi", output_text="there")
        result = _render_prompt(rubric, record)
        assert "Ref: (none)" in result


# ── _is_free (judges) ────────────────────────────────────────────────────────


class TestIsFree:
    """_is_free edge cases."""

    def test_both_zero(self):
        assert _is_free({"pricing": {"prompt": "0", "completion": "0"}}) is True

    def test_prompt_paid(self):
        assert _is_free({"pricing": {"prompt": "0.001", "completion": "0"}}) is False

    def test_completion_paid(self):
        assert _is_free({"pricing": {"prompt": "0", "completion": "0.001"}}) is False

    def test_missing_pricing(self):
        # Missing pricing key means both prompt and completion default to 0 → free
        assert _is_free({}) is True

    def test_pricing_is_none(self):
        # None pricing defaults to {} → both prices are 0 → free
        assert _is_free({"pricing": None}) is True

    def test_invalid_pricing_type(self):
        assert _is_free({"pricing": {"prompt": "free", "completion": "free"}}) is False

    def test_integer_pricing_zero(self):
        assert _is_free({"pricing": {"prompt": 0, "completion": 0}}) is True


# ── ingest: _parse_since edge cases ─────────────────────────────────────────


class TestParseSince:
    """_parse_since edge cases."""

    def test_none_returns_none(self):
        assert _parse_since(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_since("") is None

    def test_timezone_naive_gets_utc(self):
        result = _parse_since("2024-01-15")
        assert result is not None
        assert result.tzinfo is not None

    def test_timezone_aware_preserved(self):
        result = _parse_since("2024-01-15T10:30:00+05:00")
        assert result is not None
        assert result.tzinfo is not None


# ── ingest: _row_after_since edge cases ──────────────────────────────────────


class TestRowAfterSince:
    """_row_after_since edge cases."""

    def test_none_since_always_true(self):
        assert _row_after_since("2024-01-01", None) is True

    def test_none_row_ts_when_since_set(self):
        # If since is set but row_ts is None, return since is None → False
        assert _row_after_since(None, _parse_since("2024-01-01")) is False

    def test_invalid_row_ts_returns_false(self):
        assert _row_after_since("not-a-date", _parse_since("2024-01-01")) is False

    def test_row_before_since_returns_false(self):
        since = _parse_since("2024-06-01")
        assert _row_after_since("2024-01-01", since) is False

    def test_row_after_since_returns_true(self):
        since = _parse_since("2024-01-01")
        assert _row_after_since("2024-06-01", since) is True

    def test_row_equals_since_returns_true(self):
        since = _parse_since("2024-01-01T00:00:00+00:00")
        assert _row_after_since("2024-01-01T00:00:00+00:00", since) is True


# ── ingest: _make_record edge cases ──────────────────────────────────────────


class TestMakeRecord:
    """_make_record edge cases."""

    def test_missing_input_returns_none(self):
        opts = IngestOptions()
        assert _make_record({"output": "o"}, opts, None) is None

    def test_missing_output_returns_none(self):
        opts = IngestOptions()
        assert _make_record({"input": "i"}, opts, None) is None

    def test_empty_input_returns_none(self):
        opts = IngestOptions()
        assert _make_record({"input": "", "output": "o"}, opts, None) is None

    def test_empty_output_returns_none(self):
        opts = IngestOptions()
        assert _make_record({"input": "i", "output": ""}, opts, None) is None

    def test_non_string_input_returns_none(self):
        opts = IngestOptions()
        assert _make_record({"input": 123, "output": "o"}, opts, None) is None

    def test_non_string_output_returns_none(self):
        opts = IngestOptions()
        assert _make_record({"input": "i", "output": 456}, opts, None) is None

    def test_empty_reference_becomes_none(self):
        opts = IngestOptions()
        rec = _make_record({"input": "i", "output": "o", "reference": ""}, opts, None)
        assert rec is not None
        assert rec.reference_text is None

    def test_non_string_reference_becomes_none(self):
        opts = IngestOptions()
        rec = _make_record({"input": "i", "output": "o", "reference": 123}, opts, None)
        assert rec is not None
        assert rec.reference_text is None

    def test_metadata_excludes_mapped_columns(self):
        opts = IngestOptions(input_col="prompt", output_col="response")
        rec = _make_record(
            {"prompt": "q", "response": "a", "extra": "meta"}, opts, "file.jsonl"
        )
        assert rec is not None
        assert rec.metadata == {"extra": "meta"}
        assert rec.source_file == "file.jsonl"


# ── ingest: CSV missing columns ──────────────────────────────────────────────


class TestIngestCsvMissingColumns:
    """CSV ingestion with missing required columns is untested."""

    def test_missing_input_col(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        p.write_text("output,reference\nhello,world\n")
        opts = IngestOptions(input_col="input", output_col="output")
        records = list(ingest_csv(p, opts))
        assert records == []

    def test_missing_output_col(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.csv"
        p.write_text("input,reference\nhello,world\n")
        opts = IngestOptions(input_col="input", output_col="output")
        records = list(ingest_csv(p, opts))
        assert records == []

    def test_no_header_row(self, tmp_path: Path) -> None:
        p = tmp_path / "noheader.csv"
        p.write_text("just some data\n")
        opts = IngestOptions()
        records = list(ingest_csv(p, opts))
        assert records == []


# ── ingest: stdin unsupported format ─────────────────────────────────────────


class TestIngestStdinUnsupportedFormat:
    """ingest_stdin with unsupported format is untested."""

    def test_unsupported_format_raises(self):
        import io
        stream = io.StringIO("data")
        with pytest.raises(ValueError, match="unsupported stdin format"):
            list(ingest_stdin(stream, fmt="xml"))


# ── ingest: _apply_post_filters edge cases ────────────────────────────────────


class TestApplyPostFilters:
    """_apply_post_filters edge cases."""

    def test_sample_zero_returns_empty(self):
        records = [EvalRecord(input_text="a", output_text="b")]
        opts = IngestOptions(sample=0)
        result = list(_apply_post_filters(iter(records), opts))
        assert result == []

    def test_sample_on_iterator(self):
        records = [EvalRecord(input_text=f"i{x}", output_text=f"o{x}") for x in range(10)]
        opts = IngestOptions(sample=3, seed=42)
        result = list(_apply_post_filters(iter(records), opts))
        assert len(result) == 3

    def test_limit_zero_returns_empty(self):
        records = [EvalRecord(input_text="a", output_text="b")]
        opts = IngestOptions(limit=0)
        result = list(_apply_post_filters(iter(records), opts))
        # limit=0 means count >= 0 is True immediately, so nothing is yielded
        assert result == []


# ── ingest: JSONL with non-object JSON ───────────────────────────────────────


class TestIngestJsonlNonObject:
    """JSONL with non-object JSON values is untested."""

    def test_array_json_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "array.jsonl"
        p.write_text('[1, 2, 3]\n{"input": "i", "output": "o"}\n')
        records = list(ingest_jsonl(p))
        assert len(records) == 1
        assert records[0].input_text == "i"

    def test_string_json_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "str.jsonl"
        p.write_text('"just a string"\n{"input": "i", "output": "o"}\n')
        records = list(ingest_jsonl(p))
        assert len(records) == 1

    def test_number_json_skipped(self, tmp_path: Path) -> None:
        p = tmp_path / "num.jsonl"
        p.write_text('42\n{"input": "i", "output": "o"}\n')
        records = list(ingest_jsonl(p))
        assert len(records) == 1


# ── ingest: CSV with custom timestamp column ─────────────────────────────────


class TestIngestCsvCustomTimestampCol:
    """CSV with custom timestamp column name is untested."""

    def test_custom_timestamp_col(self, tmp_path: Path) -> None:
        p = tmp_path / "custom_ts.csv"
        p.write_text("prompt,response,ground_truth,date\nq1,a1,r1,2024-01-01\nq2,a2,,2024-06-01\n")
        opts = IngestOptions(
            input_col="prompt",
            output_col="response",
            reference_col="ground_truth",
            timestamp_col="date",
            since="2024-05-01",
        )
        records = list(ingest_csv(p, opts))
        assert len(records) == 1
        assert records[0].input_text == "q2"


# ── db: list_runs with limit ─────────────────────────────────────────────────


class TestDbListRunsLimit:
    """list_runs with limit parameter is untested."""

    def test_list_runs_custom_limit(self, db: Database) -> None:
        for _i in range(5):
            db.insert_run(EvalRun(config={}))
        runs = db.list_runs(limit=3)
        assert len(runs) == 3

    def test_list_runs_limit_1(self, db: Database) -> None:
        for _i in range(3):
            db.insert_run(EvalRun(config={}))
        runs = db.list_runs(limit=1)
        assert len(runs) == 1

    def test_list_runs_ordered_newest_first(self, db: Database) -> None:
        for i in range(3):
            run = EvalRun(config={"index": i})
            db.insert_run(run)
        runs = db.list_runs(limit=10)
        # Newest first means the last inserted (highest index) should be first
        assert runs[0].config["index"] == 2


# ── db: rollback edge cases ──────────────────────────────────────────────────


class TestDbRollback:
    """rollback edge cases are untested."""

    def test_rollback_invalid_target_negative(self, db: Database) -> None:
        with pytest.raises(RuntimeError, match="target version must be >= 0"):
            db.rollback(-1)

    def test_rollback_to_current_version_raises(self, db: Database) -> None:
        with pytest.raises(RuntimeError, match="must be less than current version"):
            db.rollback(CURRENT_SCHEMA_VERSION)

    def test_rollback_to_future_version_raises(self, db: Database) -> None:
        with pytest.raises(RuntimeError, match="must be less than current version"):
            db.rollback(CURRENT_SCHEMA_VERSION + 1)


# ── db: insert_result FK constraint ──────────────────────────────────────────


class TestDbInsertResultForeignKey:
    """insert_result FK constraint is untested."""

    def test_insert_result_without_record_raises(self, db: Database) -> None:
        run = EvalRun(config={})
        db.insert_run(run)
        result = EvalResult(
            record_id="nonexistent",
            run_id=run.run_id,
            faithfulness=0.5,
            task_completion=0.5,
            combined_score=0.5,
            pass_fail=PassFail.FAIL,
            judge_model="m",
        )
        with pytest.raises(Exception):
            db.insert_result(result)


# ── db: insert_record without run_id ─────────────────────────────────────────


class TestDbInsertRecordValidation:
    """insert_record without run_id is untested."""

    def test_insert_record_without_run_id_raises(self, db: Database) -> None:
        rec = EvalRecord(input_text="i", output_text="o", run_id=None)
        with pytest.raises(ValueError, match="run_id is required"):
            db.insert_record(rec)


# ── db: update_run preserves config_json ─────────────────────────────────────


class TestDbUpdateRunPreservesConfig:
    """update_run preserves config changes."""

    def test_update_run_config_roundtrip(self, db: Database) -> None:
        run = EvalRun(config={"original": True})
        db.insert_run(run)
        run.config = {"updated": True}
        db.update_run(run)
        fetched = db.get_run(run.run_id)
        assert fetched is not None
        assert fetched.config == {"updated": True}


# ── db: get_results for nonexistent run ──────────────────────────────────────


class TestDbGetResultsNonexistentRun:
    """get_results for a run with no results."""

    def test_get_results_empty_list(self, db: Database) -> None:
        run = EvalRun(config={})
        db.insert_run(run)
        results = db.get_results(run.run_id)
        assert results == []


# ── db: get_cache for nonexistent key ────────────────────────────────────────


class TestDbGetCacheNonexistent:
    """get_cache for a key that doesn't exist."""

    def test_get_cache_missing_key(self, db: Database) -> None:
        assert db.get_cache("nonexistent_key") is None


# ── db: put_cache upsert updates response ────────────────────────────────────


class TestDbCacheUpsert:
    """put_cache upsert behavior is untested."""

    def test_put_cache_overwrites_on_conflict(self, db: Database) -> None:
        entry = JudgeCacheEntry(
            cache_key="k1",
            model_id="m",
            rubric_version="1.0",
            response={"faithfulness": 0.5, "task_completion": 0.5},
        )
        db.put_cache(entry)
        # Put again with different response
        entry2 = JudgeCacheEntry(
            cache_key="k1",
            model_id="m",
            rubric_version="1.0",
            response={"faithfulness": 0.9, "task_completion": 0.9},
        )
        db.put_cache(entry2)
        got = db.get_cache("k1")
        assert got is not None
        assert got.response["faithfulness"] == 0.9


# ── db: touch_cache for nonexistent key ──────────────────────────────────────


class TestDbTouchCacheNonexistent:
    """touch_cache for a key that doesn't exist should not crash."""

    def test_touch_cache_missing_key_no_error(self, db: Database) -> None:
        db.touch_cache("nonexistent")  # Should not raise


# ── db: export_run with empty results ────────────────────────────────────────


class TestDbExportEmptyResults:
    """export_run with a run that has records but no results."""

    def test_export_json_empty_results(self, db: Database, tmp_path: Path) -> None:
        run = EvalRun(config={})
        db.insert_run(run)
        rec = EvalRecord(input_text="i", output_text="o", run_id=run.run_id)
        db.insert_record(rec)
        out = tmp_path / "empty_results.json"
        db.export_run(run.run_id, out, fmt="json")
        data = json.loads(out.read_text())
        assert len(data["results"]) == 0
        assert len(data["records"]) == 1


# ── evaluator: progress callback ─────────────────────────────────────────────


class TestEvaluatorProgressCallback:
    """progress_cb parameter in evaluate() is untested."""

    @pytest.mark.asyncio
    async def test_progress_callback_called(
        self, db: Database, httpx_mock: HTTPXMock, openrouter_payload: dict
    ) -> None:
        run = EvalRun(config={}, judge_model="judge/free")
        db.insert_run(run)
        records = [
            EvalRecord(input_text=f"i{x}", output_text=f"o{x}", run_id=run.run_id)
            for x in range(5)
        ]
        for rec in records:
            db.insert_record(rec)
        for _ in range(5):
            httpx_mock.add_response(
                url="https://openrouter.ai/api/v1/chat/completions",
                method="POST",
                json=openrouter_payload,
            )
        ev = LLMEvaluator(
            db=db,
            config=EvaluatorConfig(
                api_key="test",
                judges=["judge/free"],
                rubric=BUILTIN_RUBRIC_V1,
                concurrency=2,
            ),
        )
        calls = []

        def progress(done: int, total: int) -> None:
            calls.append((done, total))

        await ev.evaluate(run, records, progress_cb=progress)
        assert len(calls) > 0
        assert calls[-1] == (5, 5)


# ── evaluator: evaluate with empty records list ──────────────────────────────


class TestEvaluatorEmptyRecords:
    """evaluate() with empty records list is untested."""

    @pytest.mark.asyncio
    async def test_evaluate_empty_records_returns_empty_list(
        self, db: Database
    ) -> None:
        run = EvalRun(config={}, judge_model="judge/free")
        db.insert_run(run)
        ev = LLMEvaluator(
            db=db,
            config=EvaluatorConfig(
                api_key="test",
                judges=["judge/free"],
                rubric=BUILTIN_RUBRIC_V1,
            ),
        )
        results = await ev.evaluate(run, [])
        assert results == []


# ── evaluator: _call_judge with OpenRouter error format ──────────────────────


class TestCallJudgeOpenRouterError:
    """_call_judge with OpenRouter application-level error is untested."""

    @pytest.mark.asyncio
    async def test_openrouter_app_error_raises_runtime_error(
        self, db: Database, httpx_mock: HTTPXMock
    ) -> None:
        run = EvalRun(config={}, judge_model="judge/free")
        db.insert_run(run)
        rec = EvalRecord(input_text="hi", output_text="hello", run_id=run.run_id)
        db.insert_record(rec)
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={"error": {"message": "Model not found"}},
        )
        ev = LLMEvaluator(
            db=db,
            config=EvaluatorConfig(
                api_key="test",
                judges=["judge/free"],
                rubric=BUILTIN_RUBRIC_V1,
                concurrency=1,
            ),
        )
        results = await ev.evaluate(run, [rec])
        assert results[0].error is not None
        assert "Model not found" in results[0].error


# ── evaluator: _call_judge with empty choices ────────────────────────────────


class TestCallJudgeEmptyChoices:
    """_call_judge with empty choices array is untested."""

    @pytest.mark.asyncio
    async def test_empty_choices_raises_runtime_error(
        self, db: Database, httpx_mock: HTTPXMock
    ) -> None:
        run = EvalRun(config={}, judge_model="judge/free")
        db.insert_run(run)
        rec = EvalRecord(input_text="hi", output_text="hello", run_id=run.run_id)
        db.insert_record(rec)
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={"choices": []},
        )
        ev = LLMEvaluator(
            db=db,
            config=EvaluatorConfig(
                api_key="test",
                judges=["judge/free"],
                rubric=BUILTIN_RUBRIC_V1,
                concurrency=1,
            ),
        )
        results = await ev.evaluate(run, [rec])
        assert results[0].error is not None
        assert "no choices" in results[0].error


# ── evaluator: generate_all_feedback with multiple records ───────────────────


class TestGenerateAllFeedbackMultiple:
    """generate_all_feedback with multiple records is untested."""

    @pytest.mark.asyncio
    async def test_feedback_generated_for_failing_records(
        self, db: Database
    ) -> None:
        run = EvalRun(run_id="test-run", config={}, status=RunStatus.RUNNING)
        db.insert_run(run)
        records = [
            EvalRecord(record_id="r1", input_text="q1", output_text="a1", run_id="test-run"),
            EvalRecord(record_id="r2", input_text="q2", output_text="a2", run_id="test-run"),
        ]
        for rec in records:
            db.insert_record(rec)
        results = [
            EvalResult(
                record_id="r1", run_id="test-run",
                faithfulness=0.3, task_completion=0.3, combined_score=0.3,
                pass_fail=PassFail.FAIL, judge_model="j", judge_tried=["j"],
            ),
            EvalResult(
                record_id="r2", run_id="test-run",
                faithfulness=0.9, task_completion=0.9, combined_score=0.9,
                pass_fail=PassFail.PASS, judge_model="j", judge_tried=["j"],
            ),
        ]

        config = EvaluatorConfig(api_key="test", judges=["j"])
        evaluator = LLMEvaluator(db, config)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({"suggestions": ["be more accurate"]})
                }
            }]
        }

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)

        with patch("src.evaluator.httpx.AsyncClient") as MockClient:
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_context.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_context
            await evaluator.generate_all_feedback(run, records, results)

        # r1 (fail) should have feedback, r2 (pass) should not
        assert results[0].feedback is not None
        assert "suggestions" in results[0].feedback
        assert results[1].feedback is None


# ── evaluator: generate_feedback with network error ──────────────────────────


class TestGenerateFeedbackNetworkError:
    """generate_feedback with network-level error is untested."""

    @pytest.mark.asyncio
    async def test_generate_feedback_network_error_returns_none(
        self, db: Database
    ) -> None:
        config = EvaluatorConfig(api_key="test", judges=["j"])
        evaluator = LLMEvaluator(db, config)
        record = EvalRecord(input_text="q", output_text="a")
        result = EvalResult(
            record_id=record.record_id, run_id="r",
            faithfulness=0.3, task_completion=0.3, combined_score=0.3,
            pass_fail=PassFail.FAIL, judge_model="j", judge_tried=["j"],
        )
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=Exception("Connection refused"))
        feedback = await evaluator.generate_feedback(mock_client, record, result)
        assert feedback is None


# ── reporter: export_results unsupported format ──────────────────────────────


class TestExportResultsUnsupportedFormat:
    """export_results with unsupported format is untested."""

    def test_export_results_unsupported_format(self, tmp_path: Path) -> None:
        run = EvalRun(config={})
        results = [
            EvalResult(
                record_id="r1", run_id="run1",
                faithfulness=0.5, task_completion=0.5, combined_score=0.5,
                pass_fail=PassFail.FAIL, judge_model="m", judge_tried=["m"],
            ),
        ]
        with pytest.raises(ValueError, match="unsupported export format"):
            export_results(run, results, tmp_path / "x.xml", fmt="xml")


# ── reporter: render_table with judge_usage ──────────────────────────────────


class TestRenderTableWithJudgeUsage:
    """render_table with judge_usage is untested."""

    def test_render_table_shows_judge_usage(self) -> None:
        run = EvalRun(config={})
        results = [
            EvalResult(
                record_id="r1", run_id="run1",
                faithfulness=0.9, task_completion=0.8, combined_score=0.85,
                pass_fail=PassFail.PASS, judge_model="m1", judge_tried=["m1"],
            ),
            EvalResult(
                record_id="r2", run_id="run1",
                faithfulness=0.4, task_completion=0.5, combined_score=0.45,
                pass_fail=PassFail.FAIL, judge_model="m2", judge_tried=["m2"],
            ),
        ]
        summary = build_summary(run, results)
        table = render_table(summary)
        assert "m1" in table
        assert "m2" in table


# ── reporter: ascii_histogram single bucket ──────────────────────────────────


class TestAsciiHistogramSingleValue:
    """ascii_histogram with single value is untested."""

    def test_single_value(self) -> None:
        out = ascii_histogram([0.5], buckets=10)
        assert isinstance(out, str)
        assert "0.50" in out or "0.5" in out

    def test_all_same_bucket(self) -> None:
        out = ascii_histogram([0.1, 0.15, 0.18], buckets=10)
        assert isinstance(out, str)
        assert "#" in out


# ── reporter: build_summary with all passing ─────────────────────────────────


class TestBuildSummaryAllPass:
    """build_summary when all records pass."""

    def test_all_pass_100_percent(self) -> None:
        run = EvalRun(config={})
        results = [
            EvalResult(
                record_id=f"r{i}", run_id="run1",
                faithfulness=0.9, task_completion=0.9, combined_score=0.9,
                pass_fail=PassFail.PASS, judge_model="m", judge_tried=["m"],
            )
            for i in range(5)
        ]
        summary = build_summary(run, results)
        assert summary.pass_rate == 1.0
        assert summary.passed == 5
        assert summary.failed == 0


# ── reporter: build_summary with all failing ─────────────────────────────────


class TestBuildSummaryAllFail:
    """build_summary when all records fail."""

    def test_all_fail_0_percent(self) -> None:
        run = EvalRun(config={})
        results = [
            EvalResult(
                record_id=f"r{i}", run_id="run1",
                faithfulness=0.1, task_completion=0.1, combined_score=0.1,
                pass_fail=PassFail.FAIL, judge_model="m", judge_tried=["m"],
            )
            for i in range(5)
        ]
        summary = build_summary(run, results)
        assert summary.pass_rate == 0.0
        assert summary.passed == 0
        assert summary.failed == 5


# ── reporter: judge_usage empty ───────────────────────────────────────────────


class TestJudgeUsageEmpty:
    """judge_usage with empty results."""

    def test_judge_usage_empty(self) -> None:
        assert judge_usage([]) == {}


# ── trend: compute_trends with enough runs for display ───────────────────────


class TestComputeTrendsEnoughRuns:
    """compute_trends with enough runs for display is untested."""

    def test_trend_with_three_runs(self, db: Database) -> None:
        for score in [0.8, 0.75, 0.85]:
            run = EvalRun(
                config={}, judge_model="j", status=RunStatus.COMPLETED,
                record_count=1, mean_score=score, pass_rate=0.8,
            )
            db.insert_run(run)
        result = compute_trends(db)
        assert result.total_runs == 3
        assert len(result.points) == 3
        assert result.mean_score_overall is not None
        assert result.latest_score == 0.85
        assert result.earliest_score == 0.8

    def test_trend_with_judge_model_filter(self, db: Database) -> None:
        for score, judge in [(0.8, "j1"), (0.7, "j2"), (0.9, "j1")]:
            run = EvalRun(
                config={}, judge_model=judge, status=RunStatus.COMPLETED,
                record_count=1, mean_score=score, pass_rate=0.8,
            )
            db.insert_run(run)
        result = compute_trends(db, judge_model="j1")
        assert result.total_runs == 2

    def test_trend_with_rubric_filter(self, db: Database) -> None:
        for score, rid in [(0.8, "rub-a"), (0.7, "rub-b"), (0.9, "rub-a")]:
            run = EvalRun(
                config={}, judge_model="j", status=RunStatus.COMPLETED,
                record_count=1, mean_score=score, pass_rate=0.8,
                rubric_template_id=rid,
            )
            db.insert_run(run)
        result = compute_trends(db, rubric_template_id="rub-a")
        assert result.total_runs == 2

    def test_trend_with_since_filter(self, db: Database) -> None:
        from datetime import UTC, datetime
        for _i, score in enumerate([0.8, 0.75, 0.85]):
            run = EvalRun(
                config={}, judge_model="j", status=RunStatus.COMPLETED,
                record_count=1, mean_score=score, pass_rate=0.8,
            )
            db.insert_run(run)
        # Use a very old date to include all
        result = compute_trends(db, since=datetime(2000, 1, 1, tzinfo=UTC))
        assert result.total_runs == 3

    def test_trend_since_filters_all_out(self, db: Database) -> None:
        from datetime import UTC, datetime
        run = EvalRun(
            config={}, judge_model="j", status=RunStatus.COMPLETED,
            record_count=1, mean_score=0.8, pass_rate=0.8,
        )
        db.insert_run(run)
        # Future date filters out everything
        result = compute_trends(db, since=datetime(2099, 1, 1, tzinfo=UTC))
        assert result.total_runs == 0
        assert result.points == []


# ── trend: regression detection ──────────────────────────────────────────────


class TestTrendRegressionDetection:
    """Regression detection in compute_trends is untested."""

    def test_regression_detected(self, db: Database) -> None:
        # Need at least MIN_RUNS_REGRESSION (5) runs to trigger regression check
        scores = [0.9, 0.85, 0.88, 0.82, 0.65]  # Last one drops > 0.10 from 0.82
        for score in scores:
            run = EvalRun(
                config={}, judge_model="j", status=RunStatus.COMPLETED,
                record_count=1, mean_score=score, pass_rate=0.8,
            )
            db.insert_run(run)
        result = compute_trends(db)
        assert result.has_regression is True
        # The last point should be flagged
        assert result.points[-1].is_regression is True

    def test_no_regression_stable_scores(self, db: Database) -> None:
        scores = [0.8, 0.81, 0.79, 0.82, 0.8]
        for score in scores:
            run = EvalRun(
                config={}, judge_model="j", status=RunStatus.COMPLETED,
                record_count=1, mean_score=score, pass_rate=0.8,
            )
            db.insert_run(run)
        result = compute_trends(db)
        assert result.has_regression is False
        assert all(not p.is_regression for p in result.points)


# ── trend: skipped runs (not completed or no mean_score) ────────────────────


class TestTrendSkippedRuns:
    """Runs that are not completed or have no mean_score should be skipped."""

    def test_running_runs_excluded(self, db: Database) -> None:
        run = EvalRun(
            config={}, judge_model="j", status=RunStatus.RUNNING,
            record_count=1, mean_score=0.8, pass_rate=0.8,
        )
        db.insert_run(run)
        result = compute_trends(db)
        assert result.total_runs == 0

    def test_null_mean_score_excluded(self, db: Database) -> None:
        run = EvalRun(
            config={}, judge_model="j", status=RunStatus.COMPLETED,
            record_count=1, mean_score=None, pass_rate=0.8,
        )
        db.insert_run(run)
        result = compute_trends(db)
        assert result.total_runs == 0


# ── gate: check with pass_rate=None ──────────────────────────────────────────


class TestGateCheckPassRateNone:
    """GateRunner.check when pass_rate is None is untested."""

    def test_check_pass_rate_none_raises(self, db: Database) -> None:
        run_id = "no-pass-rate"
        run = EvalRun(
            run_id=run_id, status=RunStatus.COMPLETED,
            record_count=5, mean_score=0.8, pass_rate=None,
            config={}, judge_model="j",
        )
        db.insert_run(run)
        runner = GateRunner(db)
        with pytest.raises(ValueError, match="no pass_rate"):
            runner.check(run_id, threshold=0.7)


# ── gate: suggest_baseline with single run ─────────────────────────────────────


class TestGateSuggestBaselineSingleRun:
    """suggest_baseline with a single run is untested."""

    def test_suggest_baseline_single_run(self, db: Database) -> None:
        run = EvalRun(
            run_id="single", status=RunStatus.COMPLETED,
            record_count=5, mean_score=0.75, pass_rate=0.75,
            config={}, judge_model="j",
        )
        db.insert_run(run)
        runner = GateRunner(db)
        suggestion = runner.suggest_baseline()
        assert suggestion is not None
        assert suggestion["runs_analyzed"] == 1
        assert suggestion["recommended_baseline"] == 0.75


# ── gate: suggest_baseline with many runs ──────────────────────────────────────


class TestGateSuggestBaselineManyRuns:
    """suggest_baseline with many runs calculates percentiles correctly."""

    def test_suggest_baseline_many_runs(self, db: Database) -> None:
        rates = [0.5, 0.6, 0.7, 0.8, 0.9]
        for i, rate in enumerate(rates):
            run = EvalRun(
                run_id=f"run-{i}", status=RunStatus.COMPLETED,
                record_count=10, mean_score=rate, pass_rate=rate,
                config={}, judge_model="j",
            )
            db.insert_run(run)
        runner = GateRunner(db)
        suggestion = runner.suggest_baseline()
        assert suggestion is not None
        assert suggestion["runs_analyzed"] == 5
        # 25th percentile of [0.5, 0.6, 0.7, 0.8, 0.9] = index 1 = 0.6
        assert suggestion["recommended_baseline"] == 0.6
        assert suggestion["median_pass_rate"] == 0.7
        assert suggestion["min_pass_rate"] == 0.5
        assert suggestion["max_pass_rate"] == 0.9


# ── gate: CheckGateResult boundary conditions ────────────────────────────────


class TestCheckGateResultBoundary:
    """CheckGateResult boundary conditions."""

    def test_exactly_at_threshold_passes(self) -> None:
        result = CheckGateResult(
            run_id="boundary", pass_rate=0.7, threshold=0.7,
            passed=True, mean_score=0.7, record_count=10,
        )
        assert result.passed is True
        assert result.exit_code == 0

    def test_just_below_threshold_fails(self) -> None:
        result = CheckGateResult(
            run_id="below", pass_rate=0.699, threshold=0.7,
            passed=False, mean_score=0.699, record_count=10,
        )
        assert result.passed is False
        assert result.exit_code == 1

    def test_zero_threshold_always_passes(self) -> None:
        result = CheckGateResult(
            run_id="zero-thresh", pass_rate=0.01, threshold=0.0,
            passed=True, mean_score=0.01, record_count=10,
        )
        assert result.passed is True

    def test_threshold_1_requires_perfect(self) -> None:
        result = CheckGateResult(
            run_id="perfect", pass_rate=0.99, threshold=1.0,
            passed=False, mean_score=0.99, record_count=10,
        )
        assert result.passed is False


# ── calibrate: _compute_pair_agreement edge cases ────────────────────────────


class TestComputePairAgreement:
    """_compute_pair_agreement edge cases."""

    def test_no_shared_records_returns_zero(self) -> None:
        results = [
            EvalResult(
                record_id="r1", run_id="run", faithfulness=0.9,
                task_completion=0.9, combined_score=0.9,
                pass_fail=PassFail.PASS, judge_model="j1", judge_tried=["j1"],
            ),
            EvalResult(
                record_id="r2", run_id="run", faithfulness=0.3,
                task_completion=0.3, combined_score=0.3,
                pass_fail=PassFail.FAIL, judge_model="j2", judge_tried=["j2"],
            ),
        ]
        agreement = _compute_pair_agreement(results, ["j1", "j2"])
        # j1 and j2 have no shared records
        assert agreement["j1"]["j2"] == 0.0

    def test_all_agree_on_all_records(self) -> None:
        results = []
        for judge in ["j1", "j2"]:
            results.append(
                EvalResult(
                    record_id="r1", run_id="run", faithfulness=0.9,
                    task_completion=0.9, combined_score=0.9,
                    pass_fail=PassFail.PASS, judge_model=judge, judge_tried=[judge],
                )
            )
        agreement = _compute_pair_agreement(results, ["j1", "j2"])
        assert agreement["j1"]["j2"] == 1.0


# ── calibrate: asyncio_run ───────────────────────────────────────────────────


class TestAsyncioRun:
    """asyncio_run helper is untested."""

    def test_asyncio_run_fresh_loop(self) -> None:
        async def coro():
            return 42
        result = asyncio_run(coro())
        assert result == 42


# ── calibrate: CalibrationSummary.from_results with single judge ─────────────


class TestCalibrationSummarySingleJudge:
    """CalibrationSummary.from_results with a single judge."""

    def test_single_judge_zero_std_dev(self) -> None:
        results = [
            EvalResult(
                record_id="r1", run_id="run", faithfulness=0.8,
                task_completion=0.8, combined_score=0.8,
                pass_fail=PassFail.PASS, judge_model="j1", judge_tried=["j1"],
            ),
            EvalResult(
                record_id="r2", run_id="run", faithfulness=0.6,
                task_completion=0.6, combined_score=0.6,
                pass_fail=PassFail.FAIL, judge_model="j1", judge_tried=["j1"],
            ),
        ]
        summary = CalibrationSummary.from_results(results, "run", ["j1"])
        assert summary.total_records == 2
        assert summary.total_judges == 1
        assert summary.mean_std_dev == 0.0
        # Single judge: pass_agreement_rate is 1 (all agree with themselves)
        assert summary.pass_agreement_rate == 1


# ── CLI: rubric command tests ────────────────────────────────────────────────


class TestCLIRubric:
    """CLI rubric command is untested."""

    def test_rubric_list(self, db: Database) -> None:
        result = runner.invoke(app, ["rubric", "--list", "--db", str(db.path)])
        assert result.exit_code == 0
        assert "faithfulness-v1" in result.stdout

    def test_rubric_list_json(self, db: Database) -> None:
        result = runner.invoke(
            app, ["rubric", "--list", "--json", "--db", str(db.path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert isinstance(data, list)
        assert len(data) >= 5
        assert all("template_id" in t for t in data)

    def test_rubric_show(self, db: Database) -> None:
        result = runner.invoke(
            app, ["rubric", "--show", "faithfulness-v1", "--db", str(db.path)]
        )
        assert result.exit_code == 0
        assert "Faithfulness" in result.stdout

    def test_rubric_show_json(self, db: Database) -> None:
        result = runner.invoke(
            app, ["rubric", "--show", "faithfulness-v1", "--json", "--db", str(db.path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["template_id"] == "faithfulness-v1"

    def test_rubric_show_not_found(self, db: Database) -> None:
        result = runner.invoke(
            app, ["rubric", "--show", "nonexistent", "--db", str(db.path)]
        )
        assert result.exit_code == 1

    def test_rubric_create(self, db: Database, tmp_path: Path) -> None:
        yaml_file = tmp_path / "new_rubric.yaml"
        yaml_file.write_text(
            "dimensions:\n- name: my_dim\n  weight: 1.0\n  description: test\n"
            "scoring:\n  scale: 0-1\n"
            "output_format:\n  my_dim: float\n"
        )
        result = runner.invoke(
            app,
            [
                "rubric", "--create-name", "My Rubric",
                "--create-file", str(yaml_file), "--db", str(db.path),
            ],
        )
        assert result.exit_code == 0
        assert "created template" in result.stdout

    def test_rubric_delete(self, db: Database, tmp_path: Path) -> None:
        # Create a template first
        yaml_file = tmp_path / "del_rubric.yaml"
        yaml_file.write_text(
            "dimensions:\n- name: del_dim\n  weight: 1.0\n  description: test\n"
            "scoring:\n  scale: 0-1\n"
            "output_format:\n  del_dim: float\n"
        )
        create_result = runner.invoke(
            app,
            [
                "rubric", "--create-name", "Delete Me",
                "--create-file", str(yaml_file), "--db", str(db.path),
            ],
        )
        assert create_result.exit_code == 0
        # Extract template_id from the output
        template_id = create_result.stdout.strip().split(": ")[-1]
        delete_result = runner.invoke(
            app, ["rubric", "--delete", template_id, "--db", str(db.path)]
        )
        assert delete_result.exit_code == 0
        assert "deleted" in delete_result.stdout

    def test_rubric_no_action(self, db: Database) -> None:
        result = runner.invoke(app, ["rubric", "--db", str(db.path)])
        assert result.exit_code == 1


# ── CLI: trend command with data ────────────────────────────────────────────


class TestCLITrendWithData:
    """trend command with actual data is untested."""

    def test_trend_json_with_runs(self, db: Database) -> None:
        for score in [0.8, 0.75, 0.85]:
            run = EvalRun(
                config={}, judge_model="j", status=RunStatus.COMPLETED,
                record_count=1, mean_score=score, pass_rate=0.8,
            )
            db.insert_run(run)
        result = runner.invoke(
            app, ["trend", "--json", "--db", str(db.path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["total_runs"] == 3
        assert len(data["points"]) == 3

    def test_trend_table_with_runs(self, db: Database) -> None:
        for score in [0.8, 0.75, 0.85]:
            run = EvalRun(
                config={}, judge_model="j", status=RunStatus.COMPLETED,
                record_count=1, mean_score=score, pass_rate=0.8,
            )
            db.insert_run(run)
        result = runner.invoke(
            app, ["trend", "--db", str(db.path)]
        )
        assert result.exit_code == 0
        assert "Score Trend" in result.stdout

    def test_trend_with_judge_filter(self, db: Database) -> None:
        for score, judge in [(0.8, "j1"), (0.7, "j2"), (0.9, "j1")]:
            run = EvalRun(
                config={}, judge_model=judge, status=RunStatus.COMPLETED,
                record_count=1, mean_score=score, pass_rate=0.8,
            )
            db.insert_run(run)
        result = runner.invoke(
            app, ["trend", "--judge", "j1", "--json", "--db", str(db.path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["total_runs"] == 2


# ── CLI: gate suggest-baseline ───────────────────────────────────────────────


class TestCLIGateSuggestBaseline:
    """gate suggest-baseline CLI command is untested."""

    def test_gate_suggest_baseline_json(self, db: Database) -> None:
        run = EvalRun(
            run_id="run-1", status=RunStatus.COMPLETED,
            record_count=10, mean_score=0.8, pass_rate=0.8,
            config={}, judge_model="j",
        )
        db.insert_run(run)
        result = runner.invoke(
            app, ["gate", "--suggest-baseline", "--json", "--db", str(db.path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "recommended_baseline" in data
        assert data["runs_analyzed"] == 1

    def test_gate_suggest_baseline_no_runs(self, db: Database) -> None:
        result = runner.invoke(
            app, ["gate", "--suggest-baseline", "--db", str(db.path)]
        )
        assert result.exit_code == 2


# ── CLI: run with --feedback flag ────────────────────────────────────────────


class TestCLIRunWithFeedback:
    """run command with --feedback flag is untested."""

    def test_run_with_feedback(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        # Mock evaluation response
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "faithfulness": 0.3,
                            "task_completion": 0.3,
                            "reasoning": "too short",
                            "faithfulness_reasoning": "missing info",
                            "task_completion_reasoning": "incomplete",
                        })
                    }
                }]
            },
        )
        # Mock feedback response
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({"suggestions": ["Add more detail"]})
                    }
                }]
            },
        )
        result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--feedback", "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 1  # Low score → fail
        assert "suggestion" in result.stdout.lower() or "improvement" in result.stdout.lower()


# ── CLI: run with --degrade flag ─────────────────────────────────────────────


class TestCLIRunWithDegrade:
    """run command with --degrade flag is untested."""

    def test_run_with_degrade_uses_heuristic(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        p.write_text('{"input":"What is 2+2?","output":"The answer is 4","reference":"4"}\n')
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            status_code=500,
        )
        result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--degrade", "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        # Should succeed with heuristic fallback
        assert result.exit_code in (0, 1)


# ── CLI: run with --compare-judges ───────────────────────────────────────────


class TestCLIRunCompareJudges:
    """run command with --compare-judges is untested."""

    def test_run_compare_judges(
        self, tmp_path: Path, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        db_path = tmp_path / "compare.db"
        p = tmp_path / "input.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        cache_path = tmp_path / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "faithfulness": 0.9,
                            "task_completion": 0.85,
                            "reasoning": "ok",
                            "faithfulness_reasoning": "ok",
                            "task_completion_reasoning": "ok",
                        })
                    }
                }]
            },
        )
        result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--compare-judges", "--db", str(db_path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 0


# ── CLI: run with --pass-threshold boundary ──────────────────────────────────


class TestCLIRunPassThreshold:
    """run command with custom pass-threshold."""

    def test_run_custom_pass_threshold(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "faithfulness": 0.8,
                            "task_completion": 0.8,
                            "reasoning": "ok",
                            "faithfulness_reasoning": "ok",
                            "task_completion_reasoning": "ok",
                        })
                    }
                }]
            },
        )
        # With threshold 0.5, score 0.8 should pass
        result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--pass-threshold", "0.5",
                "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 0


# ── CLI: run with --resume flag ──────────────────────────────────────────────


class TestCLIRunResume:
    """run command with --resume flag is untested."""

    def test_run_resume_skips_evaluated(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        p.write_text(
            '{"input":"q1","output":"a1"}\n{"input":"q2","output":"a2"}\n'
        )
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        # First run: evaluate both records
        for _ in range(2):
            httpx_mock.add_response(
                url="https://openrouter.ai/api/v1/chat/completions",
                method="POST",
                json={
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "faithfulness": 0.9,
                                "task_completion": 0.9,
                                "reasoning": "ok",
                                "faithfulness_reasoning": "ok",
                                "task_completion_reasoning": "ok",
                            })
                        }
                    }]
                },
            )
        first_result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert first_result.exit_code == 0

        # Second run with --resume: all records already evaluated → no new results
        # The CLI creates a new run, inserts records, then evaluates with resume=True.
        # Since all records already have results in DB, evaluate returns empty list.
        # The run completes with 0 results → exit code 0 (no failures)
        second_result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--resume", "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        # With resume, already-evaluated records are skipped.
        # The run completes successfully with whatever records were new (none in this case).
        # Exit code 0 because no failures occurred.
        assert second_result.exit_code == 0


# ── CLI: export with CSV format ──────────────────────────────────────────────


class TestCLIExportCsv:
    """export command with CSV format is untested."""

    def test_export_csv(self, db: Database) -> None:
        run = EvalRun(config={}, judge_model="j", status=RunStatus.COMPLETED)
        db.insert_run(run)
        rec = EvalRecord(input_text="q", output_text="a", run_id=run.run_id)
        db.insert_record(rec)
        result = EvalResult(
            record_id=rec.record_id, run_id=run.run_id,
            faithfulness=0.8, task_completion=0.8, combined_score=0.8,
            pass_fail=PassFail.PASS, judge_model="j", judge_tried=["j"],
        )
        db.insert_result(result)
        out_path = db.path.parent / "export.csv"
        result = runner.invoke(
            app,
            [
                "export", "--run-id", run.run_id,
                "--format", "csv",
                "--output-file", str(out_path),
                "--db", str(db.path),
            ],
        )
        assert result.exit_code == 0
        assert out_path.exists()
        content = out_path.read_text()
        assert "faithfulness" in content
        assert "0.8" in content


# ── CLI: report with table output ────────────────────────────────────────────


class TestCLIReportTableOutput:
    """report command with table output is untested."""

    def test_report_table_output(self, db: Database) -> None:
        run = EvalRun(config={}, judge_model="j", status=RunStatus.COMPLETED)
        db.insert_run(run)
        rec = EvalRecord(input_text="q", output_text="a", run_id=run.run_id)
        db.insert_record(rec)
        result = EvalResult(
            record_id=rec.record_id, run_id=run.run_id,
            faithfulness=0.8, task_completion=0.8, combined_score=0.8,
            pass_fail=PassFail.PASS, judge_model="j", judge_tried=["j"],
        )
        db.insert_result(result)
        result = runner.invoke(
            app, ["report", "--run-id", run.run_id, "--db", str(db.path)]
        )
        assert result.exit_code == 0
        assert "Summary" in result.stdout or "pass" in result.stdout.lower()


# ── CLI: report with JSON output ─────────────────────────────────────────────


class TestCLIReportJsonOutput:
    """report command with JSON output is untested."""

    def test_report_json_output(self, db: Database) -> None:
        run = EvalRun(config={}, judge_model="j", status=RunStatus.COMPLETED)
        db.insert_run(run)
        rec = EvalRecord(input_text="q", output_text="a", run_id=run.run_id)
        db.insert_record(rec)
        result = EvalResult(
            record_id=rec.record_id, run_id=run.run_id,
            faithfulness=0.8, task_completion=0.8, combined_score=0.8,
            pass_fail=PassFail.PASS, judge_model="j", judge_tried=["j"],
        )
        db.insert_result(result)
        result = runner.invoke(
            app, ["report", "--run-id", run.run_id, "--output", "json", "--db", str(db.path)]
        )
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["run_id"] == run.run_id


# ── CLI: list-runs with custom limit ─────────────────────────────────────────


class TestCLIListRunsCustomLimit:
    """list-runs with custom limit is untested."""

    def test_list_runs_custom_limit(self, db: Database) -> None:
        for _i in range(5):
            db.insert_run(EvalRun(config={}))
        result = runner.invoke(
            app, ["list-runs", "--limit", "2", "--db", str(db.path)]
        )
        assert result.exit_code == 0


# ── CLI: run with --quiet flag ───────────────────────────────────────────────


class TestCLIRunQuiet:
    """run command with --quiet flag is untested."""

    def test_run_quiet(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "faithfulness": 0.9,
                            "task_completion": 0.85,
                            "reasoning": "ok",
                            "faithfulness_reasoning": "ok",
                            "task_completion_reasoning": "ok",
                        })
                    }
                }]
            },
        )
        result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--quiet", "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 0


# ── CLI: run with --output table to file ─────────────────────────────────────


class TestCLIRunOutputTableToFile:
    """run command with --output table --output-file is untested."""

    def test_run_output_table_to_file(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "faithfulness": 0.9,
                            "task_completion": 0.85,
                            "reasoning": "ok",
                            "faithfulness_reasoning": "ok",
                            "task_completion_reasoning": "ok",
                        })
                    }
                }]
            },
        )
        out_file = db.path.parent / "output.txt"
        result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--output", "table",
                "--output-file", str(out_file),
                "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()


# ── CLI: run with stdin input ─────────────────────────────────────────────────


class TestCLIRunStdin:
    """run command with stdin input is untested."""

    def test_run_stdin(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "faithfulness": 0.9,
                            "task_completion": 0.85,
                            "reasoning": "ok",
                            "faithfulness_reasoning": "ok",
                            "task_completion_reasoning": "ok",
                        })
                    }
                }]
            },
        )
        result = runner.invoke(
            app,
            [
                "run", "-", "--judge", "x/free",
                "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
            input='{"input":"hi","output":"hello"}\n',
        )
        assert result.exit_code == 0


# ── CLI: run with --format csv ───────────────────────────────────────────────


class TestCLIRunCsvFormat:
    """run command with --format csv is untested."""

    def test_run_csv_format(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.csv"
        p.write_text("prompt,response,reference\nhello,world,truth\n")
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "faithfulness": 0.9,
                            "task_completion": 0.85,
                            "reasoning": "ok",
                            "faithfulness_reasoning": "ok",
                            "task_completion_reasoning": "ok",
                        })
                    }
                }]
            },
        )
        result = runner.invoke(
            app,
            [
                "run", str(p), "--format", "csv",
                "--input-col", "prompt", "--output-col", "response",
                "--reference-col", "reference",
                "--judge", "x/free",
                "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 0


# ── CLI: run with --sample and --limit combined ──────────────────────────────


class TestCLIRunSampleAndLimit:
    """run command with --sample and --limit combined is untested."""

    def test_run_sample_with_limit(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        lines = '\n'.join(f'{{"input":"q{i}","output":"a{i}"}}' for i in range(10))
        p.write_text(lines + "\n")
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        # Only 3 records should be evaluated (sample 5, limit 3)
        for _ in range(3):
            httpx_mock.add_response(
                url="https://openrouter.ai/api/v1/chat/completions",
                method="POST",
                json={
                    "choices": [{
                        "message": {
                            "content": json.dumps({
                                "faithfulness": 0.9,
                                "task_completion": 0.85,
                                "reasoning": "ok",
                                "faithfulness_reasoning": "ok",
                                "task_completion_reasoning": "ok",
                            })
                        }
                    }]
                },
            )
        result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--sample", "5", "--limit", "3",
                "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 0


# ── CLI: run with nonexistent judge ───────────────────────────────────────────


class TestCLIRunNoJudges:
    """run command when no judges are available."""

    def test_run_no_judges_available(
        self, db: Database, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all judges fail with permanent error, evaluation fails."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        # No API key → should exit with code 2
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = runner.invoke(
            app,
            [
                "run", str(p),
                "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 2


# ── CLI: run with --timeout flag ─────────────────────────────────────────────


class TestCLIRunTimeout:
    """run command with custom timeout."""

    def test_run_custom_timeout(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "faithfulness": 0.9,
                            "task_completion": 0.85,
                            "reasoning": "ok",
                            "faithfulness_reasoning": "ok",
                            "task_completion_reasoning": "ok",
                        })
                    }
                }]
            },
        )
        result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--timeout", "120",
                "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 0


# ── CLI: run with --rpm-limit flag ────────────────────────────────────────────


class TestCLIRunRpmLimit:
    """run command with --rpm-limit flag."""

    def test_run_rpm_limit(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "faithfulness": 0.9,
                            "task_completion": 0.85,
                            "reasoning": "ok",
                            "faithfulness_reasoning": "ok",
                            "task_completion_reasoning": "ok",
                        })
                    }
                }]
            },
        )
        result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--rpm-limit", "30",
                "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 0


# ── CLI: run with --max-fallbacks 0 ──────────────────────────────────────────


class TestCLIRunMaxFallbacksZero:
    """run command with --max-fallbacks 0."""

    def test_run_max_fallbacks_zero(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "faithfulness": 0.9,
                            "task_completion": 0.85,
                            "reasoning": "ok",
                            "faithfulness_reasoning": "ok",
                            "task_completion_reasoning": "ok",
                        })
                    }
                }]
            },
        )
        result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--max-fallbacks", "0",
                "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 0


# ── CLI: run with --no-fallback ──────────────────────────────────────────────


class TestCLIRunNoFallback:
    """run command with --no-fallback flag."""

    def test_run_no_fallback(
        self, db: Database, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = db.path.parent / "input.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        cache_path = db.path.parent / "j.json"
        cache_path.write_text(
            json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "faithfulness": 0.9,
                            "task_completion": 0.85,
                            "reasoning": "ok",
                            "faithfulness_reasoning": "ok",
                            "task_completion_reasoning": "ok",
                        })
                    }
                }]
            },
        )
        result = runner.invoke(
            app,
            [
                "run", str(p), "--judge", "x/free",
                "--no-fallback",
                "--db", str(db.path),
                "--judges-cache", str(cache_path), "--yes",
            ],
        )
        assert result.exit_code == 0


# ── CLI: run with --verbose flag ─────────────────────────────────────────────


# ── CLI: _judge_list helper ──────────────────────────────────────────────────


class TestJudgeList:
    """_judge_list helper is untested."""

    def test_explicit_judge_first(self, tmp_path: Path) -> None:
        cache = tmp_path / "j.json"
        cache.write_text(json.dumps({
            "models": [
                {"id": "a/free", "name": "A", "context_length": 100, "free": True},
                {"id": "b/free", "name": "B", "context_length": 200, "free": True},
            ]
        }))
        registry = JudgeRegistry(cache_path=cache)
        result = _judge_list("b/free", registry, no_fallback=False, max_fallbacks=3)
        assert result[0] == "b/free"
        assert "a/free" in result

    def test_no_fallback_returns_single(self, tmp_path: Path) -> None:
        cache = tmp_path / "j.json"
        cache.write_text(json.dumps({
            "models": [
                {"id": "a/free", "name": "A", "context_length": 100, "free": True},
                {"id": "b/free", "name": "B", "context_length": 200, "free": True},
            ]
        }))
        registry = JudgeRegistry(cache_path=cache)
        result = _judge_list(None, registry, no_fallback=True, max_fallbacks=3)
        assert len(result) == 1

    def test_max_fallbacks_limits_judges(self, tmp_path: Path) -> None:
        cache = tmp_path / "j.json"
        cache.write_text(json.dumps({
            "models": [
                {"id": f"j{i}", "name": f"J{i}", "context_length": 100, "free": True}
                for i in range(5)
            ]
        }))
        registry = JudgeRegistry(cache_path=cache)
        result = _judge_list(None, registry, no_fallback=False, max_fallbacks=2)
        # max_fallbacks=2 means max(1, 2+1) = 3 judges
        assert len(result) == 3


# ── CLI: _get_api_key helper ─────────────────────────────────────────────────


class TestGetApiKey:
    """_get_api_key helper is untested."""

    def test_get_api_key_returns_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "my-key")
        assert _get_api_key() == "my-key"

    def test_get_api_key_missing_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        assert _get_api_key() == ""


# ── judges: fetch with HTTP error ────────────────────────────────────────────


class TestJudgesFetchHttpError:
    """JudgeRegistry.fetch with HTTP error is untested."""

    def test_fetch_http_error_raises(self, tmp_path: Path, httpx_mock: HTTPXMock) -> None:
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/models",
            method="GET",
            status_code=500,
        )
        registry = JudgeRegistry(cache_path=tmp_path / "j.json")
        with pytest.raises(httpx.HTTPStatusError):
            registry.fetch(refresh=True)


# ── judges: fetch with malformed response ────────────────────────────────────


class TestJudgesFetchMalformedResponse:
    """JudgeRegistry.fetch with malformed response is untested."""

    def test_fetch_no_data_key_returns_empty_list(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/models",
            method="GET",
            json={"unexpected": "format"},
        )
        registry = JudgeRegistry(cache_path=tmp_path / "j.json")
        models = registry.fetch(refresh=True)
        assert models == []


# ── judges: fetch with models missing id field ───────────────────────────────


class TestJudgesFetchMissingId:
    """JudgeRegistry.fetch with models missing id field."""

    def test_fetch_model_missing_id_skipped(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/models",
            method="GET",
            json={
                "data": [
                    {"name": "No ID model", "context_length": 100, "pricing": {"prompt": "0", "completion": "0"}},
                    {"id": "good-model", "name": "Good", "context_length": 200, "pricing": {"prompt": "0", "completion": "0"}},
                ]
            },
        )
        registry = JudgeRegistry(cache_path=tmp_path / "j.json")
        models = registry.fetch(refresh=True)
        # Model with empty id should still be included (id defaults to "")
        # But the one with "good-model" should be present
        ids = [m.id for m in models]
        assert "good-model" in ids


# ── judges: list with valid cache ─────────────────────────────────────────────


class TestJudgesListWithCache:
    """JudgeRegistry.list with valid cache is untested."""

    def test_list_returns_cache_contents(self, tmp_path: Path) -> None:
        cache = tmp_path / "j.json"
        cache.write_text(json.dumps({
            "models": [
                {"id": "cached/model", "name": "Cached", "context_length": 500, "free": True}
            ]
        }))
        registry = JudgeRegistry(cache_path=cache)
        models = registry.list()
        assert len(models) == 1
        assert models[0].id == "cached/model"


# ── judges: JudgeRegistry without cache path ──────────────────────────────────


class TestJudgesRegistryNoCachePath:
    """JudgeRegistry without cache path falls back to default."""

    def test_no_cache_path_uses_default(self) -> None:
        registry = JudgeRegistry(cache_path=None)
        assert registry.cache_path == Path.home() / ".eval-harness" / "judges.json"


# ── models: EvalResult with boundary scores ──────────────────────────────────


class TestEvalResultBoundaryScores:
    """EvalResult with boundary scores (0.0 and 1.0)."""

    def test_score_exactly_zero_valid(self) -> None:
        result = EvalResult(
            record_id="r1", run_id="run1",
            faithfulness=0.0, task_completion=0.0, combined_score=0.0,
            pass_fail=PassFail.FAIL, judge_model="m",
        )
        assert result.faithfulness == 0.0

    def test_score_exactly_one_valid(self) -> None:
        result = EvalResult(
            record_id="r1", run_id="run1",
            faithfulness=1.0, task_completion=1.0, combined_score=1.0,
            pass_fail=PassFail.PASS, judge_model="m",
        )
        assert result.faithfulness == 1.0

    def test_score_above_one_invalid(self) -> None:
        with pytest.raises(Exception):
            EvalResult(
                record_id="r1", run_id="run1",
                faithfulness=1.001, task_completion=0.5, combined_score=0.75,
                pass_fail=PassFail.PASS, judge_model="m",
            )

    def test_score_below_zero_invalid(self) -> None:
        with pytest.raises(Exception):
            EvalResult(
                record_id="r1", run_id="run1",
                faithfulness=-0.001, task_completion=0.5, combined_score=0.25,
                pass_fail=PassFail.PASS, judge_model="m",
            )


# ── models: EvalRun with all fields set ──────────────────────────────────────


class TestEvalRunAllFields:
    """EvalRun with all optional fields set."""

    def test_eval_run_with_all_fields(self) -> None:
        run = EvalRun(
            config={"key": "value"},
            record_count=10,
            rubric_id="custom-rubric",
            judge_model="judge/model",
            status=RunStatus.COMPLETED,
            mean_score=0.85,
            pass_rate=0.9,
            eval_time_seconds=45.5,
        )
        assert run.mean_score == 0.85
        assert run.pass_rate == 0.9
        assert run.eval_time_seconds == 45.5


# ── models: JudgeCacheEntry with custom hits ─────────────────────────────────


class TestJudgeCacheEntryCustomHits:
    """JudgeCacheEntry with custom hits value."""

    def test_custom_hits_value(self) -> None:
        entry = JudgeCacheEntry(
            cache_key="k", model_id="m", rubric_version="1.0",
            response={"faithfulness": 0.5, "task_completion": 0.5},
            hits=10,
        )
        assert entry.hits == 10
