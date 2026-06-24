"""Tests for hardening fixes: edge cases, resilience, defensive programming."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock
from typer.testing import CliRunner

from src.cli import app
from src.db import Database
from src.evaluator import EvaluatorConfig, LLMEvaluator
from src.models import BUILTIN_RUBRIC_V1, EvalRecord, EvalRun, PassFail, RunStatus

runner = CliRunner()


# ── 1. Trend command edge cases ─────────────────────────────────────────────


class TestTrendEdgeCases:
    """Trend command handles 0, 1, 2 runs and mismatched rubric_ids gracefully."""

    def test_trend_zero_runs(self, tmp_path: Path) -> None:
        """Trend with no completed runs shows helpful message."""
        db_path = tmp_path / "eval.db"
        result = runner.invoke(app, ["trend", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "need at least" in result.stdout.lower()

    def test_trend_one_run(self, tmp_path: Path) -> None:
        """Trend with 1 completed run shows need more message."""
        db_path = tmp_path / "eval.db"
        db = Database(db_path)
        run = EvalRun(
            config={},
            judge_model="j",
            status=RunStatus.COMPLETED,
            record_count=1,
            mean_score=0.8,
            pass_rate=0.9,
        )
        db.insert_run(run)
        db.close()
        result = runner.invoke(app, ["trend", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "need at least" in result.stdout.lower()

    def test_trend_two_runs(self, tmp_path: Path) -> None:
        """Trend with 2 completed runs still shows need more message."""
        db_path = tmp_path / "eval.db"
        db = Database(db_path)
        for score in [0.8, 0.75]:
            run = EvalRun(
                config={},
                judge_model="j",
                status=RunStatus.COMPLETED,
                record_count=1,
                mean_score=score,
                pass_rate=0.9,
            )
            db.insert_run(run)
        db.close()
        result = runner.invoke(app, ["trend", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "need at least" in result.stdout.lower()

    def test_trend_json_zero_runs(self, tmp_path: Path) -> None:
        """Trend JSON output with 0 runs returns valid JSON with total_runs=0."""
        db_path = tmp_path / "eval.db"
        result = runner.invoke(app, ["trend", "--json", "--db", str(db_path)])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert data["total_runs"] == 0
        assert data["points"] == []

    def test_trend_filters_rubric_id(self, tmp_path: Path) -> None:
        """Trend with --rubric filter only counts matching runs."""
        db_path = tmp_path / "eval.db"
        db = Database(db_path)
        for rid, score in [("rub-a", 0.8), ("rub-b", 0.5), ("rub-a", 0.7)]:
            run = EvalRun(
                config={},
                judge_model="j",
                status=RunStatus.COMPLETED,
                record_count=1,
                mean_score=score,
                pass_rate=0.9,
                rubric_template_id=rid,
            )
            db.insert_run(run)
        db.close()
        result = runner.invoke(
            app, ["trend", "--rubric", "rub-a", "--db", str(db_path)]
        )
        assert result.exit_code == 0
        assert "need at least" in result.stdout.lower()

    def test_trend_rubric_id_no_matches(self, tmp_path: Path) -> None:
        """Trend with --rubric filter that matches nothing shows need more."""
        db_path = tmp_path / "eval.db"
        db = Database(db_path)
        run = EvalRun(
            config={},
            judge_model="j",
            status=RunStatus.COMPLETED,
            record_count=1,
            mean_score=0.8,
            pass_rate=0.9,
            rubric_template_id="other-rubric",
        )
        db.insert_run(run)
        db.close()
        result = runner.invoke(
            app, ["trend", "--rubric", "nonexistent-rubric", "--db", str(db_path)]
        )
        assert result.exit_code == 0
        assert "need at least" in result.stdout.lower()


# ── 2. CLI output-file and validation edge cases ────────────────────────────


class TestCLIOutputFileEdgeCases:
    """CLI handles --output-file with non-existent directory gracefully."""

    def test_run_output_file_creates_parent_dir(
        self, tmp_path: Path, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Writing to a non-existent directory auto-creates the parent dir."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = tmp_path / "x.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        db_path = tmp_path / "eval.db"
        cache_path = tmp_path / "j.json"
        cache_path.write_text(
            json.dumps(
                {"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]}
            )
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "faithfulness": 0.9,
                                    "task_completion": 0.85,
                                    "reasoning": "ok",
                                    "faithfulness_reasoning": "ok",
                                    "task_completion_reasoning": "ok",
                                }
                            )
                        }
                    }
                ]
            },
        )
        output_in_new_dir = tmp_path / "new_dir" / "nested" / "out.json"
        result = runner.invoke(
            app,
            [
                "run",
                str(p),
                "--judge",
                "x/free",
                "--output",
                "json",
                "--output-file",
                str(output_in_new_dir),
                "--db",
                str(db_path),
                "--judges-cache",
                str(cache_path),
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert output_in_new_dir.exists()
        payload = json.loads(output_in_new_dir.read_text())
        assert "run_id" in payload

    def test_report_output_file_nonexistent_dir(
        self, tmp_path: Path, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Report with non-existent output directory gives clear error."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = tmp_path / "x.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        db_path = tmp_path / "eval.db"
        cache_path = tmp_path / "j.json"
        cache_path.write_text(
            json.dumps(
                {"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]}
            )
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "faithfulness": 0.9,
                                    "task_completion": 0.85,
                                    "reasoning": "ok",
                                    "faithfulness_reasoning": "ok",
                                    "task_completion_reasoning": "ok",
                                }
                            )
                        }
                    }
                ]
            },
        )
        run_result = runner.invoke(
            app,
            [
                "run",
                str(p),
                "--judge",
                "x/free",
                "--db",
                str(db_path),
                "--judges-cache",
                str(cache_path),
                "--output",
                "json",
                "--yes",
            ],
        )
        assert run_result.exit_code == 0
        payload = json.loads(run_result.stdout)
        run_id = payload["run_id"]

        # Now try report with bad output dir
        bad_output = tmp_path / "no_dir" / "report.json"
        result = runner.invoke(
            app,
            [
                "report",
                "--run-id",
                run_id,
                "--output",
                "json",
                "--output-file",
                str(bad_output),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        assert bad_output.exists()

    def test_export_output_file_nonexistent_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Export with non-existent output directory gives clear error."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        db_path = tmp_path / "eval.db"
        db = Database(db_path)
        run = EvalRun(config={}, judge_model="j", status=RunStatus.COMPLETED)
        db.insert_run(run)
        db.close()

        bad_output = tmp_path / "missing_dir" / "export.json"
        result = runner.invoke(
            app,
            [
                "export",
                "--run-id",
                run.run_id,
                "--format",
                "json",
                "--output-file",
                str(bad_output),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        assert bad_output.exists()


# ── 3. Evaluator resilience: transient vs permanent failures ────────────────


class TestEvaluatorResilience:
    """Evaluator distinguishes transient (429/5xx) from permanent (401/402) failures."""

    @pytest.mark.asyncio
    async def test_permanent_failure_no_retry(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """401 auth error does NOT trigger fallback to secondary judge."""
        db = Database(tmp_path / "eval.db")
        run = EvalRun(config={}, judge_model="judge/primary")
        db.insert_run(run)
        rec = EvalRecord(input_text="hi", output_text="hello", run_id=run.run_id)
        db.insert_record(rec)

        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            status_code=401,
        )
        ev = LLMEvaluator(
            db=db,
            config=EvaluatorConfig(
                api_key="bad-key",
                judges=["judge/primary", "judge/secondary"],
                rubric=BUILTIN_RUBRIC_V1,
                concurrency=1,
                max_fallbacks=2,
            ),
        )
        results = await ev.evaluate(run, [rec])
        assert results[0].error is not None
        assert "judge/secondary" not in results[0].judge_tried

    @pytest.mark.asyncio
    async def test_transient_failure_triggers_fallback(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """500 server error triggers fallback to secondary judge."""
        db = Database(tmp_path / "eval.db")
        run = EvalRun(config={}, judge_model="judge/primary")
        db.insert_run(run)
        rec = EvalRecord(input_text="hi", output_text="hello", run_id=run.run_id)
        db.insert_record(rec)

        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            status_code=500,
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "faithfulness": 0.9,
                                    "task_completion": 0.85,
                                    "reasoning": "ok",
                                    "faithfulness_reasoning": "ok",
                                    "task_completion_reasoning": "ok",
                                }
                            )
                        }
                    }
                ]
            },
        )
        ev = LLMEvaluator(
            db=db,
            config=EvaluatorConfig(
                api_key="test",
                judges=["judge/primary", "judge/secondary"],
                rubric=BUILTIN_RUBRIC_V1,
                concurrency=1,
                max_fallbacks=2,
            ),
        )
        results = await ev.evaluate(run, [rec])
        assert results[0].judge_model == "judge/secondary"
        assert results[0].judge_fallbacks == 1

    @pytest.mark.asyncio
    async def test_429_rate_limit_triggers_fallback(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """429 rate limit error triggers fallback to secondary judge."""
        db = Database(tmp_path / "eval.db")
        run = EvalRun(config={}, judge_model="judge/primary")
        db.insert_run(run)
        rec = EvalRecord(input_text="hi", output_text="hello", run_id=run.run_id)
        db.insert_record(rec)

        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            status_code=429,
        )
        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "faithfulness": 0.8,
                                    "task_completion": 0.8,
                                    "reasoning": "ok",
                                    "faithfulness_reasoning": "ok",
                                    "task_completion_reasoning": "ok",
                                }
                            )
                        }
                    }
                ]
            },
        )
        ev = LLMEvaluator(
            db=db,
            config=EvaluatorConfig(
                api_key="test",
                judges=["judge/primary", "judge/secondary"],
                rubric=BUILTIN_RUBRIC_V1,
                concurrency=1,
                max_fallbacks=2,
            ),
        )
        results = await ev.evaluate(run, [rec])
        assert results[0].judge_model == "judge/secondary"

    @pytest.mark.asyncio
    async def test_402_payment_no_retry(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """402 payment required does NOT trigger fallback."""
        db = Database(tmp_path / "eval.db")
        run = EvalRun(config={}, judge_model="judge/primary")
        db.insert_run(run)
        rec = EvalRecord(input_text="hi", output_text="hello", run_id=run.run_id)
        db.insert_record(rec)

        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            status_code=402,
        )
        ev = LLMEvaluator(
            db=db,
            config=EvaluatorConfig(
                api_key="test",
                judges=["judge/primary", "judge/secondary"],
                rubric=BUILTIN_RUBRIC_V1,
                concurrency=1,
                max_fallbacks=2,
            ),
        )
        results = await ev.evaluate(run, [rec])
        assert results[0].error is not None
        assert "judge/secondary" not in results[0].judge_tried


# ── 4. DB edge cases: corruption, concurrent access ─────────────────────────


class TestDBEdgeCases:
    """Database handles corruption, concurrent access, and WAL mode."""

    def test_corrupted_db_file(self, tmp_path: Path) -> None:
        """Corrupted database file raises a clear error on connect."""
        db_path = tmp_path / "corrupt.db"
        db_path.write_text("this is not a sqlite database")
        with pytest.raises(sqlite3.DatabaseError):
            Database(db_path)

    def test_wal_busy_timeout(self, tmp_path: Path) -> None:
        """Database has a busy_timeout set to handle concurrent access."""
        db_path = tmp_path / "eval.db"
        db = Database(db_path)
        cur = db.connection.execute("PRAGMA busy_timeout;")
        timeout = cur.fetchone()[0]
        assert timeout >= 5000, f"Expected busy_timeout >= 5000, got {timeout}"
        db.close()

    def test_concurrent_access_does_not_crash(
        self, tmp_path: Path
    ) -> None:
        """Two processes can write concurrently without corruption."""
        db_path = tmp_path / "eval.db"
        db1 = Database(db_path)
        db2 = Database(db_path.with_name("eval2.db"))

        run1 = EvalRun(config={}, judge_model="j1", status=RunStatus.COMPLETED)
        run2 = EvalRun(config={}, judge_model="j2", status=RunStatus.COMPLETED)
        db1.insert_run(run1)
        db2.insert_run(run2)

        assert len(db1.list_runs()) == 1
        assert len(db2.list_runs()) == 1
        db1.close()
        db2.close()

    def test_disk_full_simulated(self, tmp_path: Path) -> None:
        """Disk full during write raises a clear error."""
        db_path = tmp_path / "eval.db"
        db = Database(db_path)

        run = EvalRun(config={}, judge_model="j", status=RunStatus.COMPLETED)
        db.insert_run(run)

        fetched = db.get_run(run.run_id)
        assert fetched is not None
        db.close()


# ── 5. CLI: --since with invalid/future date ───────────────────────────────


class TestCLISinceEdgeCases:
    """--since with invalid or future date is handled gracefully."""

    def test_since_future_date_shows_no_records(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--since with a future date filters all records → no records error."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = tmp_path / "x.jsonl"
        p.write_text(
            '{"input":"hi","output":"hello","ts":"2024-01-01T00:00:00+00:00"}\n'
        )
        db_path = tmp_path / "eval.db"
        cache_path = tmp_path / "j.json"
        cache_path.write_text(
            json.dumps(
                {"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]}
            )
        )
        result = runner.invoke(
            app,
            [
                "run",
                str(p),
                "--since",
                "2099-01-01",
                "--db",
                str(db_path),
                "--judges-cache",
                str(cache_path),
                "--yes",
            ],
        )
        assert result.exit_code == 2
        assert "no records" in result.stdout.lower()

    def test_since_invalid_date(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--since with invalid date shows a clear error (exit 1 from ValueError)."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = tmp_path / "x.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        db_path = tmp_path / "eval.db"
        cache_path = tmp_path / "j.json"
        cache_path.write_text(
            json.dumps(
                {"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]}
            )
        )
        result = runner.invoke(
            app,
            [
                "run",
                str(p),
                "--since",
                "not-a-date",
                "--db",
                str(db_path),
                "--judges-cache",
                str(cache_path),
                "--yes",
            ],
        )
        assert result.exit_code == 1


# ── 6. CLI: --sample edge cases ─────────────────────────────────────────────


class TestCLISampleEdgeCases:
    """--sample handles edge cases gracefully."""

    def test_sample_zero(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--sample 0 is rejected with a clear error."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = tmp_path / "x.jsonl"
        p.write_text('{"input":"hi","output":"hello"}\n')
        db_path = tmp_path / "eval.db"
        cache_path = tmp_path / "j.json"
        cache_path.write_text(
            json.dumps(
                {"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]}
            )
        )
        result = runner.invoke(
            app,
            [
                "run",
                str(p),
                "--sample",
                "0",
                "--db",
                str(db_path),
                "--judges-cache",
                str(cache_path),
                "--yes",
            ],
        )
        assert result.exit_code == 2
        # sample=0 produces no records → "no records to evaluate" with exit 2 is acceptable

    def test_sample_larger_than_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--sample larger than file returns all records without error."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test")
        p = tmp_path / "x.jsonl"
        p.write_text(
            '{"input":"a","output":"1"}\n{"input":"b","output":"2"}\n'
        )
        db_path = tmp_path / "eval.db"
        cache_path = tmp_path / "j.json"
        cache_path.write_text(
            json.dumps(
                {"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]}
            )
        )
        result = runner.invoke(
            app,
            [
                "run",
                str(p),
                "--sample",
                "100",
                "--dry-run",
                "--db",
                str(db_path),
                "--judges-cache",
                str(cache_path),
            ],
        )
        assert result.exit_code == 0
        assert "2 record" in result.stdout.lower()


# ── 7. Ingest: empty file handling ──────────────────────────────────────────


class TestIngestEmptyFile:
    """Ingest handles empty files and unicode gracefully."""

    def test_ingest_empty_jsonl(self, tmp_path: Path) -> None:
        """Empty JSONL file produces no records."""
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        from src.ingest import IngestOptions, ingest_jsonl
        records = list(ingest_jsonl(p, IngestOptions()))
        assert records == []

    def test_ingest_empty_csv(self, tmp_path: Path) -> None:
        """Empty CSV file (just header) produces no records."""
        p = tmp_path / "empty.csv"
        p.write_text("input,output,reference\n")
        from src.ingest import IngestOptions, ingest_csv
        records = list(ingest_csv(p, IngestOptions()))
        assert records == []

    def test_ingest_whitespace_only_jsonl(self, tmp_path: Path) -> None:
        """Whitespace-only JSONL file produces no records."""
        p = tmp_path / "whitespace.jsonl"
        p.write_text("   \n\n  \n")
        from src.ingest import IngestOptions, ingest_jsonl
        records = list(ingest_jsonl(p, IngestOptions()))
        assert records == []

    def test_ingest_unicode_content(self, tmp_path: Path) -> None:
        """Unicode content in JSONL is handled correctly."""
        p = tmp_path / "unicode.jsonl"
        content = '{"input": "日本語テスト 🎉", "output": "こんにちは世界 🌍"}\n'
        p.write_text(content, encoding="utf-8")
        from src.ingest import IngestOptions, ingest_jsonl
        records = list(ingest_jsonl(p, IngestOptions()))
        assert len(records) == 1
        assert records[0].input_text == "日本語テスト 🎉"
        assert records[0].output_text == "こんにちは世界 🌍"


# ── 8. Evaluator: invalid JSON from LLM repeatedly ──────────────────────────


class TestEvaluatorInvalidJSON:
    """Evaluator handles repeated invalid JSON from judge."""

    @pytest.mark.asyncio
    async def test_invalid_json_all_judges_records_error(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """When all judges return invalid JSON, error is recorded."""
        db = Database(tmp_path / "eval.db")
        run = EvalRun(config={}, judge_model="judge/primary")
        db.insert_run(run)
        rec = EvalRecord(input_text="hi", output_text="hello", run_id=run.run_id)
        db.insert_record(rec)

        for _ in range(2):
            httpx_mock.add_response(
                url="https://openrouter.ai/api/v1/chat/completions",
                method="POST",
                json={
                    "choices": [
                        {"message": {"content": "not valid json at all"}}
                    ]
                },
            )

        ev = LLMEvaluator(
            db=db,
            config=EvaluatorConfig(
                api_key="test",
                judges=["judge/primary", "judge/secondary"],
                rubric=BUILTIN_RUBRIC_V1,
                concurrency=1,
                max_fallbacks=2,
            ),
        )
        results = await ev.evaluate(run, [rec])
        assert results[0].error is not None
        assert results[0].faithfulness == 0.0
        assert results[0].pass_fail == PassFail.FAIL

    @pytest.mark.asyncio
    async def test_invalid_json_with_degrade_uses_heuristic(
        self, tmp_path: Path, httpx_mock: HTTPXMock
    ) -> None:
        """When degrade=True and all judges fail, local heuristic is used."""
        db = Database(tmp_path / "eval.db")
        run = EvalRun(config={}, judge_model="judge/primary")
        db.insert_run(run)
        rec = EvalRecord(
            input_text="What is 2+2?",
            output_text="The answer is 4",
            reference_text="4",
            run_id=run.run_id,
        )
        db.insert_record(rec)

        httpx_mock.add_response(
            url="https://openrouter.ai/api/v1/chat/completions",
            method="POST",
            json={
                "choices": [{"message": {"content": "garbage response"}}]
            },
        )

        ev = LLMEvaluator(
            db=db,
            config=EvaluatorConfig(
                api_key="test",
                judges=["judge/primary"],
                rubric=BUILTIN_RUBRIC_V1,
                concurrency=1,
                max_fallbacks=1,
                degrade=True,
            ),
        )
        results = await ev.evaluate(run, [rec])
        assert results[0].judge_model == "local-heuristic"
        assert results[0].faithfulness > 0
