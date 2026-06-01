"""SQLite persistence layer for eval-harness.

Implements WAL-mode connections, schema versioning, idempotent migrations,
and CRUD helpers for runs, records, results, and the judge cache.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.models import (
    EvalRecord,
    EvalResult,
    EvalRun,
    JudgeCacheEntry,
    PassFail,
    RunStatus,
)

CURRENT_SCHEMA_VERSION = 1


_SCHEMA_V1 = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at TEXT NOT NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_runs (
        run_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        config_json TEXT NOT NULL,
        record_count INTEGER DEFAULT 0,
        rubric_id TEXT DEFAULT 'faithfulness-v1',
        judge_model TEXT,
        status TEXT DEFAULT 'running',
        completed_at TEXT,
        mean_score REAL,
        pass_rate REAL,
        eval_time_seconds REAL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS eval_records (
        record_id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL REFERENCES eval_runs(run_id),
        input_text TEXT NOT NULL,
        output_text TEXT NOT NULL,
        reference_text TEXT,
        source_file TEXT,
        metadata_json TEXT,
        created_at TEXT NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_records_run ON eval_records(run_id);",
    """
    CREATE TABLE IF NOT EXISTS eval_results (
        result_id TEXT PRIMARY KEY,
        record_id TEXT NOT NULL REFERENCES eval_records(record_id),
        run_id TEXT NOT NULL REFERENCES eval_runs(run_id),
        rubric_id TEXT DEFAULT 'faithfulness-v1',
        rubric_version TEXT DEFAULT '1.0',
        faithfulness REAL NOT NULL,
        task_completion REAL NOT NULL,
        combined_score REAL NOT NULL,
        pass_fail TEXT NOT NULL,
        reasoning TEXT DEFAULT '',
        faithfulness_reasoning TEXT DEFAULT '',
        task_completion_reasoning TEXT DEFAULT '',
        judge_model TEXT NOT NULL,
        judge_fallbacks INTEGER DEFAULT 0,
        judge_tried TEXT DEFAULT '[]',
        tokens_estimated INTEGER,
        evaluated_at TEXT NOT NULL,
        error TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_results_run ON eval_results(run_id);",
    "CREATE INDEX IF NOT EXISTS idx_results_record ON eval_results(record_id);",
    """
    CREATE TABLE IF NOT EXISTS judge_cache (
        cache_key TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        rubric_version TEXT NOT NULL,
        response_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        hits INTEGER DEFAULT 1
    );
    """,
]


def _iso(dt: datetime | None) -> str | None:
    """Return ISO-8601 string for datetime, or None."""
    return dt.isoformat() if dt else None


def _parse_iso(s: str | None) -> datetime | None:
    """Parse ISO-8601 string back to datetime; return None for empty input."""
    return datetime.fromisoformat(s) if s else None


class Database:
    """SQLite wrapper exposing typed CRUD for eval-harness models."""

    def __init__(self, path: str | Path) -> None:
        """Open or create a database at ``path`` and run pending migrations.

        Args:
            path: Filesystem path to the sqlite database file.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL;")
        self.connection.execute("PRAGMA foreign_keys=ON;")
        self._migrate()

    def close(self) -> None:
        """Close the underlying sqlite connection."""
        self.connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _migrate(self) -> None:
        """Apply all pending schema migrations idempotently."""
        cur = self.connection.cursor()
        for stmt in _SCHEMA_V1:
            cur.execute(stmt)
        cur.execute("SELECT MAX(version) FROM schema_version;")
        row = cur.fetchone()
        current = row[0] if row and row[0] is not None else 0
        if current < CURRENT_SCHEMA_VERSION:
            cur.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, ?);",
                (CURRENT_SCHEMA_VERSION, datetime.now(UTC).isoformat()),
            )
        self.connection.commit()

    def get_schema_version(self) -> int:
        """Return the current schema version stored in the database."""
        cur = self.connection.execute("SELECT MAX(version) FROM schema_version;")
        v = cur.fetchone()[0]
        return v or 0

    def insert_run(self, run: EvalRun) -> None:
        """Insert a new evaluation run row."""
        self.connection.execute(
            """
            INSERT INTO eval_runs (
                run_id, created_at, config_json, record_count, rubric_id,
                judge_model, status, completed_at, mean_score, pass_rate,
                eval_time_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                run.run_id,
                _iso(run.created_at),
                json.dumps(run.config),
                run.record_count,
                run.rubric_id,
                run.judge_model,
                run.status.value,
                _iso(run.completed_at),
                run.mean_score,
                run.pass_rate,
                run.eval_time_seconds,
            ),
        )
        self.connection.commit()

    def update_run(self, run: EvalRun) -> None:
        """Update mutable fields of a run."""
        self.connection.execute(
            """
            UPDATE eval_runs SET
                record_count=?, judge_model=?, status=?, completed_at=?,
                mean_score=?, pass_rate=?, eval_time_seconds=?, config_json=?
            WHERE run_id=?;
            """,
            (
                run.record_count,
                run.judge_model,
                run.status.value,
                _iso(run.completed_at),
                run.mean_score,
                run.pass_rate,
                run.eval_time_seconds,
                json.dumps(run.config),
                run.run_id,
            ),
        )
        self.connection.commit()

    def get_run(self, run_id: str) -> EvalRun | None:
        """Return the run with ``run_id`` or None."""
        cur = self.connection.execute("SELECT * FROM eval_runs WHERE run_id=?;", (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        return EvalRun(
            run_id=row["run_id"],
            created_at=_parse_iso(row["created_at"]),
            config=json.loads(row["config_json"]),
            record_count=row["record_count"] or 0,
            rubric_id=row["rubric_id"] or "faithfulness-v1",
            judge_model=row["judge_model"],
            status=RunStatus(row["status"]),
            completed_at=_parse_iso(row["completed_at"]),
            mean_score=row["mean_score"],
            pass_rate=row["pass_rate"],
            eval_time_seconds=row["eval_time_seconds"],
        )

    def list_runs(self, limit: int = 100) -> list[EvalRun]:
        """Return up to ``limit`` runs, newest first."""
        cur = self.connection.execute(
            "SELECT run_id FROM eval_runs ORDER BY created_at DESC LIMIT ?;",
            (limit,),
        )
        ids = [r[0] for r in cur.fetchall()]
        out: list[EvalRun] = []
        for rid in ids:
            run = self.get_run(rid)
            if run is not None:
                out.append(run)
        return out

    def insert_record(self, record: EvalRecord) -> None:
        """Insert a single eval record."""
        if not record.run_id:
            raise ValueError("record.run_id is required")
        self.connection.execute(
            """
            INSERT INTO eval_records (
                record_id, run_id, input_text, output_text, reference_text,
                source_file, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                record.record_id,
                record.run_id,
                record.input_text,
                record.output_text,
                record.reference_text,
                record.source_file,
                json.dumps(record.metadata),
                _iso(record.created_at),
            ),
        )
        self.connection.commit()

    def get_records(self, run_id: str) -> list[EvalRecord]:
        """Return all records for a run in insertion order."""
        cur = self.connection.execute(
            "SELECT * FROM eval_records WHERE run_id=? ORDER BY created_at;",
            (run_id,),
        )
        out: list[EvalRecord] = []
        for row in cur.fetchall():
            out.append(
                EvalRecord(
                    record_id=row["record_id"],
                    run_id=row["run_id"],
                    input_text=row["input_text"],
                    output_text=row["output_text"],
                    reference_text=row["reference_text"],
                    source_file=row["source_file"],
                    metadata=json.loads(row["metadata_json"] or "{}"),
                    created_at=_parse_iso(row["created_at"]),
                )
            )
        return out

    def insert_result(self, result: EvalResult) -> None:
        """Insert a judge evaluation result."""
        self.connection.execute(
            """
            INSERT INTO eval_results (
                result_id, record_id, run_id, rubric_id, rubric_version,
                faithfulness, task_completion, combined_score, pass_fail,
                reasoning, faithfulness_reasoning, task_completion_reasoning,
                judge_model, judge_fallbacks, judge_tried, tokens_estimated,
                evaluated_at, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                result.result_id,
                result.record_id,
                result.run_id,
                result.rubric_id,
                result.rubric_version,
                result.faithfulness,
                result.task_completion,
                result.combined_score,
                result.pass_fail.value,
                result.reasoning,
                result.faithfulness_reasoning,
                result.task_completion_reasoning,
                result.judge_model,
                result.judge_fallbacks,
                json.dumps(result.judge_tried),
                result.tokens_estimated,
                _iso(result.evaluated_at),
                result.error,
            ),
        )
        self.connection.commit()

    def get_results(self, run_id: str) -> list[EvalResult]:
        """Return all results for a run in insertion order."""
        cur = self.connection.execute(
            "SELECT * FROM eval_results WHERE run_id=? ORDER BY evaluated_at;",
            (run_id,),
        )
        out: list[EvalResult] = []
        for row in cur.fetchall():
            out.append(
                EvalResult(
                    result_id=row["result_id"],
                    record_id=row["record_id"],
                    run_id=row["run_id"],
                    rubric_id=row["rubric_id"] or "faithfulness-v1",
                    rubric_version=row["rubric_version"] or "1.0",
                    faithfulness=row["faithfulness"],
                    task_completion=row["task_completion"],
                    combined_score=row["combined_score"],
                    pass_fail=PassFail(row["pass_fail"]),
                    reasoning=row["reasoning"] or "",
                    faithfulness_reasoning=row["faithfulness_reasoning"] or "",
                    task_completion_reasoning=row["task_completion_reasoning"] or "",
                    judge_model=row["judge_model"],
                    judge_fallbacks=row["judge_fallbacks"] or 0,
                    judge_tried=json.loads(row["judge_tried"] or "[]"),
                    tokens_estimated=row["tokens_estimated"],
                    evaluated_at=_parse_iso(row["evaluated_at"]),
                    error=row["error"],
                )
            )
        return out

    def get_result_for_record(self, record_id: str) -> EvalResult | None:
        """Return the first result associated with a record (or None)."""
        cur = self.connection.execute(
            "SELECT * FROM eval_results WHERE record_id=? LIMIT 1;", (record_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        return EvalResult(
            result_id=row["result_id"],
            record_id=row["record_id"],
            run_id=row["run_id"],
            rubric_id=row["rubric_id"] or "faithfulness-v1",
            rubric_version=row["rubric_version"] or "1.0",
            faithfulness=row["faithfulness"],
            task_completion=row["task_completion"],
            combined_score=row["combined_score"],
            pass_fail=PassFail(row["pass_fail"]),
            reasoning=row["reasoning"] or "",
            faithfulness_reasoning=row["faithfulness_reasoning"] or "",
            task_completion_reasoning=row["task_completion_reasoning"] or "",
            judge_model=row["judge_model"],
            judge_fallbacks=row["judge_fallbacks"] or 0,
            judge_tried=json.loads(row["judge_tried"] or "[]"),
            tokens_estimated=row["tokens_estimated"],
            evaluated_at=_parse_iso(row["evaluated_at"]),
            error=row["error"],
        )

    def put_cache(self, entry: JudgeCacheEntry) -> None:
        """Upsert a judge cache entry."""
        self.connection.execute(
            """
            INSERT INTO judge_cache (
                cache_key, model_id, rubric_version, response_json, created_at, hits
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                response_json=excluded.response_json,
                hits=judge_cache.hits + 1;
            """,
            (
                entry.cache_key,
                entry.model_id,
                entry.rubric_version,
                json.dumps(entry.response),
                _iso(entry.created_at),
                entry.hits,
            ),
        )
        self.connection.commit()

    def get_cache(self, key: str) -> JudgeCacheEntry | None:
        """Return cache entry by key or None."""
        cur = self.connection.execute("SELECT * FROM judge_cache WHERE cache_key=?;", (key,))
        row = cur.fetchone()
        if not row:
            return None
        return JudgeCacheEntry(
            cache_key=row["cache_key"],
            model_id=row["model_id"],
            rubric_version=row["rubric_version"],
            response=json.loads(row["response_json"]),
            created_at=_parse_iso(row["created_at"]),
            hits=row["hits"] or 1,
        )

    def touch_cache(self, key: str) -> None:
        """Increment the hit counter for a cache entry."""
        self.connection.execute("UPDATE judge_cache SET hits = hits + 1 WHERE cache_key=?;", (key,))
        self.connection.commit()

    def clear_cache(self) -> int:
        """Delete all cache entries; return number removed."""
        cur = self.connection.execute("DELETE FROM judge_cache;")
        self.connection.commit()
        return cur.rowcount

    def cache_stats(self) -> dict[str, Any]:
        """Return aggregate cache statistics."""
        cur = self.connection.execute(
            "SELECT COUNT(*) AS c, COALESCE(SUM(hits),0) AS h FROM judge_cache;"
        )
        row = cur.fetchone()
        return {"count": row["c"], "hits": row["h"]}

    def export_run(self, run_id: str, out_path: str | Path, fmt: str = "json") -> Path:
        """Export a run's results (and records) to JSON or CSV.

        Args:
            run_id: The run identifier to export.
            out_path: Destination file path.
            fmt: 'json' or 'csv'.

        Returns:
            The output path written.
        """
        out_path = Path(out_path)
        run = self.get_run(run_id)
        records = self.get_records(run_id)
        results = self.get_results(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")

        if fmt == "json":
            payload = {
                "run": run.model_dump(mode="json"),
                "records": [r.model_dump(mode="json") for r in records],
                "results": [r.model_dump(mode="json") for r in results],
            }
            out_path.write_text(json.dumps(payload, indent=2, default=str))
        elif fmt == "csv":
            recs_by_id = {r.record_id: r for r in records}
            fields = [
                "result_id",
                "record_id",
                "run_id",
                "input_text",
                "output_text",
                "reference_text",
                "faithfulness",
                "task_completion",
                "combined_score",
                "pass_fail",
                "judge_model",
                "judge_fallbacks",
                "evaluated_at",
                "error",
            ]
            with out_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for res in results:
                    rec = recs_by_id.get(res.record_id)
                    w.writerow(
                        {
                            "result_id": res.result_id,
                            "record_id": res.record_id,
                            "run_id": res.run_id,
                            "input_text": rec.input_text if rec else "",
                            "output_text": rec.output_text if rec else "",
                            "reference_text": (rec.reference_text or "") if rec else "",
                            "faithfulness": res.faithfulness,
                            "task_completion": res.task_completion,
                            "combined_score": res.combined_score,
                            "pass_fail": res.pass_fail.value,
                            "judge_model": res.judge_model,
                            "judge_fallbacks": res.judge_fallbacks,
                            "evaluated_at": _iso(res.evaluated_at),
                            "error": res.error or "",
                        }
                    )
        else:
            raise ValueError(f"unsupported export format: {fmt}")
        return out_path
