"""Tests for src/db.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.db import CURRENT_SCHEMA_VERSION, Database
from src.models import EvalRecord, EvalResult, EvalRun, JudgeCacheEntry, PassFail


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "eval.db")


def test_init_creates_schema(db: Database) -> None:
    assert db.get_schema_version() == CURRENT_SCHEMA_VERSION


def test_wal_mode_enabled(db: Database) -> None:
    cur = db.connection.execute("PRAGMA journal_mode;")
    mode = cur.fetchone()[0]
    assert mode.lower() == "wal"


def test_foreign_keys_enabled(db: Database) -> None:
    cur = db.connection.execute("PRAGMA foreign_keys;")
    assert cur.fetchone()[0] == 1


def test_insert_and_get_run(db: Database) -> None:
    run = EvalRun(config={"x": 1}, judge_model="m")
    db.insert_run(run)
    fetched = db.get_run(run.run_id)
    assert fetched is not None
    assert fetched.run_id == run.run_id
    assert fetched.config == {"x": 1}


def test_update_run(db: Database) -> None:
    run = EvalRun(config={})
    db.insert_run(run)
    run.record_count = 5
    run.mean_score = 0.9
    db.update_run(run)
    again = db.get_run(run.run_id)
    assert again is not None
    assert again.record_count == 5
    assert again.mean_score == 0.9


def test_insert_record_and_get_records(db: Database) -> None:
    run = EvalRun(config={})
    db.insert_run(run)
    rec = EvalRecord(input_text="i", output_text="o", run_id=run.run_id)
    db.insert_record(rec)
    records = db.get_records(run.run_id)
    assert len(records) == 1
    assert records[0].input_text == "i"


def test_insert_result(db: Database) -> None:
    run = EvalRun(config={})
    db.insert_run(run)
    rec = EvalRecord(input_text="i", output_text="o", run_id=run.run_id)
    db.insert_record(rec)
    res = EvalResult(
        record_id=rec.record_id,
        run_id=run.run_id,
        faithfulness=0.8,
        task_completion=0.8,
        combined_score=0.8,
        pass_fail=PassFail.PASS,
        judge_model="m1",
        judge_tried=["m1"],
    )
    db.insert_result(res)
    results = db.get_results(run.run_id)
    assert len(results) == 1
    assert results[0].judge_model == "m1"


def test_cache_roundtrip(db: Database) -> None:
    entry = JudgeCacheEntry(
        cache_key="k1",
        model_id="m",
        rubric_version="1.0",
        response={"faithfulness": 0.5, "task_completion": 0.5},
    )
    db.put_cache(entry)
    got = db.get_cache("k1")
    assert got is not None
    assert got.response["faithfulness"] == 0.5


def test_cache_hit_increments(db: Database) -> None:
    entry = JudgeCacheEntry(
        cache_key="k1",
        model_id="m",
        rubric_version="1.0",
        response={"x": 1},
    )
    db.put_cache(entry)
    db.touch_cache("k1")
    db.touch_cache("k1")
    again = db.get_cache("k1")
    assert again is not None
    assert again.hits >= 3


def test_clear_cache(db: Database) -> None:
    db.put_cache(JudgeCacheEntry(cache_key="a", model_id="m", rubric_version="1.0", response={}))
    db.put_cache(JudgeCacheEntry(cache_key="b", model_id="m", rubric_version="1.0", response={}))
    n = db.clear_cache()
    assert n == 2
    assert db.cache_stats()["count"] == 0


def test_cache_stats(db: Database) -> None:
    db.put_cache(JudgeCacheEntry(cache_key="a", model_id="m", rubric_version="1.0", response={}))
    stats = db.cache_stats()
    assert stats["count"] == 1
    assert "hits" in stats


def test_list_runs(db: Database) -> None:
    for _ in range(3):
        db.insert_run(EvalRun(config={}))
    runs = db.list_runs()
    assert len(runs) == 3


def test_export_json(db: Database, tmp_path: Path) -> None:
    run = EvalRun(config={})
    db.insert_run(run)
    rec = EvalRecord(input_text="i", output_text="o", run_id=run.run_id)
    db.insert_record(rec)
    res = EvalResult(
        record_id=rec.record_id,
        run_id=run.run_id,
        faithfulness=0.9,
        task_completion=0.7,
        combined_score=0.8,
        pass_fail=PassFail.PASS,
        judge_model="m",
    )
    db.insert_result(res)
    out = tmp_path / "x.json"
    db.export_run(run.run_id, out, fmt="json")
    data = json.loads(out.read_text())
    assert data["run"]["run_id"] == run.run_id
    assert len(data["records"]) == 1
    assert len(data["results"]) == 1


def test_export_csv(db: Database, tmp_path: Path) -> None:
    run = EvalRun(config={})
    db.insert_run(run)
    rec = EvalRecord(input_text="i", output_text="o", run_id=run.run_id)
    db.insert_record(rec)
    res = EvalResult(
        record_id=rec.record_id,
        run_id=run.run_id,
        faithfulness=0.9,
        task_completion=0.7,
        combined_score=0.8,
        pass_fail=PassFail.PASS,
        judge_model="m",
    )
    db.insert_result(res)
    out = tmp_path / "x.csv"
    db.export_run(run.run_id, out, fmt="csv")
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == 1
    assert rows[0]["judge_model"] == "m"


def test_foreign_key_violation(db: Database) -> None:
    rec = EvalRecord(input_text="i", output_text="o", run_id="missing")
    with pytest.raises(Exception):
        db.insert_record(rec)


def test_migration_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "db.sqlite"
    Database(p).close()
    db = Database(p)
    assert db.get_schema_version() == CURRENT_SCHEMA_VERSION
    db.close()


def test_result_record_index_exists(db: Database) -> None:
    cur = db.connection.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_results_record';"
    )
    assert cur.fetchone() is not None


def test_get_result_for_record(db: Database) -> None:
    run = EvalRun(config={})
    db.insert_run(run)
    rec = EvalRecord(input_text="i", output_text="o", run_id=run.run_id)
    db.insert_record(rec)
    assert db.get_result_for_record(rec.record_id) is None
    res = EvalResult(
        record_id=rec.record_id,
        run_id=run.run_id,
        faithfulness=0.9,
        task_completion=0.7,
        combined_score=0.8,
        pass_fail=PassFail.PASS,
        judge_model="m",
    )
    db.insert_result(res)
    fetched = db.get_result_for_record(rec.record_id)
    assert fetched is not None
    assert fetched.judge_model == "m"


def test_export_unsupported_format(db: Database, tmp_path: Path) -> None:
    run = EvalRun(config={})
    db.insert_run(run)
    out = tmp_path / "x.txt"
    with pytest.raises(ValueError):
        db.export_run(run.run_id, out, fmt="xml")


def test_export_missing_run_raises(db: Database, tmp_path: Path) -> None:
    out = tmp_path / "x.json"
    with pytest.raises(ValueError):
        db.export_run("missing", out, fmt="json")
