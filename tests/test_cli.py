"""Integration tests for the Typer CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock
from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    out = result.stdout
    for cmd in ("run", "judges", "report", "export", "cache"):
        assert cmd in out


def test_judges_command(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    cache_path = tmp_path / "j.json"
    cache_path.write_text(
        json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
    )
    result = runner.invoke(
        app,
        [
            "judges",
            "--json",
            "--judges-cache",
            str(cache_path),
            "--db",
            str(db_path),
        ],
    )
    assert result.exit_code == 0
    parsed = json.loads(result.stdout)
    assert parsed[0]["id"] == "x/free"


def test_dry_run_does_not_call_api(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text('{"input":"hi","output":"hello"}\n')
    db_path = tmp_path / "eval.db"
    cache_path = tmp_path / "j.json"
    cache_path.write_text(
        json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
    )
    result = runner.invoke(
        app,
        [
            "run",
            str(p),
            "--dry-run",
            "--db",
            str(db_path),
            "--judges-cache",
            str(cache_path),
        ],
    )
    assert result.exit_code == 0
    assert "dry-run" in result.stdout.lower() or "1 record" in result.stdout.lower()


def test_run_end_to_end(
    tmp_path: Path, httpx_mock: HTTPXMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENRIXER_API_KEY", "test")
    p = tmp_path / "input.jsonl"
    p.write_text('{"input":"hi","output":"hello"}\n')
    db_path = tmp_path / "eval.db"
    cache_path = tmp_path / "j.json"
    cache_path.write_text(
        json.dumps({"models": [{"id": "x/free", "name": "X", "context_length": 100, "free": True}]})
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
    result = runner.invoke(
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
            "--yes",
        ],
    )
    assert result.exit_code == 0
    assert "pass" in result.stdout.lower() or "summary" in result.stdout.lower()


def test_run_fails_without_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENRIXER_API_KEY", raising=False)
    p = tmp_path / "input.jsonl"
    p.write_text('{"input":"hi","output":"hello"}\n')
    db_path = tmp_path / "eval.db"
    result = runner.invoke(
        app,
        ["run", str(p), "--db", str(db_path), "--yes"],
    )
    assert result.exit_code == 2


def test_report_unknown_run(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    result = runner.invoke(app, ["report", "--run-id", "nope", "--db", str(db_path)])
    assert result.exit_code != 0


def test_export_unknown_run(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    out = tmp_path / "out.json"
    result = runner.invoke(
        app,
        [
            "export",
            "--run-id",
            "nope",
            "--format",
            "json",
            "--output-file",
            str(out),
            "--db",
            str(db_path),
        ],
    )
    assert result.exit_code != 0


def test_cache_stats(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    result = runner.invoke(app, ["cache", "--stats", "--db", str(db_path)])
    assert result.exit_code == 0


def test_cache_clear(tmp_path: Path) -> None:
    db_path = tmp_path / "eval.db"
    result = runner.invoke(app, ["cache", "--clear", "--db", str(db_path)])
    assert result.exit_code == 0
