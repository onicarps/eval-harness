"""Tests for agent-report and agent-export CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from src.cli import app

runner = CliRunner()


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_agent_run(db_path: Path, suite_id: str = "echo-v1") -> tuple[str, str]:
    """Create an agent run with results in the DB. Returns (run_id, suite_id)."""
    from src.agent_models import AgentRun, AgentResult, AgentStatus
    from src.db import Database

    db = Database(db_path)
    run = AgentRun(
        suite_id=suite_id,
        agent_type="python",
        status=AgentStatus.COMPLETED,
        config={"agent_name": "test-agent"},
    )
    db.insert_agent_run(run)

    results = [
        AgentResult(step_id="step-1", agent_output="hello", success=True, score=0.9),
        AgentResult(step_id="step-2", agent_output="world", success=True, score=0.8),
        AgentResult(step_id="step-3", agent_output="", success=False, score=0.2, error="timeout"),
    ]
    for r in results:
        db.insert_agent_result(run.run_id, r)

    db.close()
    return run.run_id, suite_id


def _make_agent_run_multi(db_path: Path, suite_id: str = "echo-v1", num_steps: int = 5) -> str:
    """Create an agent run with multiple results. Returns run_id."""
    from src.agent_models import AgentRun, AgentResult, AgentStatus
    from src.db import Database

    db = Database(db_path)
    run = AgentRun(
        suite_id=suite_id,
        agent_type="subprocess",
        status=AgentStatus.COMPLETED,
        config={"agent_name": "multi-step-agent"},
    )
    db.insert_agent_run(run)

    for i in range(num_steps):
        r = AgentResult(
            step_id=f"step-{i + 1}",
            agent_output=f"output-{i + 1}",
            success=i % 2 == 0,
            score=0.9 - (i * 0.1),
        )
        db.insert_agent_result(run.run_id, r)

    db.close()
    return run.run_id


# ── agent-report tests ──────────────────────────────────────────────────────


class TestAgentReportCommand:
    """Tests for the 'agent-report' CLI command."""

    def test_agent_report_help(self) -> None:
        """Verify agent-report appears in help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "agent-report" in result.stdout

    def test_agent_report_unknown_run(self, tmp_path: Path) -> None:
        """agent-report with non-existent run-id should fail."""
        db_path = tmp_path / "eval.db"
        result = runner.invoke(
            app,
            ["agent-report", "--run-id", "nonexistent", "--db", str(db_path)],
        )
        assert result.exit_code != 0

    def test_agent_report_table_output(self, tmp_path: Path) -> None:
        """agent-report with default table output shows summary and steps."""
        db_path = tmp_path / "eval.db"
        run_id, _ = _make_agent_run(db_path)

        result = runner.invoke(
            app,
            ["agent-report", "--run-id", run_id, "--db", str(db_path)],
        )
        assert result.exit_code == 0
        out = result.stdout
        # Should show summary info
        assert "Agent" in out or "agent" in out
        assert run_id[:8] in out or run_id in out

    def test_agent_report_json_output(self, tmp_path: Path) -> None:
        """agent-report with --output json returns valid JSON."""
        db_path = tmp_path / "eval.db"
        run_id, _ = _make_agent_run(db_path)

        result = runner.invoke(
            app,
            [
                "agent-report",
                "--run-id",
                run_id,
                "--output",
                "json",
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert "run_id" in parsed
        assert "suite_id" in parsed
        assert "agent_type" in parsed
        assert "status" in parsed
        assert "results" in parsed
        assert len(parsed["results"]) == 3

    def test_agent_report_json_output_content(self, tmp_path: Path) -> None:
        """agent-report JSON output contains correct result data."""
        db_path = tmp_path / "eval.db"
        run_id, suite_id = _make_agent_run(db_path)

        result = runner.invoke(
            app,
            [
                "agent-report",
                "--run-id",
                run_id,
                "--output",
                "json",
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert parsed["run_id"] == run_id
        assert parsed["suite_id"] == suite_id
        assert parsed["agent_type"] == "python"
        assert parsed["status"] == "completed"
        # Check step results
        step_ids = [r["step_id"] for r in parsed["results"]]
        assert "step-1" in step_ids
        assert "step-2" in step_ids
        assert "step-3" in step_ids

    def test_agent_report_output_file(self, tmp_path: Path) -> None:
        """agent-report with --output-file writes to file."""
        db_path = tmp_path / "eval.db"
        run_id, _ = _make_agent_run(db_path)
        out_file = tmp_path / "report.txt"

        result = runner.invoke(
            app,
            [
                "agent-report",
                "--run-id",
                run_id,
                "--output-file",
                str(out_file),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        assert len(content) > 0

    def test_agent_report_json_output_file(self, tmp_path: Path) -> None:
        """agent-report with --output json --output-file writes JSON to file."""
        db_path = tmp_path / "eval.db"
        run_id, _ = _make_agent_run(db_path)
        out_file = tmp_path / "report.json"

        result = runner.invoke(
            app,
            [
                "agent-report",
                "--run-id",
                run_id,
                "--output",
                "json",
                "--output-file",
                str(out_file),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        parsed = json.loads(out_file.read_text())
        assert parsed["run_id"] == run_id

    def test_agent_report_no_results(self, tmp_path: Path) -> None:
        """agent-report for a run with no results still works."""
        from src.agent_models import AgentRun, AgentStatus
        from src.db import Database

        db_path = tmp_path / "eval.db"
        db = Database(db_path)
        run = AgentRun(
            suite_id="echo-v1",
            agent_type="python",
            status=AgentStatus.RUNNING,
        )
        db.insert_agent_run(run)
        db.close()

        result = runner.invoke(
            app,
            ["agent-report", "--run-id", run.run_id, "--db", str(db_path)],
        )
        assert result.exit_code == 0


# ── agent-export tests ──────────────────────────────────────────────────────


class TestAgentExportCommand:
    """Tests for the 'agent-export' CLI command."""

    def test_agent_export_help(self) -> None:
        """Verify agent-export appears in help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "agent-export" in result.stdout

    def test_agent_export_unknown_run(self, tmp_path: Path) -> None:
        """agent-export with non-existent run-id should fail."""
        db_path = tmp_path / "eval.db"
        out_file = tmp_path / "export.json"
        result = runner.invoke(
            app,
            [
                "agent-export",
                "--run-id",
                "nonexistent",
                "--format",
                "json",
                "--output-file",
                str(out_file),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code != 0

    def test_agent_export_json(self, tmp_path: Path) -> None:
        """agent-export to JSON produces valid file."""
        db_path = tmp_path / "eval.db"
        run_id, _ = _make_agent_run(db_path)
        out_file = tmp_path / "export.json"

        result = runner.invoke(
            app,
            [
                "agent-export",
                "--run-id",
                run_id,
                "--format",
                "json",
                "--output-file",
                str(out_file),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        parsed = json.loads(out_file.read_text())
        assert "run" in parsed
        assert "results" in parsed
        assert parsed["run"]["run_id"] == run_id
        assert len(parsed["results"]) == 3

    def test_agent_export_csv(self, tmp_path: Path) -> None:
        """agent-export to CSV produces valid file with headers."""
        db_path = tmp_path / "eval.db"
        run_id, _ = _make_agent_run(db_path)
        out_file = tmp_path / "export.csv"

        result = runner.invoke(
            app,
            [
                "agent-export",
                "--run-id",
                run_id,
                "--format",
                "csv",
                "--output-file",
                str(out_file),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
        content = out_file.read_text()
        # CSV should have headers
        assert "step_id" in content
        assert "run_id" in content
        # Should have data rows
        lines = content.strip().split("\n")
        assert len(lines) == 4  # header + 3 results

    def test_agent_export_csv_content(self, tmp_path: Path) -> None:
        """agent-export CSV contains correct data."""
        db_path = tmp_path / "eval.db"
        run_id, _ = _make_agent_run(db_path)
        out_file = tmp_path / "export.csv"

        result = runner.invoke(
            app,
            [
                "agent-export",
                "--run-id",
                run_id,
                "--format",
                "csv",
                "--output-file",
                str(out_file),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        content = out_file.read_text()
        assert run_id in content
        assert "step-1" in content
        assert "python" in content  # agent_type

    def test_agent_export_json_structure(self, tmp_path: Path) -> None:
        """agent-export JSON has correct structure with all expected fields."""
        db_path = tmp_path / "eval.db"
        run_id, suite_id = _make_agent_run(db_path)
        out_file = tmp_path / "export.json"

        result = runner.invoke(
            app,
            [
                "agent-export",
                "--run-id",
                run_id,
                "--format",
                "json",
                "--output-file",
                str(out_file),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(out_file.read_text())

        # Check run fields
        run = parsed["run"]
        assert run["run_id"] == run_id
        assert run["suite_id"] == suite_id
        assert run["agent_type"] == "python"
        assert run["status"] == "completed"

        # Check result fields
        for r in parsed["results"]:
            assert "step_id" in r
            assert "agent_output" in r
            assert "success" in r
            assert "score" in r

    def test_agent_export_multi_step(self, tmp_path: Path) -> None:
        """agent-export handles multi-step runs correctly."""
        db_path = tmp_path / "eval.db"
        run_id = _make_agent_run_multi(db_path, num_steps=5)
        out_file = tmp_path / "export.json"

        result = runner.invoke(
            app,
            [
                "agent-export",
                "--run-id",
                run_id,
                "--format",
                "json",
                "--output-file",
                str(out_file),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(out_file.read_text())
        assert len(parsed["results"]) == 5

    def test_agent_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        """agent-export creates parent directories if needed."""
        db_path = tmp_path / "eval.db"
        run_id, _ = _make_agent_run(db_path)
        out_file = tmp_path / "deep" / "nested" / "dir" / "export.json"

        result = runner.invoke(
            app,
            [
                "agent-export",
                "--run-id",
                run_id,
                "--format",
                "json",
                "--output-file",
                str(out_file),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()

    def test_agent_export_no_results(self, tmp_path: Path) -> None:
        """agent-export for a run with no results still produces valid output."""
        from src.agent_models import AgentRun, AgentStatus
        from src.db import Database

        db_path = tmp_path / "eval.db"
        db = Database(db_path)
        run = AgentRun(
            suite_id="echo-v1",
            agent_type="python",
            status=AgentStatus.RUNNING,
        )
        db.insert_agent_run(run)
        db.close()

        out_file = tmp_path / "export.json"
        result = runner.invoke(
            app,
            [
                "agent-export",
                "--run-id",
                run.run_id,
                "--format",
                "json",
                "--output-file",
                str(out_file),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(out_file.read_text())
        assert parsed["results"] == []


# ── Integration: both commands appear in help ──────────────────────────────


class TestAgentCommandsIntegration:
    """Integration tests for agent-report and agent-export."""

    def test_both_commands_in_help(self) -> None:
        """Both agent-report and agent-export appear in --help output."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "agent-report" in result.stdout
        assert "agent-export" in result.stdout

    def test_agent_report_uses_same_db_as_agent_eval(self, tmp_path: Path) -> None:
        """agent-report can read data written by the agent eval flow."""
        db_path = tmp_path / "eval.db"
        run_id, suite_id = _make_agent_run(db_path, suite_id="echo-v1")

        # The run should be retrievable by agent-report
        result = runner.invoke(
            app,
            [
                "agent-report",
                "--run-id",
                run_id,
                "--output",
                "json",
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        parsed = json.loads(result.stdout)
        assert parsed["suite_id"] == suite_id

    def test_agent_export_uses_same_db_as_agent_eval(self, tmp_path: Path) -> None:
        """agent-export can read data written by the agent eval flow."""
        db_path = tmp_path / "eval.db"
        run_id, suite_id = _make_agent_run(db_path, suite_id="echo-v1")

        out_file = tmp_path / "out.csv"
        result = runner.invoke(
            app,
            [
                "agent-export",
                "--run-id",
                run_id,
                "--format",
                "csv",
                "--output-file",
                str(out_file),
                "--db",
                str(db_path),
            ],
        )
        assert result.exit_code == 0
        assert out_file.exists()
