"""Tests for DB migration v4 — agent-specific tables."""

from __future__ import annotations

from pathlib import Path

from src.db import CURRENT_SCHEMA_VERSION, Database


class TestMigrationV4:
    """Tests for the v4 schema migration (agent tables)."""

    def test_current_version_is_4(self) -> None:
        """CURRENT_SCHEMA_VERSION should be 4 after migration."""
        assert CURRENT_SCHEMA_VERSION == 4

    def test_agent_runs_table_exists(self, tmp_path: Path) -> None:
        """agent_runs table should be created in v4."""
        db = Database(tmp_path / "eval.db")
        cur = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_runs';"
        )
        assert cur.fetchone() is not None
        db.close()

    def test_agent_results_table_exists(self, tmp_path: Path) -> None:
        """agent_results table should be created in v4."""
        db = Database(tmp_path / "eval.db")
        cur = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_results';"
        )
        assert cur.fetchone() is not None
        db.close()

    def test_agent_task_suites_table_exists(self, tmp_path: Path) -> None:
        """agent_task_suites table should be created in v4."""
        db = Database(tmp_path / "eval.db")
        cur = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_task_suites';"
        )
        assert cur.fetchone() is not None
        db.close()

    def test_agent_runs_schema(self, tmp_path: Path) -> None:
        """agent_runs table should have the correct columns."""
        db = Database(tmp_path / "eval.db")
        cur = db.connection.execute("PRAGMA table_info(agent_runs);")
        columns = {row[1] for row in cur.fetchall()}
        expected = {
            "run_id", "suite_id", "agent_type", "status",
            "created_at", "completed_at", "config_json",
            "total_steps", "completed_steps", "mean_score",
        }
        assert expected.issubset(columns), f"Missing columns: {expected - columns}"
        db.close()

    def test_agent_results_schema(self, tmp_path: Path) -> None:
        """agent_results table should have the correct columns."""
        db = Database(tmp_path / "eval.db")
        cur = db.connection.execute("PRAGMA table_info(agent_results);")
        columns = {row[1] for row in cur.fetchall()}
        expected = {
            "result_id", "run_id", "step_id", "agent_output",
            "success", "score", "error", "duration_seconds",
            "tokens_used", "completed_at",
        }
        assert expected.issubset(columns), f"Missing columns: {expected - columns}"
        db.close()

    def test_agent_task_suites_schema(self, tmp_path: Path) -> None:
        """agent_task_suites table should have the correct columns."""
        db = Database(tmp_path / "eval.db")
        cur = db.connection.execute("PRAGMA table_info(agent_task_suites);")
        columns = {row[1] for row in cur.fetchall()}
        expected = {
            "suite_id", "name", "description", "yaml_content",
            "is_builtin", "created_at",
        }
        assert expected.issubset(columns), f"Missing columns: {expected - columns}"
        db.close()

    def test_insert_agent_run(self, tmp_path: Path) -> None:
        """Should be able to insert and retrieve an agent run."""
        from src.agent_models import AgentRun, AgentStatus

        db = Database(tmp_path / "eval.db")
        run = AgentRun(
            suite_id="echo-v1",
            agent_type="python",
            status=AgentStatus.COMPLETED,
            config={"test": True},
        )
        db.insert_agent_run(run)
        fetched = db.get_agent_run(run.run_id)
        assert fetched is not None
        assert fetched.suite_id == "echo-v1"
        assert fetched.agent_type == "python"
        assert fetched.status == AgentStatus.COMPLETED
        db.close()

    def test_insert_agent_result(self, tmp_path: Path) -> None:
        """Should be able to insert and retrieve agent results."""
        from src.agent_models import AgentResult, AgentRun

        db = Database(tmp_path / "eval.db")
        run = AgentRun(suite_id="echo-v1", agent_type="python")
        db.insert_agent_run(run)

        result = AgentResult(
            step_id="step-1",
            agent_output="hello",
            success=True,
            score=1.0,
        )
        db.insert_agent_result(run.run_id, result)
        results = db.get_agent_results(run.run_id)
        assert len(results) == 1
        assert results[0].step_id == "step-1"
        assert results[0].score == 1.0
        db.close()

    def test_insert_agent_task_suite(self, tmp_path: Path) -> None:
        """Should be able to insert and retrieve agent task suites."""
        from src.agent_models import TaskSuite

        db = Database(tmp_path / "eval.db")
        suite = TaskSuite(
            suite_id="custom-v1",
            name="Custom Suite",
            description="A custom test suite",
            steps=[],
        )
        db.insert_agent_task_suite(suite)
        suites = db.list_agent_task_suites()
        # 5 built-in + 1 custom = 6
        assert len(suites) == 6
        custom = [s for s in suites if s.suite_id == "custom-v1"]
        assert len(custom) == 1
        assert custom[0].name == "Custom Suite"
        db.close()

    def test_list_agent_runs(self, tmp_path: Path) -> None:
        """Should be able to list agent runs."""
        from src.agent_models import AgentRun

        db = Database(tmp_path / "eval.db")
        for i in range(3):
            run = AgentRun(suite_id=f"suite-{i}", agent_type="python")
            db.insert_agent_run(run)
        runs = db.list_agent_runs()
        assert len(runs) == 3
        db.close()

    def test_seeded_builtin_suites(self, tmp_path: Path) -> None:
        """Built-in task suites should be seeded on migration."""
        db = Database(tmp_path / "eval.db")
        suites = db.list_agent_task_suites()
        suite_ids = {s.suite_id for s in suites}
        assert "echo-v1" in suite_ids
        assert "math-v1" in suite_ids
        assert "file-read-v1" in suite_ids
        assert "string-reversal-v1" in suite_ids
        assert "multi-step-v1" in suite_ids
        db.close()

    def test_migration_from_v3_preserves_data(self, tmp_path: Path) -> None:
        """Migrating from v3 to v4 should preserve existing data."""
        # Create a v3 database manually
        db_path = tmp_path / "eval_v3.db"
        db = Database(db_path)
        # Verify it's at current version
        assert db.get_schema_version() == CURRENT_SCHEMA_VERSION
        # Existing tables should still work
        from src.models import EvalRun
        run = EvalRun(config={"old": True}, judge_model="legacy")
        db.insert_run(run)
        fetched = db.get_run(run.run_id)
        assert fetched is not None
        assert fetched.config == {"old": True}
        db.close()

    def test_rollback_from_v4_to_v3(self, tmp_path: Path) -> None:
        """Rollback from v4 to v3 should remove agent tables."""
        db = Database(tmp_path / "eval.db")
        assert db.get_schema_version() == 4
        db.rollback(3)
        assert db.get_schema_version() == 3
        # Agent tables should be gone
        cur = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='agent_runs';"
        )
        assert cur.fetchone() is None
        db.close()

    def test_agent_results_foreign_key(self, tmp_path: Path) -> None:
        """agent_results should have a foreign key to agent_runs."""
        db = Database(tmp_path / "eval.db")
        cur = db.connection.execute("PRAGMA foreign_key_list(agent_results);")
        fks = cur.fetchall()
        assert len(fks) > 0
        # FK should reference agent_runs(run_id)
        assert any(fk[2] == "agent_runs" and fk[3] == "run_id" for fk in fks)
        db.close()

    def test_agent_runs_index(self, tmp_path: Path) -> None:
        """agent_runs should have an index on suite_id."""
        db = Database(tmp_path / "eval.db")
        cur = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='agent_runs';"
        )
        indexes = {row[0] for row in cur.fetchall()}
        assert "idx_agent_runs_suite" in indexes
        db.close()

    def test_agent_results_index(self, tmp_path: Path) -> None:
        """agent_results should have an index on run_id."""
        db = Database(tmp_path / "eval.db")
        cur = db.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='agent_results';"
        )
        indexes = {row[0] for row in cur.fetchall()}
        assert "idx_agent_results_run" in indexes
        db.close()
