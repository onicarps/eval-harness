"""Tests for src/agent_evaluator.py — AgentEvaluator that runs suites and scores trajectories."""

from __future__ import annotations

import pytest

from src.agent import PythonAgent
from src.agent_evaluator import AgentEvaluator, EvaluatorConfig
from src.agent_models import (
    AgentResult,
    AgentRun,
    AgentStatus,
    ScoringSummary,
    TaskStep,
    TaskSuite,
)
from src.task_suite import get_suite_by_id

# ── EvaluatorConfig tests ───────────────────────────────────────────────────────


class TestEvaluatorConfig:
    """Tests for the EvaluatorConfig dataclass."""

    def test_default_config(self) -> None:
        config = EvaluatorConfig()
        assert config.pass_threshold == 0.7
        assert config.max_steps_per_run == 50
        assert config.timeout_seconds == 60.0
        assert config.scoring_method == "exact_match"

    def test_custom_config(self) -> None:
        config = EvaluatorConfig(
            pass_threshold=0.8,
            max_steps_per_run=100,
            timeout_seconds=120.0,
            scoring_method="fuzzy_match",
        )
        assert config.pass_threshold == 0.8
        assert config.max_steps_per_run == 100
        assert config.timeout_seconds == 120.0


# ── AgentEvaluator tests ────────────────────────────────────────────────────────


class TestAgentEvaluator:
    """Tests for the AgentEvaluator."""

    @pytest.fixture()
    def perfect_echo_agent(self) -> PythonAgent:
        """An agent that perfectly echoes input."""
        def echo(task: str) -> str:
            return task
        return PythonAgent(name="perfect-echo", handler=echo)

    @pytest.fixture()
    def perfect_math_agent(self) -> PythonAgent:
        """An agent that correctly answers math questions."""
        def math_fn(task: str) -> str:
            answers = {
                "What is 2+2?": "4",
                "What is 10-3?": "7",
                "What is 3*4?": "12",
                "What is 15/3?": "5",
                "What is 7+8?": "15",
            }
            return answers.get(task, "unknown")
        return PythonAgent(name="perfect-math", handler=math_fn)

    @pytest.fixture()
    def failing_agent(self) -> PythonAgent:
        """An agent that always fails."""
        def fail(task: str) -> str:
            raise RuntimeError("always fails")
        return PythonAgent(name="failing", handler=fail)

    @pytest.fixture()
    def suite_echo(self) -> TaskSuite:
        return get_suite_by_id("echo-v1")

    @pytest.fixture()
    def suite_math(self) -> TaskSuite:
        return get_suite_by_id("math-v1")

    @pytest.mark.asyncio
    async def test_evaluate_echo_suite(
        self, perfect_echo_agent: PythonAgent, suite_echo: TaskSuite
    ) -> None:
        evaluator = AgentEvaluator()
        run = await evaluator.evaluate(perfect_echo_agent, suite_echo)
        assert run.status == AgentStatus.COMPLETED
        assert len(run.results) == 5
        assert all(r.success for r in run.results)

    @pytest.mark.asyncio
    async def test_evaluate_math_suite(
        self, perfect_math_agent: PythonAgent, suite_math: TaskSuite
    ) -> None:
        evaluator = AgentEvaluator()
        run = await evaluator.evaluate(perfect_math_agent, suite_math)
        assert run.status == AgentStatus.COMPLETED
        assert len(run.results) == 5
        assert all(r.success for r in run.results)

    @pytest.mark.asyncio
    async def test_evaluate_failing_agent(
        self, failing_agent: PythonAgent, suite_echo: TaskSuite
    ) -> None:
        evaluator = AgentEvaluator()
        run = await evaluator.evaluate(failing_agent, suite_echo)
        assert run.status == AgentStatus.COMPLETED
        assert len(run.results) == 5
        assert all(not r.success for r in run.results)

    @pytest.mark.asyncio
    async def test_evaluate_creates_agent_run(
        self, perfect_echo_agent: PythonAgent, suite_echo: TaskSuite
    ) -> None:
        evaluator = AgentEvaluator()
        run = await evaluator.evaluate(perfect_echo_agent, suite_echo)
        assert isinstance(run, AgentRun)
        assert run.suite_id == "echo-v1"
        assert run.agent_type == "python"
        assert run.completed_at is not None

    @pytest.mark.asyncio
    async def test_evaluate_respects_max_steps(
        self, perfect_echo_agent: PythonAgent
    ) -> None:
        """Evaluator stops after max_steps_per_run steps."""
        suite = TaskSuite(
            suite_id="big",
            name="Big Suite",
            steps=[
                TaskStep(id=f"s{i}", prompt=f"task{i}", expected_output=f"task{i}")
                for i in range(100)
            ],
        )
        config = EvaluatorConfig(max_steps_per_run=3)
        evaluator = AgentEvaluator(config=config)
        run = await evaluator.evaluate(perfect_echo_agent, suite)
        assert len(run.results) == 3

    @pytest.mark.asyncio
    async def test_evaluate_empty_suite(
        self, perfect_echo_agent: PythonAgent
    ) -> None:
        """Evaluator handles empty suites gracefully."""
        suite = TaskSuite(suite_id="empty", name="Empty", steps=[])
        evaluator = AgentEvaluator()
        run = await evaluator.evaluate(perfect_echo_agent, suite)
        assert run.status == AgentStatus.COMPLETED
        assert len(run.results) == 0


# ── ScoringSummary tests ────────────────────────────────────────────────────────


class TestScoringSummary:
    """Tests for the ScoringSummary computation."""

    def test_perfect_score(self) -> None:
        summary = ScoringSummary(
            run_id="r1",
            suite_id="echo-v1",
            total_steps=5,
            completed_steps=5,
            passed_steps=5,
            mean_score=1.0,
            pass_rate=1.0,
            efficiency=1.0,
            trajectory_score=1.0,
        )
        assert summary.pass_rate == 1.0
        assert summary.trajectory_score == 1.0

    def test_zero_score(self) -> None:
        summary = ScoringSummary(
            run_id="r2",
            suite_id="echo-v1",
            total_steps=5,
            completed_steps=5,
            passed_steps=0,
            mean_score=0.0,
            pass_rate=0.0,
            efficiency=1.0,
            trajectory_score=0.0,
        )
        assert summary.pass_rate == 0.0

    def test_partial_score(self) -> None:
        summary = ScoringSummary(
            run_id="r3",
            suite_id="echo-v1",
            total_steps=5,
            completed_steps=5,
            passed_steps=3,
            mean_score=0.6,
            pass_rate=0.6,
            efficiency=1.0,
            trajectory_score=0.6,
        )
        assert summary.passed_steps == 3
        assert summary.pass_rate == 0.6


# ── Trajectory scoring tests ───────────────────────────────────────────────────


class TestTrajectoryScoring:
    """Tests for trajectory scoring logic."""

    @pytest.mark.asyncio
    async def test_compute_summary_perfect(self) -> None:
        """compute_summary returns perfect scores for a perfect run."""
        evaluator = AgentEvaluator()
        run = AgentRun(
            run_id="test",
            suite_id="echo-v1",
            agent_type="python",
            status=AgentStatus.COMPLETED,
            results=[
                AgentResult(step_id=f"s{i}", agent_output="ok", success=True, score=1.0)
                for i in range(5)
            ],
        )
        suite = get_suite_by_id("echo-v1")
        assert suite is not None
        summary = evaluator.compute_summary(run, suite)
        assert summary.mean_score == 1.0
        assert summary.pass_rate == 1.0
        assert summary.trajectory_score == 1.0

    @pytest.mark.asyncio
    async def test_compute_summary_partial(self) -> None:
        """compute_summary handles partial success."""
        evaluator = AgentEvaluator()
        run = AgentRun(
            run_id="test",
            suite_id="echo-v1",
            agent_type="python",
            status=AgentStatus.COMPLETED,
            results=[
                AgentResult(step_id="s1", agent_output="ok", success=True, score=1.0),
                AgentResult(step_id="s2", agent_output="ok", success=True, score=1.0),
                AgentResult(step_id="s3", agent_output="", success=False, score=0.0),
                AgentResult(step_id="s4", agent_output="", success=False, score=0.0),
                AgentResult(step_id="s5", agent_output="partial", success=True, score=0.5),
            ],
        )
        suite = get_suite_by_id("echo-v1")
        assert suite is not None
        summary = evaluator.compute_summary(run, suite)
        assert summary.completed_steps == 5
        assert summary.passed_steps == 2  # score >= 0.7 threshold (1.0, 1.0 pass; 0.0, 0.0, 0.5 fail)
        assert summary.mean_score == pytest.approx(0.5)
        assert summary.pass_rate == pytest.approx(0.4)

    @pytest.mark.asyncio
    async def test_compute_summary_empty_run(self) -> None:
        """compute_summary handles empty runs."""
        evaluator = AgentEvaluator()
        run = AgentRun(
            run_id="test",
            suite_id="echo-v1",
            agent_type="python",
            status=AgentStatus.COMPLETED,
            results=[],
        )
        suite = get_suite_by_id("echo-v1")
        assert suite is not None
        summary = evaluator.compute_summary(run, suite)
        assert summary.total_steps == 5
        assert summary.completed_steps == 0
        assert summary.mean_score == 0.0

    @pytest.mark.asyncio
    async def test_compute_summary_efficiency(self) -> None:
        """Efficiency reflects completed/total ratio."""
        evaluator = AgentEvaluator()
        run = AgentRun(
            run_id="test",
            suite_id="echo-v1",
            agent_type="python",
            status=AgentStatus.COMPLETED,
            results=[
                AgentResult(step_id="s1", agent_output="ok", success=True, score=1.0),
            ],
        )
        suite = get_suite_by_id("echo-v1")
        assert suite is not None
        summary = evaluator.compute_summary(run, suite)
        assert summary.efficiency == pytest.approx(0.2)  # 1/5


# ── Integration tests ───────────────────────────────────────────────────────────


class TestIntegration:
    """Integration tests for the full agent evaluation pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_echo(self) -> None:
        """Full pipeline: agent -> evaluator -> summary for echo suite."""
        def echo(task: str) -> str:
            return task

        agent = PythonAgent(name="echo", handler=echo)
        suite = get_suite_by_id("echo-v1")
        assert suite is not None

        evaluator = AgentEvaluator()
        run = await evaluator.evaluate(agent, suite)
        summary = evaluator.compute_summary(run, suite)

        assert run.status == AgentStatus.COMPLETED
        assert summary.total_steps == 5
        assert summary.completed_steps == 5
        assert summary.trajectory_score == 1.0

    @pytest.mark.asyncio
    async def test_full_pipeline_math(self) -> None:
        """Full pipeline for math suite."""
        def math_fn(task: str) -> str:
            answers = {
                "What is 2+2?": "4",
                "What is 10-3?": "7",
                "What is 3*4?": "12",
                "What is 15/3?": "5",
                "What is 7+8?": "15",
            }
            return answers.get(task, "0")

        agent = PythonAgent(name="math", handler=math_fn)
        suite = get_suite_by_id("math-v1")
        assert suite is not None

        evaluator = AgentEvaluator()
        run = await evaluator.evaluate(agent, suite)
        summary = evaluator.compute_summary(run, suite)

        assert summary.completed_steps == 5
        assert summary.mean_score == 1.0

    @pytest.mark.asyncio
    async def test_full_pipeline_with_context_manager(self) -> None:
        """Full pipeline using async context manager."""
        def echo(task: str) -> str:
            return task

        async with PythonAgent(name="echo-cm", handler=echo) as agent:
            suite = get_suite_by_id("echo-v1")
            assert suite is not None
            evaluator = AgentEvaluator()
            run = await evaluator.evaluate(agent, suite)
            assert run.status == AgentStatus.COMPLETED
