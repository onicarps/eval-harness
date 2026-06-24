"""Tests for LLM-judge scoring method in agent evaluation.

Tests the integration between AgentEvaluator and LLMEvaluator when
scoring_method='llm_judge' is selected in EvaluatorConfig.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from src.agent import PythonAgent
from src.agent_evaluator import AgentEvaluator, EvaluatorConfig
from src.agent_models import (
    AgentResult,
    AgentStatus,
    TaskStep,
    TaskStepType,
    TaskSuite,
)
from src.evaluator import (
    LLMEvaluator,
    EvaluatorConfig as LLMEvaluatorConfig,
    combine_scores,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture()
def open_ended_suite() -> TaskSuite:
    """A suite with open-ended tasks requiring LLM judge scoring."""
    return TaskSuite(
        suite_id="llm-judge-v1",
        name="LLM Judge Tests",
        description="Open-ended tasks that require LLM judge scoring.",
        steps=[
            TaskStep(
                id="open-1",
                prompt="Explain quantum computing in one sentence.",
                expected_output="Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computation.",
                step_type=TaskStepType.ECHO,
            ),
            TaskStep(
                id="open-2",
                prompt="Write a haiku about machine learning.",
                expected_output="An AI model learns patterns from data to make predictions.",
                step_type=TaskStepType.ECHO,
            ),
            TaskStep(
                id="open-3",
                prompt="Summarize the benefits of exercise.",
                expected_output="Exercise improves cardiovascular health, strengthens muscles, and boosts mental well-being.",
                step_type=TaskStepType.ECHO,
            ),
            TaskStep(
                id="open-4",
                prompt="What is the meaning of life?",
                expected_output="The meaning of life is a philosophical question about purpose and significance.",
                step_type=TaskStepType.ECHO,
            ),
            TaskStep(
                id="open-5",
                prompt="Describe a sunset to someone who has never seen one.",
                expected_output="A sunset is the sun disappearing below the horizon, painting the sky in warm colors.",
                step_type=TaskStepType.ECHO,
            ),
        ],
    )


@pytest.fixture()
def good_agent() -> PythonAgent:
    """An agent that gives good open-ended responses."""
    responses = {
        "Explain quantum computing in one sentence.": "Quantum computing harnesses superposition and entanglement to solve problems faster than classical computers.",
        "Write a haiku about machine learning.": "Data flows through code / Patterns emerge from the noise / Machines learn and grow",
        "Summarize the benefits of exercise.": "Regular exercise strengthens the heart, builds muscle, and improves mental health through endorphin release.",
        "What is the meaning of life?": "The meaning of life is a deep philosophical question about purpose, existence, and what makes our experiences significant.",
        "Describe a sunset to someone who has never seen one.": "A sunset is when the sun slowly sinks below the horizon, painting the sky in brilliant shades of orange, pink, and purple.",
    }

    def handler(task: str) -> str:
        return responses.get(task, "I don't know the answer to that.")

    return PythonAgent(name="good-agent", handler=handler)


@pytest.fixture()
def poor_agent() -> PythonAgent:
    """An agent that gives poor open-ended responses."""
    responses = {
        "Explain quantum computing in one sentence.": "Computers are fast.",
        "Write a haiku about machine learning.": "I like pizza.",
        "Summarize the benefits of exercise.": "Exercise is okay I guess.",
        "What is the meaning of life?": "42.",
        "Describe a sunset to someone who has never seen one.": "It's dark outside.",
    }

    def handler(task: str) -> str:
        return responses.get(task, "no")

    return PythonAgent(name="poor-agent", handler=handler)


@pytest.fixture()
def mock_db(tmp_path_factory) -> Any:
    """Create a temporary Database for LLMEvaluator."""
    from src.db import Database
    db = Database(tmp_path_factory.mktemp("test_db") / "test.db")
    yield db
    db.close()


@pytest.fixture()
def mock_llm_evaluator(mock_db: Any) -> LLMEvaluator:
    """Create an LLMEvaluator with mocked API key for testing."""
    config = LLMEvaluatorConfig(
        api_key="test-key",
        judges=["test/judge-model"],
        concurrency=2,
        timeout=30.0,
        use_cache=False,
    )
    return LLMEvaluator(mock_db, config)


# ── EvaluatorConfig llm_judge tests ────────────────────────────────────────────


class TestEvaluatorConfigLLMJudge:
    """Tests for EvaluatorConfig with llm_judge scoring method."""

    def test_default_scoring_method_is_exact_match(self) -> None:
        """Default scoring method remains 'exact_match'."""
        config = EvaluatorConfig()
        assert config.scoring_method == "exact_match"

    def test_llm_judge_scoring_method_accepted(self) -> None:
        """EvaluatorConfig accepts 'llm_judge' as scoring_method."""
        config = EvaluatorConfig(scoring_method="llm_judge")
        assert config.scoring_method == "llm_judge"

    def test_llm_judge_with_custom_threshold(self) -> None:
        """EvaluatorConfig accepts llm_judge with custom pass_threshold."""
        config = EvaluatorConfig(scoring_method="llm_judge", pass_threshold=0.5)
        assert config.scoring_method == "llm_judge"
        assert config.pass_threshold == 0.5


# ── LLMEvaluator integration tests ─────────────────────────────────────────────


class TestLLMEvaluatorScoreStep:
    """Tests for LLMEvaluator.score_step method."""

    @pytest.mark.asyncio
    async def test_score_step_returns_agent_result(
        self, mock_llm_evaluator: LLMEvaluator
    ) -> None:
        """score_step returns an AgentResult with a score from the judge."""
        step = TaskStep(
            id="test-1",
            prompt="Explain AI",
            expected_output="AI is intelligence demonstrated by machines.",
        )
        result = await mock_llm_evaluator.score_step(step)
        assert isinstance(result, AgentResult)
        assert result.step_id == "test-1"
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_score_step_calls_evaluate_batch(
        self, mock_db: Any
    ) -> None:
        """score_step internally uses evaluate_batch."""
        config = LLMEvaluatorConfig(
            api_key="test-key",
            judges=["test/judge"],
            use_cache=False,
        )
        evaluator = LLMEvaluator(mock_db, config)

        step = TaskStep(
            id="step-1",
            prompt="What is 2+2?",
            expected_output="4",
        )

        # Mock the _call_judge method to return a fixed response
        judge_response = {
            "faithfulness": 0.9,
            "task_completion": 0.85,
            "reasoning": "good answer",
            "faithfulness_reasoning": "matches reference",
            "task_completion_reasoning": "completes the task",
        }

        with patch.object(
            evaluator, "_call_judge", new_callable=AsyncMock, return_value=judge_response
        ) as mock_call:
            result = await evaluator.score_step(step)
            mock_call.assert_called_once()

        assert result.step_id == "step-1"
        expected_score = combine_scores(0.9, 0.85)
        assert result.score == pytest.approx(expected_score)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_score_step_handles_judge_failure(
        self, mock_db: Any
    ) -> None:
        """score_step returns score 0.0 when judge fails."""
        config = LLMEvaluatorConfig(
            api_key="test-key",
            judges=["test/judge"],
            use_cache=False,
        )
        evaluator = LLMEvaluator(mock_db, config)

        step = TaskStep(
            id="step-fail",
            prompt="Impossible question",
            expected_output="Expected output",
        )

        with patch.object(
            evaluator, "_call_judge", new_callable=AsyncMock, side_effect=Exception("API error")
        ):
            result = await evaluator.score_step(step)

        assert result.score == 0.0
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_score_step_empty_expected_output(
        self, mock_llm_evaluator: LLMEvaluator
    ) -> None:
        """score_step with empty expected_output still calls judge."""
        step = TaskStep(
            id="step-empty",
            prompt="Say anything",
            expected_output="",
        )
        result = await mock_llm_evaluator.score_step(step)
        assert isinstance(result, AgentResult)
        assert result.step_id == "step-empty"


# ── AgentEvaluator with llm_judge scoring ──────────────────────────────────────


class TestAgentEvaluatorLLMJudge:
    """Tests for AgentEvaluator when scoring_method='llm_judge'."""

    @pytest.mark.asyncio
    async def test_evaluate_with_llm_judge_delegates_to_evaluator(
        self, good_agent: PythonAgent, open_ended_suite: TaskSuite, mock_db: Any
    ) -> None:
        """When scoring_method='llm_judge', evaluate() delegates scoring to LLMEvaluator."""
        config = EvaluatorConfig(scoring_method="llm_judge")
        evaluator = AgentEvaluator(config=config)

        # Create a mock LLMEvaluator
        mock_llm_eval = AsyncMock(spec=LLMEvaluator)
        mock_llm_eval.score_step = AsyncMock(
            return_value=AgentResult(
                step_id="test",
                agent_output="mocked output",
                success=True,
                score=0.9,
            )
        )

        with patch.object(evaluator, "_get_judge_evaluator", return_value=mock_llm_eval):
            run = await evaluator.evaluate(good_agent, open_ended_suite)

        assert run.status == AgentStatus.COMPLETED
        assert len(run.results) == 5
        # All results should come from the mock
        assert all(r.score == 0.9 for r in run.results)
        assert mock_llm_eval.score_step.call_count == 5

    @pytest.mark.asyncio
    async def test_evaluate_with_exact_match_does_not_use_llm(
        self, good_agent: PythonAgent, open_ended_suite: TaskSuite
    ) -> None:
        """When scoring_method='exact_match' (default), LLMEvaluator is NOT used."""
        config = EvaluatorConfig(scoring_method="exact_match")
        evaluator = AgentEvaluator(config=config)

        # If LLMEvaluator was called, it would fail since we don't have a mock
        run = await evaluator.evaluate(good_agent, open_ended_suite)
        assert run.status == AgentStatus.COMPLETED
        # With exact_match, open-ended responses won't match perfectly
        assert any(r.score < 1.0 for r in run.results)

    @pytest.mark.asyncio
    async def test_evaluate_llm_judge_with_poor_agent(
        self, poor_agent: PythonAgent, open_ended_suite: TaskSuite, mock_db: Any
    ) -> None:
        """LLM judge scores poor agent lower than good agent."""
        config = EvaluatorConfig(scoring_method="llm_judge")
        evaluator = AgentEvaluator(config=config)

        mock_llm_eval = AsyncMock(spec=LLMEvaluator)
        # Simulate low scores for poor agent
        mock_llm_eval.score_step = AsyncMock(
            return_value=AgentResult(
                step_id="test",
                agent_output="bad output",
                success=True,
                score=0.2,
            )
        )

        with patch.object(evaluator, "_get_judge_evaluator", return_value=mock_llm_eval):
            run = await evaluator.evaluate(poor_agent, open_ended_suite)

        assert all(r.score == 0.2 for r in run.results)

    @pytest.mark.asyncio
    async def test_compute_summary_with_llm_judge_scores(
        self, good_agent: PythonAgent, open_ended_suite: TaskSuite, mock_db: Any
    ) -> None:
        """compute_summary correctly aggregates LLM judge scores."""
        config = EvaluatorConfig(scoring_method="llm_judge", pass_threshold=0.7)
        evaluator = AgentEvaluator(config=config)

        mock_llm_eval = AsyncMock(spec=LLMEvaluator)
        # Return varying scores
        scores = [0.9, 0.8, 0.6, 0.95, 0.75]
        call_count = [0]

        async def mock_score(step: TaskStep, agent_output: str | None = None) -> AgentResult:
            score = scores[call_count[0]]
            call_count[0] += 1
            return AgentResult(
                step_id=step.id,
                agent_output="output",
                success=True,
                score=score,
            )

        mock_llm_eval.score_step = AsyncMock(side_effect=mock_score)

        with patch.object(evaluator, "_get_judge_evaluator", return_value=mock_llm_eval):
            run = await evaluator.evaluate(good_agent, open_ended_suite)

        summary = evaluator.compute_summary(run, open_ended_suite)

        assert summary.completed_steps == 5
        assert summary.passed_steps == 4  # 0.9, 0.8, 0.95, 0.75 >= 0.7; 0.6 < 0.7
        assert summary.mean_score == pytest.approx(0.8)
        assert summary.pass_rate == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_llm_judge_run_records_scoring_method(
        self, good_agent: PythonAgent, open_ended_suite: TaskSuite, mock_db: Any
    ) -> None:
        """AgentRun.config records the scoring method used."""
        config = EvaluatorConfig(scoring_method="llm_judge")
        evaluator = AgentEvaluator(config=config)

        mock_llm_eval = AsyncMock(spec=LLMEvaluator)
        mock_llm_eval.score_step = AsyncMock(
            return_value=AgentResult(
                step_id="test",
                agent_output="output",
                success=True,
                score=0.85,
            )
        )

        with patch.object(evaluator, "_get_judge_evaluator", return_value=mock_llm_eval):
            run = await evaluator.evaluate(good_agent, open_ended_suite)

        assert run.config["scoring_method"] == "llm_judge"


# ── Built-in llm-judge-v1 suite tests ─────────────────────────────────────────


class TestLLMJudgeV1Suite:
    """Tests for the built-in 'llm-judge-v1' task suite."""

    def test_llm_judge_v1_suite_registered(self) -> None:
        """The llm-judge-v1 suite is registered in BuiltinSuiteRegistry."""
        from src.task_suite import BuiltinSuiteRegistry
        suite = BuiltinSuiteRegistry.get("llm-judge-v1")
        assert suite is not None
        assert suite.suite_id == "llm-judge-v1"

    def test_llm_judge_v1_suite_has_steps(self) -> None:
        """The llm-judge-v1 suite has at least 3 open-ended steps."""
        from src.task_suite import BuiltinSuiteRegistry
        suite = BuiltinSuiteRegistry.get("llm-judge-v1")
        assert suite is not None
        assert len(suite.steps) >= 3

    def test_llm_judge_v1_suite_steps_are_open_ended(self) -> None:
        """The llm-judge-v1 suite steps have meaningful expected outputs."""
        from src.task_suite import BuiltinSuiteRegistry
        suite = BuiltinSuiteRegistry.get("llm-judge-v1")
        assert suite is not None
        for step in suite.steps:
            assert len(step.expected_output) > 10  # Non-trivial expected output
            assert step.id  # Has an ID

    def test_llm_judge_v1_in_list_ids(self) -> None:
        """The llm-judge-v1 suite appears in list_ids()."""
        from src.task_suite import BuiltinSuiteRegistry
        ids = BuiltinSuiteRegistry.list_ids()
        assert "llm-judge-v1" in ids

    def test_llm_judge_v1_suite_metadata(self) -> None:
        """The llm-judge-v1 suite has appropriate name and description."""
        from src.task_suite import BuiltinSuiteRegistry
        suite = BuiltinSuiteRegistry.get("llm-judge-v1")
        assert suite is not None
        assert suite.name
        assert suite.description
        assert len(suite.description) > 10


# ── LLMEvaluator.score_step implementation tests ───────────────────────────────


class TestLLMEvaluatorScoreStepImplementation:
    """Tests that verify LLMEvaluator.score_step builds EvalRecords correctly."""

    @pytest.mark.asyncio
    async def test_score_step_builds_eval_record(
        self, mock_db: Any
    ) -> None:
        """score_step creates an EvalRecord from the step data."""
        config = LLMEvaluatorConfig(
            api_key="test-key",
            judges=["test/judge"],
            use_cache=False,
        )
        evaluator = LLMEvaluator(mock_db, config)

        step = TaskStep(
            id="record-test",
            prompt="Explain gravity",
            expected_output="Gravity is a force that attracts objects with mass.",
        )

        judge_response = {
            "faithfulness": 0.75,
            "task_completion": 0.8,
            "reasoning": "decent explanation",
            "faithfulness_reasoning": "mostly matches",
            "task_completion_reasoning": "partially complete",
        }

        with patch.object(
            evaluator, "_call_judge", new_callable=AsyncMock, return_value=judge_response
        ):
            result = await evaluator.score_step(step)

        assert result.step_id == "record-test"
        expected_combined = combine_scores(0.75, 0.8)
        assert result.score == pytest.approx(expected_combined)

    @pytest.mark.asyncio
    async def test_score_step_clamps_scores(
        self, mock_db: Any
    ) -> None:
        """score_step clamps out-of-range judge scores to [0, 1]."""
        config = LLMEvaluatorConfig(
            api_key="test-key",
            judges=["test/judge"],
            use_cache=False,
        )
        evaluator = LLMEvaluator(mock_db, config)

        step = TaskStep(
            id="clamp-test",
            prompt="Test",
            expected_output="Test output",
        )

        # Judge returns out-of-range scores
        judge_response = {
            "faithfulness": 1.5,
            "task_completion": -0.3,
            "reasoning": "bad scores",
            "faithfulness_reasoning": "",
            "task_completion_reasoning": "",
        }

        with patch.object(
            evaluator, "_call_judge", new_callable=AsyncMock, return_value=judge_response
        ):
            result = await evaluator.score_step(step)

        # faithfulness clamped to 1.0, task_completion clamped to 0.0
        expected = combine_scores(1.0, 0.0)
        assert result.score == pytest.approx(expected)
        assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_score_step_uses_cache_when_enabled(
        self, tmp_path_factory: Any
    ) -> None:
        """score_step uses cached judge response when caching is enabled."""
        from src.db import Database
        db = Database(tmp_path_factory.mktemp("cache_test") / "cache.db")
        try:
            config = LLMEvaluatorConfig(
                api_key="test-key",
                judges=["cached-judge"],
                use_cache=True,
            )
            evaluator = LLMEvaluator(db, config)

            step = TaskStep(
                id="cache-step",
                prompt="Cached question",
                expected_output="Cached answer",
            )

            judge_response = {
                "faithfulness": 0.8,
                "task_completion": 0.9,
                "reasoning": "cached response",
                "faithfulness_reasoning": "",
                "task_completion_reasoning": "",
            }

            # First call should hit the API
            with patch.object(
                evaluator, "_call_judge", new_callable=AsyncMock, return_value=judge_response
            ) as mock_call:
                result1 = await evaluator.score_step(step)
                assert mock_call.call_count == 1

            # Second call should use cache (not call API again)
            with patch.object(
                evaluator, "_call_judge", new_callable=AsyncMock, return_value=judge_response
            ) as mock_call:
                result2 = await evaluator.score_step(step)
                assert mock_call.call_count == 0  # Cache hit

            assert result1.score == result2.score
        finally:
            db.close()
