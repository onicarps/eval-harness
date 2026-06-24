"""Agent evaluator — runs task suites against agents and scores trajectories.

Provides:
- EvaluatorConfig: Configuration for the agent evaluator.
- AgentEvaluator: Orchestrates running a TaskSuite against an Agent,
  collecting results and computing a ScoringSummary.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from src.agent import Agent
from src.agent_models import (
    AgentResult,
    AgentRun,
    AgentStatus,
    ScoringSummary,
    TaskSuite,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluatorConfig:
    """Configuration for :class:`AgentEvaluator`.

    Attributes:
        pass_threshold: Score threshold for a step to count as "passed".
        max_steps_per_run: Maximum number of steps to execute per run.
        timeout_seconds: Overall timeout for the entire run.
        scoring_method: Scoring method. One of 'exact_match' (default),
            'trajectory', or 'llm_judge'. When 'llm_judge' is selected,
            an LLMEvaluator is used to score open-ended agent outputs.
        judge_api_key: API key for the LLM judge (required when
            scoring_method='llm_judge').
        judge_models: List of judge model IDs for LLM judge scoring
            (defaults to a single mock model for testing).
    """

    pass_threshold: float = 0.7
    max_steps_per_run: int = 50
    timeout_seconds: float = 60.0
    scoring_method: str = "exact_match"
    judge_api_key: str | None = None
    judge_models: list[str] | None = None


class AgentEvaluator:
    """Runs a :class:`TaskSuite` against an :class:`Agent` and scores the trajectory.

    Usage::

        evaluator = AgentEvaluator()
        run = await evaluator.evaluate(agent, suite)
        summary = evaluator.compute_summary(run, suite)
    """

    def __init__(self, config: EvaluatorConfig | None = None) -> None:
        """Initialize the evaluator.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or EvaluatorConfig()
        self._judge_evaluator = None

    def _get_judge_evaluator(self) -> Any:
        """Lazily create and return the LLMEvaluator for llm_judge scoring.

        Returns:
            An LLMEvaluator instance.

        Raises:
            ValueError: If scoring_method is not 'llm_judge' or if
                required configuration is missing.
        """
        if self.config.scoring_method != "llm_judge":
            raise ValueError(
                "LLMEvaluator is only used when scoring_method='llm_judge'"
            )
        if self._judge_evaluator is None:
            import tempfile

            from src.db import Database
            from src.evaluator import EvaluatorConfig as LLMConfig
            from src.evaluator import LLMEvaluator as LLMEvaluatorClass

            db_path = tempfile.mktemp(suffix=".db")
            db = Database(db_path)

            api_key = self.config.judge_api_key or "test-key"
            models = self.config.judge_models or ["local/mock-judge"]
            llm_config = LLMConfig(
                api_key=api_key,
                judges=models,
                timeout=self.config.timeout_seconds,
            )
            self._judge_evaluator = LLMEvaluatorClass(db, llm_config)
        return self._judge_evaluator

    async def evaluate(self, agent: Agent, suite: TaskSuite) -> AgentRun:
        """Run a task suite against an agent.

        Iterates through the suite's steps (up to ``max_steps_per_run``),
        executes each step via the agent, and collects results.

        Args:
            agent: The agent to evaluate. Must already be started.
            suite: The task suite to run.

        Returns:
            An AgentRun with all collected results.
        """
        logger.info(
            "Starting evaluation: agent=%s suite=%s (%d steps)",
            agent.name,
            suite.suite_id,
            len(suite.steps),
        )

        run = AgentRun(
            suite_id=suite.suite_id,
            agent_type=type(agent).__name__.lower().replace("agent", ""),
            config={
                "agent_name": agent.name,
                "pass_threshold": self.config.pass_threshold,
                "scoring_method": self.config.scoring_method,
            },
        )

        # Start the agent if not already started
        await agent.start()

        steps_to_run = suite.steps[: self.config.max_steps_per_run]
        run_start = time.monotonic()

        for i, step in enumerate(steps_to_run):
            # Check overall timeout
            elapsed = time.monotonic() - run_start
            if elapsed >= self.config.timeout_seconds:
                logger.warning(
                    "Overall timeout reached after %d steps (%.1fs)",
                    i,
                    elapsed,
                )
                run.status = AgentStatus.TIMEOUT
                break

            logger.debug(
                "Step %d/%d: id=%s type=%s",
                i + 1,
                len(steps_to_run),
                step.id,
                step.step_type.value,
            )

            try:
                result = await agent.execute_step(step)
                # If llm_judge scoring is enabled, re-score the output
                # using the LLM judge instead of exact match.
                if self.config.scoring_method == "llm_judge":
                    judge = self._get_judge_evaluator()
                    result = await judge.score_step(step, result.agent_output)
                run.results.append(result)
                logger.debug(
                    "  result: success=%s score=%.2f",
                    result.success,
                    result.score,
                )
            except Exception as exc:
                logger.error("  step %s failed with unexpected error: %s", step.id, exc)
                run.results.append(
                    AgentResult(
                        step_id=step.id,
                        agent_output="",
                        success=False,
                        score=0.0,
                        error=f"evaluator_error: {exc}",
                    )
                )
        else:
            # Loop completed without break
            run.status = AgentStatus.COMPLETED

        run.completed_at = __import__("datetime").datetime.now(
            __import__("datetime").UTC
        )

        elapsed = time.monotonic() - run_start
        logger.info(
            "Evaluation complete: %d steps, %.1fs, status=%s",
            len(run.results),
            elapsed,
            run.status.value,
        )

        return run

    def compute_summary(self, run: AgentRun, suite: TaskSuite) -> ScoringSummary:
        """Compute a scoring summary for a completed run.

        Args:
            run: The completed AgentRun.
            suite: The TaskSuite that was run.

        Returns:
            A ScoringSummary with aggregated metrics.
        """
        total_steps = min(len(suite.steps), self.config.max_steps_per_run)
        completed_steps = len(run.results)
        passed_steps = sum(
            1 for r in run.results if r.score >= self.config.pass_threshold
        )

        if completed_steps > 0:
            mean_score = sum(r.score for r in run.results) / completed_steps
        else:
            mean_score = 0.0

        pass_rate = passed_steps / completed_steps if completed_steps > 0 else 0.0
        efficiency = completed_steps / total_steps if total_steps > 0 else 0.0

        # Trajectory score: weighted combination of pass rate and efficiency
        # For now, trajectory = mean_score * 0.6 + pass_rate * 0.2 + efficiency * 0.2
        trajectory_score = (
            mean_score * 0.6 + pass_rate * 0.2 + efficiency * 0.2
        )

        return ScoringSummary(
            run_id=run.run_id,
            suite_id=suite.suite_id,
            total_steps=total_steps,
            completed_steps=completed_steps,
            passed_steps=passed_steps,
            mean_score=round(mean_score, 4),
            pass_rate=round(pass_rate, 4),
            efficiency=round(efficiency, 4),
            trajectory_score=round(trajectory_score, 4),
        )
