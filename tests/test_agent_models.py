"""Tests for src/agent_models.py — Agent, TaskStep, TaskSuite, AgentResult, AgentRun models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agent_models import (
    AgentCapability,
    AgentResult,
    AgentRun,
    AgentStatus,
    TaskStep,
    TaskStepType,
    TaskSuite,
)

# ── TaskStep tests ──────────────────────────────────────────────────────────────


class TestTaskStep:
    """Tests for the TaskStep model."""

    def test_create_basic_task_step(self) -> None:
        step = TaskStep(
            id="step-1",
            prompt="Say hello",
            expected_output="hello",
        )
        assert step.id == "step-1"
        assert step.prompt == "Say hello"
        assert step.expected_output == "hello"
        assert step.step_type == TaskStepType.ECHO
        assert step.timeout_seconds == 60.0
        assert step.metadata == {}

    def test_task_step_with_type(self) -> None:
        step = TaskStep(
            id="step-2",
            prompt="Compute 2+2",
            expected_output="4",
            step_type=TaskStepType.MATH,
        )
        assert step.step_type == TaskStepType.MATH

    def test_task_step_with_metadata(self) -> None:
        step = TaskStep(
            id="step-3",
            prompt="Read file",
            expected_output="content",
            step_type=TaskStepType.FILE_READ,
            metadata={"path": "/tmp/test.txt"},
        )
        assert step.metadata["path"] == "/tmp/test.txt"

    def test_task_step_id_required(self) -> None:
        with pytest.raises(ValidationError):
            TaskStep(prompt="no id", expected_output="x")

    def test_task_step_prompt_required(self) -> None:
        with pytest.raises(ValidationError):
            TaskStep(id="s1", expected_output="x")

    def test_task_step_timeout_validation(self) -> None:
        with pytest.raises(ValidationError):
            TaskStep(id="s1", prompt="x", expected_output="y", timeout_seconds=0)

        with pytest.raises(ValidationError):
            TaskStep(id="s1", prompt="x", expected_output="y", timeout_seconds=-1)

    def test_task_step_model_dump(self) -> None:
        step = TaskStep(id="s1", prompt="hello", expected_output="world")
        data = step.model_dump()
        assert data["id"] == "s1"
        assert data["prompt"] == "hello"
        assert data["step_type"] == "echo"


# ── TaskSuite tests ─────────────────────────────────────────────────────────────


class TestTaskSuite:
    """Tests for the TaskSuite model."""

    def test_create_task_suite(self) -> None:
        suite = TaskSuite(
            suite_id="echo-v1",
            name="Echo Tests",
            description="Basic echo tests",
            steps=[
                TaskStep(id="e1", prompt="Say hello", expected_output="hello"),
                TaskStep(id="e2", prompt="Say world", expected_output="world"),
            ],
        )
        assert suite.suite_id == "echo-v1"
        assert suite.name == "Echo Tests"
        assert len(suite.steps) == 2

    def test_task_suite_id_required(self) -> None:
        with pytest.raises(ValidationError):
            TaskSuite(name="No ID", steps=[])

    def test_task_suite_name_required(self) -> None:
        with pytest.raises(ValidationError):
            TaskSuite(suite_id="x", steps=[])

    def test_task_suite_empty_steps_allowed(self) -> None:
        suite = TaskSuite(suite_id="empty", name="Empty", steps=[])
        assert len(suite.steps) == 0

    def test_task_suite_model_dump(self) -> None:
        suite = TaskSuite(
            suite_id="test",
            name="Test",
            steps=[TaskStep(id="s1", prompt="p", expected_output="o")],
        )
        data = suite.model_dump()
        assert data["suite_id"] == "test"
        assert len(data["steps"]) == 1


# ── AgentResult tests ───────────────────────────────────────────────────────────


class TestAgentResult:
    """Tests for the AgentResult model."""

    def test_create_agent_result(self) -> None:
        result = AgentResult(
            step_id="step-1",
            agent_output="hello",
            success=True,
            score=1.0,
        )
        assert result.step_id == "step-1"
        assert result.agent_output == "hello"
        assert result.success is True
        assert result.score == 1.0
        assert result.error is None

    def test_agent_result_score_validation(self) -> None:
        with pytest.raises(ValidationError):
            AgentResult(step_id="s1", agent_output="x", success=True, score=1.5)

        with pytest.raises(ValidationError):
            AgentResult(step_id="s1", agent_output="x", success=True, score=-0.1)

    def test_agent_result_with_error(self) -> None:
        result = AgentResult(
            step_id="s1",
            agent_output="",
            success=False,
            score=0.0,
            error="timeout",
        )
        assert result.error == "timeout"
        assert result.success is False

    def test_agent_result_step_id_required(self) -> None:
        with pytest.raises(ValidationError):
            AgentResult(agent_output="x", success=True, score=0.5)


# ── AgentRun tests ──────────────────────────────────────────────────────────────


class TestAgentRun:
    """Tests for the AgentRun model."""

    def test_create_agent_run(self) -> None:
        run = AgentRun(
            suite_id="echo-v1",
            agent_type="subprocess",
        )
        assert run.suite_id == "echo-v1"
        assert run.agent_type == "subprocess"
        assert run.status == AgentStatus.RUNNING
        assert run.results == []

    def test_agent_run_status_transitions(self) -> None:
        run = AgentRun(suite_id="test", agent_type="python")
        assert run.status == AgentStatus.RUNNING
        run.status = AgentStatus.COMPLETED
        assert run.status == AgentStatus.COMPLETED

    def test_agent_run_with_results(self) -> None:
        run = AgentRun(
            suite_id="test",
            agent_type="python",
            results=[
                AgentResult(step_id="s1", agent_output="ok", success=True, score=1.0),
            ],
        )
        assert len(run.results) == 1
        assert run.results[0].success is True

    def test_agent_run_model_dump(self) -> None:
        run = AgentRun(suite_id="test", agent_type="python")
        data = run.model_dump()
        assert data["suite_id"] == "test"
        assert data["status"] == "running"


# ── AgentCapability tests ──────────────────────────────────────────────────────


class TestAgentCapability:
    """Tests for the AgentCapability enum."""

    def test_capability_values(self) -> None:
        assert AgentCapability.ECHO.value == "echo"
        assert AgentCapability.MATH.value == "math"
        assert AgentCapability.FILE_READ.value == "file_read"
        assert AgentCapability.STRING_REVERSAL.value == "string_reversal"
        assert AgentCapability.MULTI_STEP.value == "multi_step"

    def test_capability_from_value(self) -> None:
        assert AgentCapability("echo") == AgentCapability.ECHO
        assert AgentCapability("math") == AgentCapability.MATH


# ── TaskStepType tests ─────────────────────────────────────────────────────────


class TestTaskStepType:
    """Tests for the TaskStepType enum."""

    def test_step_type_values(self) -> None:
        assert TaskStepType.ECHO.value == "echo"
        assert TaskStepType.MATH.value == "math"
        assert TaskStepType.FILE_READ.value == "file_read"
        assert TaskStepType.STRING_REVERSAL.value == "string_reversal"
        assert TaskStepType.MULTI_STEP.value == "multi_step"

    def test_step_type_from_value(self) -> None:
        assert TaskStepType("echo") == TaskStepType.ECHO
        assert TaskStepType("math") == TaskStepType.MATH


# ── AgentStatus tests ───────────────────────────────────────────────────────────


class TestAgentStatus:
    """Tests for the AgentStatus enum."""

    def test_status_values(self) -> None:
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.COMPLETED.value == "completed"
        assert AgentStatus.FAILED.value == "failed"
        assert AgentStatus.TIMEOUT.value == "timeout"
