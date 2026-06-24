"""Pydantic data models for agent evaluation in eval-harness.

Defines the core data structures for the agent evaluation system:
- TaskStep: A single task for an agent to perform.
- TaskSuite: A collection of related TaskSteps.
- AgentResult: The outcome of a single step execution.
- AgentRun: A complete run of an agent against a task suite.
- AgentCapability: Enum of built-in agent capabilities.
- TaskStepType: Enum of task categories.
- AgentStatus: Lifecycle status of an agent run.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ── Enums ────────────────────────────────────────────────────────────────────────


class TaskStepType(StrEnum):
    """Category/type of task step."""

    ECHO = "echo"
    MATH = "math"
    FILE_READ = "file_read"
    STRING_REVERSAL = "string_reversal"
    MULTI_STEP = "multi_step"


class AgentCapability(StrEnum):
    """Built-in agent capabilities for matching with task suites."""

    ECHO = "echo"
    MATH = "math"
    FILE_READ = "file_read"
    STRING_REVERSAL = "string_reversal"
    MULTI_STEP = "multi_step"


class AgentStatus(StrEnum):
    """Lifecycle status of an agent run."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


# ── TaskStep ─────────────────────────────────────────────────────────────────────


class TaskStep(BaseModel):
    """A single task for an agent to perform.

    Attributes:
        id: Unique step identifier within the suite.
        prompt: The instruction/input to give the agent.
        expected_output: The expected correct output (used for scoring).
        step_type: Category of task (default: echo).
        timeout_seconds: Maximum time allowed for the agent to respond.
        metadata: Arbitrary metadata (e.g., file paths, parameters).
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    prompt: str
    expected_output: str = ""
    step_type: TaskStepType = TaskStepType.ECHO
    timeout_seconds: float = 60.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timeout_seconds")
    @classmethod
    def _validate_timeout(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {v}")
        return v


# ── TaskSuite ────────────────────────────────────────────────────────────────────


class TaskSuite(BaseModel):
    """A collection of related task steps forming a test suite.

    Attributes:
        suite_id: Unique suite identifier.
        name: Human-readable suite name.
        description: What this suite tests.
        steps: Ordered list of TaskSteps.
        metadata: Arbitrary suite-level metadata.
    """

    model_config = ConfigDict(extra="ignore")

    suite_id: str
    name: str
    description: str = ""
    steps: list[TaskStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── AgentResult ──────────────────────────────────────────────────────────────────


class AgentResult(BaseModel):
    """The outcome of a single step execution by an agent.

    Attributes:
        step_id: The TaskStep.id this result corresponds to.
        agent_output: The raw output produced by the agent.
        success: Whether the step was completed successfully.
        score: Numeric score in [0.0, 1.0].
        error: Error message if the step failed.
        duration_seconds: Wall-clock time taken for this step.
        tokens_used: Optional token count for token-based agents.
    """

    model_config = ConfigDict(extra="ignore")

    step_id: str
    agent_output: str
    success: bool = False
    score: float = 0.0
    error: str | None = None
    duration_seconds: float = 0.0
    tokens_used: int | None = None

    @field_validator("score")
    @classmethod
    def _validate_score(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"score must be in [0.0, 1.0], got {v}")
        return v


# ── AgentRun ─────────────────────────────────────────────────────────────────────


class AgentRun(BaseModel):
    """A complete run of an agent against a task suite.

    Attributes:
        run_id: Unique run identifier.
        suite_id: The TaskSuite being run.
        agent_type: The agent adapter type (e.g., 'subprocess', 'python').
        status: Current lifecycle status.
        results: List of AgentResults collected so far.
        created_at: UTC creation time.
        completed_at: UTC completion time (if finished).
        config: Arbitrary run configuration.
    """

    model_config = ConfigDict(extra="ignore")

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    suite_id: str
    agent_type: str
    status: AgentStatus = AgentStatus.RUNNING
    results: list[AgentResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    config: dict[str, Any] = Field(default_factory=dict)


# ── ScoringSummary ──────────────────────────────────────────────────────────────


class ScoringSummary(BaseModel):
    """Aggregated scoring results for an agent run.

    Attributes:
        run_id: The AgentRun this summary belongs to.
        suite_id: The TaskSuite that was run.
        total_steps: Total number of steps.
        completed_steps: Number of steps the agent attempted.
        passed_steps: Number of steps scoring >= threshold.
        mean_score: Average score across all completed steps.
        pass_rate: Fraction of steps that passed.
        efficiency: Ratio of completed to total steps.
        trajectory_score: Overall trajectory quality [0, 1].
    """

    model_config = ConfigDict(extra="ignore")

    run_id: str
    suite_id: str
    total_steps: int = 0
    completed_steps: int = 0
    passed_steps: int = 0
    mean_score: float = 0.0
    pass_rate: float = 0.0
    efficiency: float = 0.0
    trajectory_score: float = 0.0
