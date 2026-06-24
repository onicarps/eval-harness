# Phase 2B Implementation Plan — Agent Evaluation

> **Created:** June 24 2026
> **Profile:** eval-harness (build)
> **Status:** Research complete. Ready for execution.
> **Supersedes:** PLAN_PHASE2.md Phase 2B section (this document is the authoritative spec)

---

## Table of Contents

1. [Overview & Research Summary](#overview--research-summary)
2. [Agent Interface Design](#agent-interface-design)
3. [Task Suite Format](#task-suite-format)
4. [Environment Adapters](#environment-adapters)
5. [Trajectory Scoring Approach](#trajectory-scoring-approach)
6. [Database Schema Changes (Migration v3)](#database-schema-changes-migration-v3)
7. [CLI Command Structure](#cli-command-structure)
8. [Implementation Order](#implementation-order)
9. [Test Strategy](#test-strategy)
10. [Edge Cases](#edge-cases)
11. [Success Criteria](#success-criteria)

---

## Overview & Research Summary

Phase 2B transforms eval-harness from a static-output evaluator into a platform for evaluating **agent behavior in environments**. Instead of scoring a single input→output pair, we now observe an agent's trajectory through a multi-step task: observation → action → observation → action → ... → outcome.

### Research: Existing Agent Evaluation Frameworks

**SWE-bench (Jimenez et al., 2024)**
- **Task format:** Each task is a GitHub issue + a Docker container with the repository at the base commit.
- **Agent interface:** Patches (diffs) are the agent's output. The agent reads the issue, explores the repo, and submits a patch.
- **Scoring:** Patch-based. Test suite passes = pass. Binary outcome per task, aggregate as pass rate.
- **Environment:** Docker containers with pre-built repositories. Heavy isolation.
- **Key pattern:** Static test suite as ground truth; no LLM-as-judge needed for scoring itself.

**GAIA (Mialon et al., 2023)**
- **Task format:** JSON/YAML files with questions, optional attachments, and expected answers.
- **Agent interface:** Tool-augmented agent that can browse the web, read files, run code.
- **Scoring:** Exact match + human evaluation. Binary (correct/incorrect).
- **Environment:** "GAIA Benchmark Server" with configurable tools (web search, file reader, calculator).
- **Key pattern:** Tiered difficulty (Level 1, 2, 3). Expected-answer-based scoring.

**AgentBench (Liu et al., 2023)**
- **Task format:** JSON tasks across 8 environments (Web browsing, knowledge graphs, web shopping, digital card, lateral thinking, house-holding, Web browser, coding).
- **Agent interface:** agent.action(environment_state) → (action, reward). Step-by-step.
- **Scoring:** Per-environment reward functions (mostly rule-based, 0-1 score per task).
- **Environment:** Simulated APIs (mock databases, mock shopping sites). Self-contained.
- **Key pattern:** Multiple distinct environment types; incremental development. Each environment has its own `reset()` and `step(action)` implementation.

**Inspect AI (UK AISI, 2024)**
- **Task format:** YAML-based task definitions in "suites" — each suite contains tasks with shared tools/environment config.
- **Agent interface:** Solver-defined — a "solver" is a Python async function that takes a `TaskState` (prompt + attachments) and returns an answer via grader interaction.
- **Scoring:** "Graders" defined per task — can be programmatic (assert) or LLM-as-judge. Supports partial credit.
- **Environment:** Abstracted via "sandboxes" (Docker or local). Minimum environment requirements per task.
- **Key pattern:** Composable graders (programmatic + human + LLM judge). Task difficulty/size/time import. `srands` for sandbox resolution.

### Patterns Adopted for eval-harness Phase 2B

| Aspect | Pattern From | Our Adoption |
|--------|-------------|--------------|
| Task format | GAIA, Inspect | YAML task definitions with fields for prompt, environment, scoring |
| Agent interface | AgentBench | `Agent` ABC with `start()`, `act(obs)`, `stop()` lifecycle |
| Environment | AgentBench | `Environment` ABC with `reset()`, `step(action)`, `get_observation()` |
| Scoring | SWE-bench + Inspect | Programmatic graders per task + optional LLM-as-judge for trajectory quality |
| Task suites | GAIA levels | Named suites (smoke, coding, browsing) ordered by difficulty |
| Timeout handling | AgentBench | Per-step + per-trajectory configurable timeouts |
| Database | Our existing approach | New agent-specific tables, no changes to existing eval tables |

---

## Agent Interface Design

### File: `src/agent.py`

```python
"""Agent interface — abstract base and built-in implementations for Phase 2B.

Provides:
- `Agent` ABC: the contract all agents must fulfill.
- `SubprocessAgent` CLI subprocess agent: communicates via stdin/stdout JSON.
- `PythonAgent` in-process Python agent: calls a user-defined async function.
- `AgentConfig`: declarative agent instantiation from task suite or CLI.
"""

from __future__ import annotations

import json
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AgentConfig:
    """Declarative agent configuration.
    
    Attributes:
        agent_type: 'subprocess' or 'python' — selects implementation.
        command: CLI command for SubprocessAgent (e.g. 'claude --print --prompt-file -').
        module_path: Import path for PythonAgent (e.g. 'my_agent.evaluate').
        function_name: Function to call (default: 'solve').
        timeout_per_step: Seconds before a single step times out (default: 30).
        timeout_total: Seconds before an entire trajectory times out (default: 300).
        max_steps: Hard cap on agent turns (default: 50).
        extra_args: Additional kwargs passed to the agent constructor.
    """
    agent_type: str = "subprocess" | "python"
    command: str | None = None
    module_path: str | None = None
    function_name: str = "solve"
    timeout_per_step: float = 30.0
    timeout_total: float = 300.0
    max_steps: int = 50
    extra_args: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Abstract base class for agents evaluated by eval-harness.
    
    Lifecycle:
        1. `start()` — called once before the first task.
        2. `act(observation)` — called repeatedly with environment observations.
        3. `stop()` — called once after the task completes or times out.
    
    Implementations must be async and handle cancellation gracefully.
    """

    @abstractmethod
    async def start(self) -> None:
        """Initialize the agent (e.g. spawn subprocess, import module)."""
        ...

    @abstractmethod
    async def act(self, observation: str) -> str:
        """Given an observation from the environment, return the agent's action.
        
        Args:
            observation: Text observation from the environment.
        
        Returns:
            Action string to be submitted to the environment.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Clean up resources (e.g. kill subprocess, close connections)."""
        ...


class SubprocessAgent(Agent):
    """Agent that runs as a CLI subprocess, communicating via stdin/stdout JSON.
    
    Protocol:
        - eval-harness sends JSON: {"type": "observation", "content": "..."}
        - agent responds with JSON: {"type": "action", "content": "..."}
        - Either side can send {"type": "done"} to end the interaction.
    
    The subprocess is expected to read from stdin and write to stdout.
    Each line is a separate JSON message (newline-delimited).
    """

    def __init__(
        self,
        command: str,
        timeout_per_step: float = 30.0,
        timeout_total: float = 300.0,
        max_steps: int = 50,
        **kwargs: Any,
    ) -> None:
        ...

    async def start(self) -> None:
        """Spawn the subprocess with stdin/stdout pipes."""
        ...

    async def act(self, observation: str) -> str:
        """Send observation to subprocess, return its action."""
        ...

    async def stop(self) -> None:
        """Terminate the subprocess gracefully."""
        ...


class PythonAgent(Agent):
    """Agent that calls a user-defined Python function in-process.
    
    The function signature must be:
        async def solve(observation: str, step_number: int, history: list[dict]) -> str
    
    The function receives the current observation, the step number (0-indexed),
    and the full history of (observation, action) pairs so far.
    """

    def __init__(
        self,
        module_path: str,
        function_name: str = "solve",
        timeout_per_step: float = 30.0,
        timeout_total: float = 300.0,
        max_steps: int = 50,
        **kwargs: Any,
    ) -> None:
        ...

    async def start(self) -> None:
        """Import the target module and resolve the function."""
        ...

    async def act(self, observation: str) -> str:
        """Call the user function with observation and history."""
        ...

    async def stop(self) -> None:
        """No-op for in-process agent."""
        ...


def load_agent(config: AgentConfig) -> Agent:
    """Factory: instantiate the correct Agent subclass from config.
    
    Args:
        config: Agent configuration.
    
    Returns:
        An initialized (but not started) Agent.
    
    Raises:
        ValueError: If agent_type is unknown or required fields are missing.
    """
    if config.agent_type == "subprocess":
        if not config.command:
            raise ValueError("SubprocessAgent requires 'command'")
        return SubprocessAgent(
            command=config.command,
            timeout_per_step=config.timeout_per_step,
            timeout_total=config.timeout_total,
            max_steps=config.max_steps,
            **config.extra_args,
        )
    if config.agent_type == "python":
        if not config.module_path:
            raise ValueError("PythonAgent requires 'module_path'")
        return PythonAgent(
            module_path=config.module_path,
            function_name=config.function_name,
            timeout_per_step=config.timeout_per_step,
            timeout_total=config.timeout_total,
            max_steps=config.max_steps,
            **config.extra_args,
        )
    raise ValueError(f"unknown agent_type: {config.agent_type}")
```

### Key Design Decisions

1. **Async by default:** All agent methods are async to support concurrent evaluation and timeout handling.
2. **Subprocess protocol:** Newline-delimited JSON (NDJSON) over stdin/stdout — simple, language-agnostic, debuggable.
3. **Python agent for testing:** In-process agent avoids subprocess overhead for unit tests and simple agents.
4. **Factory pattern:** `load_agent()` decouples configuration from instantiation, enabling YAML-defined agents in task suites.

---

## Task Suite Format

### File: `src/tasks.py`

```python
"""Task suite loading and validation for agent evaluation.

Provides:
- `TaskDefinition`: A single task with its grader and environment config.
- `TaskSuite`: A named collection of tasks with shared configuration.
- `load_suite(path)`: Load a task suite from a YAML file.
- `list_builtins()`: List available built-in task suites.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TaskDefinition:
    """A single evaluation task for an agent.
    
    Attributes:
        name: Unique task identifier within the suite (e.g. 'reverse-string').
        prompt: Initial instruction shown to the agent.
        environment: Environment adapter name (e.g. 'python-repl', 'mock').
        grader: Grader configuration — see GraderConfig.
        max_steps: Maximum agent turns for this task.
        metadata: Arbitrary task metadata (tags, difficulty, source).
    """
    name: str
    prompt: str
    environment: str = "mock"
    grader: GraderConfig = field(default_factory=GraderConfig)
    max_steps: int = 50
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraderConfig:
    """Configuration for scoring a task outcome.
    
    Attributes:
        type: 'programmatic' (assert on final state) or 'llm-judge' (LLM scores trajectory).
        expected_output: For programmatic: the expected final observation or state assertion.
        rubric: For llm-judge: the rubric prompt template to use.
        threshold: Score threshold for success (default: 0.7).
    """
    type: str = "programmatic" | "llm-judge"
    expected_output: str | None = None
    rubric: str | None = None
    threshold: float = 0.7


@dataclass
class TaskSuite:
    """A named collection of tasks sharing an agent configuration.
    
    Attributes:
        name: Suite name (e.g. 'smoke', 'coding', 'browsing').
        description: Human-readable description.
        agent_config: Agent configuration for this suite.
        tasks: List of task definitions.
    """
    name: str
    description: str = ""
    agent_config: AgentConfig | None = None
    tasks: list[TaskDefinition] = field(default_factory=list)


def load_suite(path: str | Path) -> TaskSuite:
    """Load a task suite from a YAML file.
    
    Expected YAML format:
        name: smoke
        description: Quick smoke tests for agent evaluation
        agent:
            type: subprocess
            command: python -m my_agent
        tasks:
            - name: reverse-string
              prompt: "Reverse the string 'hello'"
              environment: mock
              grader:
                type: programmatic
                expected_output: "olleh"
              max_steps: 5
    
    Args:
        path: Path to the YAML file.
    
    Returns:
        Validated TaskSuite.
    
    Raises:
        ValueError: If the YAML is malformed or missing required fields.
    """
    ...


def list_builtins() -> list[str]:
    """Return names of built-in task suites."""
    ...


def get_builtin_path(name: str) -> Path:
    """Return the filesystem path for a built-in suite.
    
    Built-in suites live in src/suites/<name>.yaml.
    """
    ...
```

### Built-in Task Suites (3 suites, 15 tasks)

**Suite 1: `smoke` (5 tasks)** — Minimal tasks validating the pipeline works.

| Task | Environment | Grader | Description |
|------|-------------|---------|-------------|
| `echo` | mock | programmatic | Agent echoes back the observation. |
| `reverse-string` | mock | programmatic | Agent reverses a string. |
| `count-words` | mock | programmatic | Agent counts words in a sentence. |
| `add-numbers` | mock | programmatic | Agent adds two numbers from the prompt. |
| `fail-always` | mock | programmatic | Task designed to fail (tests failure recording). |

**Suite 2: `coding` (5 tasks)** — Python REPL-based coding tasks.

| Task | Environment | Grider | Description |
|------|-------------|---------|-------------|
| `fibonacci` | python-repl | programmatic | Define a function that returns the nth Fibonacci number. |
| `palindrome-check` | python-repl | programmatic | Write a function checking if a string is a palindrome. |
| `list-sort` | python-repl | programmatic | Sort a list of numbers in the REPL. |
| `dict-merge` | python-repl | programmatic | Merge two dictionaries. |
| `fizzbuzz` | python-repl | programmatic | Print FizzBuzz for 1-20. |

**Suite 3: `browsing` (5 tasks)** — Simulated web/API interaction tasks.

| Task | Environment | Grader | Description |
|------|-------------|---------|-------------|
| `find-title` | mock | programmatic | Navigate a mock page tree to find a title. |
| `extract-price` | mock | programmatic | Extract a price from a mock product page. |
| `multi-step-search` | mock | programmatic | Search a mock knowledge base with 2 queries. |
| `form-fill` | mock | programmatic | Fill and submit a mock form. |
| `llm-judge-trajectory` | mock | llm-judge | Task scored by LLM judge (tests judge integration). |

### YAML Example: `src/suites/smoke.yaml`

```yaml
name: smoke
description: Quick smoke tests validating the agent evaluation pipeline
agent:
  type: subprocess
  command: cat
  timeout_per_step: 5.0
  timeout_total: 30.0
  max_steps: 10

tasks:
  - name: echo
    prompt: "Repeat the observation exactly: Hello World"
    environment: mock
    grader:
      type: programmatic
      expected_output: "Hello World"
    max_steps: 3

  - name: reverse-string
    prompt: "Reverse the string: abcde"
    environment: mock
    grader:
      type: programmatic
      expected_output: "edcba"
    max_steps: 5

  - name: count-words
    prompt: "Count the words in: the quick brown fox jumps"
    environment: mock
    grader:
      type: programmatic
      expected_output: "5"
    max_steps: 5

  - name: add-numbers
    prompt: "What is 7 + 3? Reply with just the number."
    environment: mock
    grader:
      type: programmatic
      expected_output: "10"
    max_steps: 5

  - name: fail-always
    prompt: "This task will always fail. Do not produce 'expected'."
    environment: mock
    grader:
      type: programmatic
      expected_output: "this-will-never-match"
    max_steps: 5
```

---

## Environment Adapters

### File: `src/environments.py`

```python
"""Environment adapters for agent evaluation.

Provides:
- `Environment` ABC: the contract all environments must fulfill.
- `MockEnvironment`: Deterministic mock environment for testing.
- `PythonReplEnvironment`: Sandboxed Python REPL via subprocess.
- `load_environment(name, config)`: Factory function.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from typing import Any


class Environment(ABC):
    """Abstract base class for agent interaction environments.
    
    Lifecycle:
        1. `reset()` — initialize the environment, return first observation.
        2. `step(action)` — execute agent action, return (observation, done, info).
        3. `cleanup()` — release resources (called once at end).
    
    The environment tracks its own state (e.g. REPL variables, mock page content).
    """

    @abstractmethod
    def reset(self) -> str:
        """Reset the environment and return the initial observation.
        
        Returns:
            Initial observation text presented to the agent.
        """
        ...

    @abstractmethod
    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        """Execute one agent action in the environment.
        
        Args:
            action: The agent's action string.
        
        Returns:
            Tuple of (observation, done_flag, info_dict).
            - observation: Next text observation for the agent.
            - done_flag: True if the task is complete (success or failure).
            - info_dict: Arbitrary metadata (e.g. error messages, step count).
        """
        ...

    def cleanup(self) -> None:
        """Release any resources. Override if needed."""
        pass


class MockEnvironment(Environment):
    """Deterministic mock environment for testing and simple tasks.
    
    Behavior is driven by a `responses` dict mapping actions to observations.
    The environment maintains a step counter and checks against expected actions.
    
    Config:
        responses: Dict[str, str] — maps agent actions to observations.
        expected_actions: list[str] — if set, the agent must produce these in order.
        terminal_action: str — action that ends the environment (default: "done").
    """

    def __init__(
        self,
        responses: dict[str, str] | None = None,
        expected_actions: list[str] | None = None,
        terminal_action: str = "done",
        **kwargs: Any,
    ) -> None:
        ...

    def reset(self) -> str:
        """Return the initial observation (or empty string)."""
        ...

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        """Look up action in responses, return mapped observation."""
        ...


class PythonReplEnvironment(Environment):
    """Python REPL environment running in a subprocess.
    
    The agent sends Python code lines. The environment executes them in a
    persistent Python subprocess and returns the output (stdout/stderr).
    
    Config:
        python_path: Path to Python interpreter (default: sys.executable).
        preload_code: Code to run before the agent starts (e.g. imports).
        timeout: Per-execution timeout in seconds (default: 5).
    """

    def __init__(
        self,
        python_path: str = "python3",
        preload_code: str = "",
        timeout: float = 5.0,
        **kwargs: Any,
    ) -> None:
        ...

    def reset(self) -> str:
        """Start the Python subprocess and return initial observation."""
        ...

    def step(self, action: str) -> tuple[str, bool, dict[str, Any]]:
        """Execute action as Python code, return output."""
        ...

    def cleanup(self) -> None:
        """Kill the Python subprocess."""
        ...


def load_environment(name: str, config: dict[str, Any] | None = None) -> Environment:
    """Factory: instantiate an Environment by name.
    
    Args:
        name: Environment name ('mock', 'python-repl').
        config: Configuration dict passed to the constructor.
    
    Returns:
        An uninitialized Environment (call reset() to start).
    
    Raises:
        ValueError: If the environment name is unknown.
    """
    config = config or {}
    if name == "mock":
        return MockEnvironment(**config)
    if name == "python-repl":
        return PythonReplEnvironment(**config)
    raise ValueError(f"unknown environment: {name!r}")
```

### Key Design Decisions

1. **Synchronous environments:** Unlike the async Agent interface, environments are synchronous. This simplifies the mock and REPL implementations. The orchestration layer handles async wrapping.
2. **Config-driven:** Each environment takes a config dict, enabling YAML-defined environments in task suites.
3. **Cleanup hook:** Explicit `cleanup()` ensures subprocesses are killed even on exceptions.
4. **Python subprocess for REPL:** Uses `python3 -u` with `-c` or stdin piping for execution. No `eval()` in the main process.

---

## Trajectory Scoring Approach

### File: `src/scoring.py`

```python
"""Trajectory scoring for agent evaluation.

Provides:
- `Trajectory`: Data class for a full agent trajectory.
- `TrajectoryScorer`: Scores trajectories using programmatic graders and optional LLM judges.
- `score_trajectory(trajectory, task) -> TrajectoryResult`: Main entry point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.models import EvalSummary
from src.tasks import GraderConfig, TaskDefinition


@dataclass
class TrajectoryStep:
    """A single step in an agent trajectory.
    
    Attributes:
        step_number: 0-indexed step number.
        observation: Observation from the environment before the action.
        action: Agent's action.
        reward: Optional per-step reward (computed by environment or grader).
    """
    step_number: int
    observation: str
    action: str
    reward: float | None = None


@dataclass
class Trajectory:
    """A complete agent trajectory for one task.
    
    Attributes:
        task_name: Name of the task being executed.
        steps: Ordered list of trajectory steps.
        final_observation: The last observation from the environment.
        done: Whether the environment signaled completion.
        info: Additional metadata from the environment.
        total_time_seconds: Wall-clock time for the trajectory.
    """
    task_name: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    final_observation: str = ""
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    total_time_seconds: float = 0.0

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class TrajectoryResult:
    """Scoring result for a trajectory.
    
    Attributes:
        task_name: Name of the task.
        success: Whether the trajectory passed the grader.
        score: Numeric score in [0.0, 1.0].
        reasoning: Explanation of the score (especially for LLM-judged trajectories).
        step_count: Number of steps taken.
        total_reward: Sum of per-step rewards (if available).
        grader_type: Which grader was used.
    """
    task_name: str
    success: bool
    score: float
    reasoning: str = ""
    step_count: int = 0
    total_reward: float = 0.0
    grader_type: str = "programmatic"


def score_trajectory(
    trajectory: Trajectory,
    task: TaskDefinition,
    judge_api_key: str | None = None,
) -> TrajectoryResult:
    """Score a trajectory against its task's grader.
    
    Args:
        trajectory: The completed trajectory.
        task: Task definition with grader config.
        judge_api_key: API key for LLM judge (required if grader type is 'llm-judge').
    
    Returns:
        TrajectoryResult with score and reasoning.
    """
    grader_type = task.grader.type
    if grader_type == "programmatic":
        return _score_programmatic(trajectory, task)
    if grader_type == "llm-judge":
        return _score_llm_judge(trajectory, task, judge_api_key)
    raise ValueError(f"unknown grader type: {grader_type}")


def _score_programmatic(
    trajectory: Trajectory,
    task: TaskDefinition,
) -> TrajectoryResult:
    """Score using exact match against expected output."""
    expected = task.grader.expected_output or ""
    actual = trajectory.final_observation.strip()
    success = actual == expected
    score = 1.0 if success else 0.0
    reasoning = (
        f"Expected: {expected!r}, Got: {actual!r}"
    )
    return TrajectoryResult(
        task_name=task.name,
        success=success,
        score=score,
        reasoning=reasoning,
        step_count=trajectory.step_count,
        grader_type="programmatic",
    )


async def _score_llm_judge(
    trajectory: Trajectory,
    task: TaskDefinition,
    api_key: str | None,
) -> TrajectoryResult:
    """Score using an LLM judge that evaluates the full trajectory."""
    # Build a prompt showing the task prompt, all steps, and the final outcome
    # Send to judge model, parse score from response
    # Uses existing LLMEvaluator infrastructure from src/evaluator.py
    ...
```

### Scoring Strategy

1. **Programmatic graders (default):** Exact match of `final_observation` against `expected_output`. Fast, deterministic, no API cost.
2. **LLM-judge graders:** For open-ended tasks, send the full trajectory to an LLM judge with a rubric. Uses the existing `LLMEvaluator` infrastructure.
3. **Per-step rewards:** Future enhancement — environments can emit per-step rewards for reinforcement learning scenarios.
4. **Partial credit:** LLM-judge graders return a float score in [0, 1], enabling partial credit for partially correct trajectories.

---

## Database Schema Changes (Migration v3)

### Migration SQL

```sql
-- Migration v3: Agent evaluation tables
-- No changes to existing tables — additive only.

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    agent_type TEXT NOT NULL DEFAULT 'subprocess',
    agent_command TEXT,
    task_suite TEXT NOT NULL,
    rubric_id TEXT DEFAULT 'agent-default-v1',
    status TEXT DEFAULT 'running',
    completed_at TEXT,
    mean_score REAL,
    pass_rate REAL,
    total_steps INTEGER DEFAULT 0,
    total_time_seconds REAL
);

CREATE TABLE IF NOT EXISTS agent_trajectories (
    trajectory_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agent_runs(run_id),
    task_name TEXT NOT NULL,
    success INTEGER NOT NULL DEFAULT 0,
    score REAL NOT NULL DEFAULT 0.0,
    step_count INTEGER NOT NULL DEFAULT 0,
    total_reward REAL DEFAULT 0.0,
    grader_type TEXT DEFAULT 'programmatic',
    reasoning TEXT DEFAULT '',
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trajectories_run ON agent_trajectories(run_id);

CREATE TABLE IF NOT EXISTS agent_steps (
    step_id TEXT PRIMARY KEY,
    trajectory_id TEXT NOT NULL REFERENCES agent_trajectories(trajectory_id),
    step_number INTEGER NOT NULL,
    observation TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL DEFAULT '',
    reward REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_steps_trajectory ON agent_steps(trajectory_id);
```

### Rollback SQL

```sql
-- Rollback v3
DROP INDEX IF EXISTS idx_steps_trajectory;
DROP TABLE IF EXISTS agent_steps;
DROP INDEX IF EXISTS idx_trajectories_run;
DROP TABLE IF EXISTS agent_trajectories;
DROP TABLE IF EXISTS agent_runs;
```

### New Models

```python
# In src/models.py, add:

class AgentRun(BaseModel):
    """A single agent evaluation run, grouping trajectories for a task suite."""
    
    model_config = ConfigDict(extra="ignore")
    
    run_id: str = Field(default_factory=_new_id)
    created_at: datetime = Field(default_factory=_utcnow)
    agent_type: str = "subprocess"
    agent_command: str | None = None
    task_suite: str
    rubric_id: str = "agent-default-v1"
    status: RunStatus = RunStatus.RUNNING
    completed_at: datetime | None = None
    mean_score: float | None = None
    pass_rate: float | None = None
    total_steps: int = 0
    total_time_seconds: float | None = None


class AgentTrajectory(BaseModel):
    """Result of evaluating one task within an agent run."""
    
    model_config = ConfigDict(extra="ignore")
    
    trajectory_id: str = Field(default_factory=_new_id)
    run_id: str
    task_name: str
    success: bool = False
    score: float = 0.0
    step_count: int = 0
    total_reward: float = 0.0
    grader_type: str = "programmatic"
    reasoning: str = ""
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class AgentStep(BaseModel):
    """A single step within an agent trajectory."""
    
    model_config = ConfigDict(extra="ignore")
    
    step_id: str = Field(default_factory=_new_id)
    trajectory_id: str
    step_number: int
    observation: str = ""
    action: str = ""
    reward: float | None = None
    created_at: datetime = Field(default_factory=_utcnow)
```

### Database Class Extensions

```python
# In src/db.py, add methods to Database class:

class Database:
    # ... existing methods unchanged ...
    
    # --- Agent evaluation methods ---
    
    def insert_agent_run(self, run: AgentRun) -> None:
        """Insert a new agent evaluation run."""
        ...
    
    def update_agent_run(self, run: AgentRun) -> None:
        """Update mutable fields of an agent run."""
        ...
    
    def get_agent_run(self, run_id: str) -> AgentRun | None:
        """Return the agent run with run_id or None."""
        ...
    
    def list_agent_runs(self, limit: int = 100) -> list[AgentRun]:
        """Return up to limit agent runs, newest first."""
        ...
    
    def insert_trajectory(self, trajectory: AgentTrajectory) -> None:
        """Insert a trajectory result."""
        ...
    
    def get_trajectories(self, run_id: str) -> list[AgentTrajectory]:
        """Return all trajectories for an agent run."""
        ...
    
    def insert_step(self, step: AgentStep) -> None:
        """Insert a single trajectory step."""
        ...
    
    def get_steps(self, trajectory_id: str) -> list[AgentStep]:
        """Return all steps for a trajectory, in order."""
        ...
    
    def export_agent_run(self, run_id: str, out_path: str | Path, fmt: str = "json") -> Path:
        """Export an agent run's full trajectory data."""
        ...
```

---

## CLI Command Structure

### New Commands

The agent evaluation adds a `agent` subcommand group with three subcommands:

```
eval-harness agent eval <suite>          # Run agent evaluation on a task suite
eval-harness agent list-suites           # List available task suites
eval-harness agent report --run-id ID    # Show detailed trajectory report
eval-harness agent export --run-id ID    # Export trajectories to JSON/CSV
eval-harness agent list-runs             # List previous agent evaluation runs
```

### Command Details

#### `eval-harness agent eval <suite>`

```python
@app.command(
    "agent",
    help="Evaluate an agent in environments using task suites.",
)
def agent_group():
    """Agent evaluation commands."""
    pass

@agent_group.command("eval", help="Run agent evaluation on a task suite.")
def agent_eval_cmd(
    suite: str = typer.Argument(
        ...,
        help="Task suite name (e.g. 'smoke', 'coding', 'browsing') or path to YAML suite file.",
    ),
    agent_command: str | None = typer.Option(
        None,
        "--agent-command",
        help="Override the suite's agent command. Uses the suite's default if omitted.",
    ),
    agent_type: str = typer.Option(
        "subprocess",
        "--agent-type",
        help="Agent type: 'subprocess' or 'python'. Overrides suite config.",
    ),
    step_timeout: float = typer.Option(
        30.0,
        "--step-timeout",
        help="Timeout per agent step in seconds.",
    ),
    trajectory_timeout: float = typer.Option(
        300.0,
        "--trajectory-timeout",
        help="Timeout per task trajectory in seconds.",
    ),
    max_steps: int = typer.Option(
        50,
        "--max-steps",
        help="Maximum steps per task.",
    ),
    concurrency: int = typer.Option(
        1,
        "--concurrency",
        help="Number of tasks to run concurrently (default: 1).",
    ),
    judge_api_key: str | None = typer.Option(
        None,
        "--judge-api-key",
        help="API key for LLM-judge graders. Defaults to OPENROUTER_API_KEY env var.",
    ),
    output: str = typer.Option(
        "table",
        "--output",
        help="Output format: 'table' or 'json'.",
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output-file",
        help="Write output to a file.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed step-by-step output.",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress progress output.",
    ),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db", help="Path to SQLite database."),
) -> None:
    """Run agent evaluation on a task suite.
    
    Loads the specified task suite, instantiates the agent and environments,
    runs each task, scores trajectories, and displays results.
    
    Exit codes:
        0 — All tasks passed.
        1 — One or more tasks failed.
        2 — Evaluator error.
    """
```

#### `eval-harness agent list-suites`

```python
@agent_group.command("list-suites", help="List available task suites.")
def agent_list_suites_cmd(
    json_out: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all available task suites (built-in and custom)."""
```

#### `eval-harness agent report --run-id`

```python
@agent_group.command("report", help="Show detailed trajectory report for an agent run.")
def agent_report_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Agent run ID."),
    trajectory_id: str | None = typer.Option(
        None,
        "--trajectory-id",
        help="Show only a specific trajectory.",
    ),
    output: str = typer.Option("table", "--output", help="Output format: 'table', 'json'."),
    output_file: Path | None = typer.Option(None, "--output-file"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
) -> None:
    """Display detailed trajectory report for a previous agent evaluation run.
    
    Shows per-step observations, actions, and rewards for each trajectory.
    Useful for debugging agent behavior.
    """
```

#### `eval-harness agent export --run-id`

```python
@agent_group.command("export", help="Export agent run trajectories to file.")
def agent_export_cmd(
    run_id: str = typer.Option(..., "--run-id", help="Agent run ID."),
    format: str = typer.Option("json", "--format", help="Export format: 'json' or 'csv'."),
    output_file: Path = typer.Option(..., "--output-file", help="Output file path."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
) -> None:
    """Export full trajectory data for an agent run."""
```

#### `eval-harness agent list-runs`

```python
@agent_group.command("list-runs", help="List previous agent evaluation runs.")
def agent_list_runs_cmd(
    limit: int = typer.Option(20, "--limit", help="Maximum runs to show."),
    json_out: bool = typer.Option(False, "--json", help="Output as JSON."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
) -> None:
    """List previous agent evaluation runs."""
```

---

## Implementation Order

### Step 1: Agent Interface (P0, Day 1-2)

**Files to create/modify:**
- `src/agent.py` (new)
- `src/models.py` (add AgentRun, AgentTrajectory, AgentStep)
- `tests/test_agent.py` (new)

**Test cases (TDD — write first):**
1. `TestAgentConfig`: Valid/invalid config construction.
2. `TestSubprocessAgent`: Start subprocess, send observation, receive action, stop.
3. `TestSubprocessAgentTimeout`: Agent hangs → timeout raised.
4. `TestSubprocessAgentCrash`: Agent process dies → graceful error.
5. `TestPythonAgent`: Import module, call function with observation + history.
6. `TestPythonAgentTimeout`: Function takes too long → timeout.
7. `TestLoadAgent`: Factory creates correct type from config.
8. `TestLoadAgentInvalid`: Unknown type raises ValueError.

### Step 2: Environment Adapters (P0, Day 2-3)

**Files to create/modify:**
- `src/environments.py` (new)
- `tests/test_environments.py` (new)

**Test cases:**
1. `TestMockEnvironment`: Reset returns initial obs, step returns mapped response.
2. `TestMockEnvironmentTerminal`: Correct terminal action sets done=True.
3. `TestMockEnvironmentUnknownAction`: Unknown action returns error observation.
4. `TestPythonReplEnvironment`: Reset starts subprocess, step executes code.
5. `TestPythonReplEnvironmentPersistentState`: Variables persist across steps.
6. `TestPythonReplEnvironmentTimeout`: Long-running code times out.
7. `TestLoadEnvironment`: Factory creates correct type.

### Step 3: Task Suite Loading (P1, Day 3-4)

**Files to create/modify:**
- `src/tasks.py` (new)
- `src/suites/smoke.yaml` (new)
- `src/suites/coding.yaml` (new)
- `src/suites/browsing.yaml` (new)
- `tests/test_tasks.py` (new)

**Test cases:**
1. `TestLoadSuite`: Load valid YAML, verify all fields.
2. `TestLoadSuiteMissingFields`: Missing name/prompt → ValueError.
3. `TestLoadSuiteInvalidGrader`: Invalid grader type → ValueError.
4. `TestListBuiltins`: Returns ['smoke', 'coding', 'browsing'].
5. `TestBuiltinSuitesValid`: All built-in YAML files load successfully.

### Step 4: Trajectory Scoring (P0, Day 4-5)

**Files to create/modify:**
- `src/scoring.py` (new)
- `tests/test_scoring.py` (new)

**Test cases:**
1. `TestProgrammaticGraderExactMatch`: Correct output → score 1.0.
2. `TestProgrammaticGraderMismatch`: Wrong output → score 0.0.
3. `TestProgrammaticGraderWhitespace`: Trailing whitespace handled.
4. `TestLLMJudgeGrader`: Mock judge API, verify score extraction.
5. `TestLLMJudgeGraderAPIError`: Judge API fails → graceful fallback.
6. `TestScoreTrajectoryIntegration`: Full trajectory → TrajectoryResult.

### Step 5: Agent Eval Core (P0, Day 5-8)

**Files to create/modify:**
- `src/agent_evaluator.py` (new) — orchestrates agent + environment + scoring
- `tests/test_agent_evaluator.py` (new)

**Core class:**

```python
# In src/agent_evaluator.py

class AgentEvaluator:
    """Orchestrates agent evaluation: runs agent on tasks, records trajectories, scores.
    
    Usage:
        evaluator = AgentEvaluator(db, agent, suite)
        run = await evaluator.run()
    """

    def __init__(
        self,
        db: Database,
        agent: Agent,
        suite: TaskSuite,
        judge_api_key: str | None = None,
        concurrency: int = 1,
    ) -> None:
        ...

    async def run(self) -> AgentRun:
        """Execute all tasks in the suite and return the completed AgentRun."""
        ...

    async def _run_single_task(
        self,
        task: TaskDefinition,
        run_id: str,
    ) -> AgentTrajectory:
        """Run one task: reset env, loop agent.act() until done or timeout."""
        ...
```

**Test cases:**
1. `TestAgentEvaluatorSingleTask`: One task, agent succeeds.
2. `TestAgentEvaluatorMultipleTasks`: 3 tasks, verify all trajectories recorded.
3. `TestAgentEvaluatorAgentHang`: Agent hangs → step timeout fires, task fails.
4. `TestAgentEvaluatorTrajectoryTimeout`: Too many steps → trajectory timeout.
5. `TestAgentEvaluatorAgentCrash`: Agent subprocess dies mid-task → failure recorded.
6. `TestAgentEvaluatorIncrementalDB`: Trajectories written to DB even if later tasks fail.
7. `TestAgentEvaluatorConcurrency`: 2 tasks run concurrently.

### Step 6: Database Migration v3 (P0, Day 8)

**Files to modify:**
- `src/db.py` (add migration v3, rollback v3, new CRUD methods)
- `tests/test_db.py` (add migration tests)

**Test cases:**
1. `TestMigrationV3`: Fresh DB creates agent tables.
2. `TestMigrationV3ExistingData`: Existing eval_runs/records untouched.
3. `TestRollbackV3`: Rollback removes agent tables, keeps eval tables.
4. `TestAgentCRUD`: Insert/get AgentRun, AgentTrajectory, AgentStep.
5. `TestExportAgentRun`: JSON export includes full trajectory data.

### Step 7: CLI Integration (P0, Day 8-9)

**Files to modify:**
- `src/cli.py` (add agent subcommand group)
- `tests/test_cli.py` (add agent command tests)

**Test cases:**
1. `TestAgentEvalCLI`: `eval-harness agent eval smoke` — end-to-end with mock agent.
2. `TestAgentEvalCLIWithFlags`: All flags respected.
3. `TestAgentListSuitesCLI`: Lists built-in suites.
4. `TestAgentReportCLI`: Shows trajectory details.
5. `TestAgentExportCLI`: Writes JSON/CSV file.
6. `TestAgentListRunsCLI`: Lists previous agent runs.
7. `TestAgentEvalExitCodes`: 0=all pass, 1=any fail, 2=error.

### Step 8: Integration & Polish (Day 9-10)

- Wire up `agent` subcommand to main CLI app.
- Update `pyproject.toml` description to mention agent evaluation.
- Update `CHANGELOG.md` with Phase 2B entry.
- Run full test suite, verify 140+ tests, 90%+ coverage.

---

## Test Strategy

### Test Organization

```
tests/
├── conftest.py                    # Existing fixtures + new agent fixtures
├── test_agent.py                  # Agent interface tests
├── test_environments.py           # Environment adapter tests
├── test_tasks.py                  # Task suite loading tests
├── test_scoring.py                # Trajectory scoring tests
├── test_agent_evaluator.py        # Agent eval orchestration tests
├── test_db.py                     # Extended with migration v3 tests
├── test_cli.py                    # Extended with agent command tests
└── suites/                        # Test-specific task suite YAMLs
    └── test-smoke.yaml
```

### New Fixtures (in `conftest.py`)

```python
@pytest.fixture()
def mock_agent():
    """Return a PythonAgent that echoes observations."""
    ...

@pytest.fixture()
def mock_suite():
    """Return a minimal TaskSuite with 2 tasks."""
    ...

@pytest.fixture()
def mock_env():
    """Return a MockEnvironment with preset responses."""
    ...
```

### Test Categories

| Category | Count | Approach |
|----------|-------|----------|
| Unit (agent) | 8 | Mock subprocess, test timeouts |
| Unit (environments) | 7 | Direct instantiation, mock subprocess |
| Unit (tasks) | 5 | YAML loading from tmp files |
| Unit (scoring) | 6 | Programmatic + mock LLM judge |
| Integration (agent_evaluator) | 7 | Full pipeline with mock agent/env |
| Integration (db migration) | 5 | Fresh DB, existing data preservation |
| Integration (CLI) | 7 | Typer CliRunner, end-to-end |
| **Total new tests** | **~45** | |

### Existing Tests

- 357 existing tests must continue passing (no regressions).
- No changes to existing test files — all new tests in new files.

### Coverage Target

- Overall: 90%+
- `src/agent.py`: 95%
- `src/environments.py`: 90%
- `src/tasks.py`: 95%
- `src/scoring.py`: 90%
- `src/agent_evaluator.py`: 85%

---

## Edge Cases

### Agent Hangs
- **Detection:** `asyncio.wait_for()` wraps each `agent.act()` call with `step_timeout`.
- **Behavior:** On timeout, the trajectory is marked as failed with reason `"step_timeout"`.
- **Config:** `--step-timeout` (default: 30s).

### Trajectory Timeout
- **Detection:** Wall-clock timer across all steps in a task.
- **Behavior:** If total time exceeds `trajectory_timeout`, the agent is stopped and the trajectory is scored as-is.
- **Config:** `--trajectory-timeout` (default: 300s).

### Agent Crash
- **Detection:** Subprocess exits unexpectedly, or Python function raises unhandled exception.
- **Behavior:** Trajectory recorded as failure with `"agent_crash"` in the error field. All completed steps are preserved.
- **Recovery:** Other tasks in the suite continue unaffected.

### Incremental DB Writes
- **Behavior:** Each trajectory is written to DB immediately after completion, not at the end of the run.
- **Benefit:** If the entire run crashes mid-way, partial results are recoverable.
- **Implementation:** `insert_trajectory()` called after each task completes.

### Environment Crash
- **Detection:** Environment subprocess dies or returns malformed response.
- **Behavior:** Current trajectory fails with `"environment_error"`. Suite continues with next task.

### Max Steps Exceeded
- **Detection:** Agent takes more than `max_steps` without the environment signaling done.
- **Behavior:** Trajectory ends, scored as incomplete (score=0, reasoning="max_steps_exceeded").

### Empty Suite
- **Behavior:** CLI prints warning, exits with code 2.

### Unknown Suite Name
- **Behavior:** CLI prints error with available suites, exits with code 2.

---

## Success Criteria

### Phase 2B (v0.3.0) Checklist

- [ ] `Agent` ABC + `SubprocessAgent` + `PythonAgent` defined and tested
- [ ] `MockEnvironment` + `PythonReplEnvironment` working
- [ ] 3 task suites (smoke, coding, browsing) with 15 total tasks
- [ ] `eval-harness agent eval smoke` works end-to-end
- [ ] `eval-harness agent eval coding` works with Python REPL
- [ ] Programmatic grader working (exact match)
- [ ] LLM-judge grader working (for trajectory quality)
- [ ] Trajectory recording to DB functional
- [ ] Agent hang/timeout handling (step + trajectory)
- [ ] Agent crash handling (subprocess death)
- [ ] Incremental DB writes (partial results recoverable)
- [ ] `agent list-suites`, `agent report`, `agent export`, `agent list-runs` working
- [ ] Migration v3 applies cleanly, rollback works
- [ ] 140+ tests total (357 existing + ~45 new), 90%+ coverage
- [ ] CI green, CHANGELOG updated

---

## File Summary

### New Files

| File | Purpose |
|------|---------|
| `src/agent.py` | Agent ABC + SubprocessAgent + PythonAgent |
| `src/environments.py` | Environment ABC + MockEnvironment + PythonReplEnvironment |
| `src/tasks.py` | Task suite YAML loading and validation |
| `src/scoring.py` | Trajectory scoring (programmatic + LLM-judge) |
| `src/agent_evaluator.py` | Orchestration: agent + environment + scoring |
| `src/suites/smoke.yaml` | Built-in smoke test suite |
| `src/suites/coding.yaml` | Built-in coding task suite |
| `src/suites/browsing.yaml` | Built-in browsing task suite |
| `tests/test_agent.py` | Agent interface tests |
| `tests/test_environments.py` | Environment adapter tests |
| `tests/test_tasks.py` | Task suite loading tests |
| `tests/test_scoring.py` | Trajectory scoring tests |
| `tests/test_agent_evaluator.py` | Agent eval orchestration tests |
| `tests/suites/test-smoke.yaml` | Test-specific minimal suite |

### Modified Files

| File | Changes |
|------|---------|
| `src/models.py` | Add AgentRun, AgentTrajectory, AgentStep models |
| `src/db.py` | Add migration v3, rollback v3, agent CRUD methods, export method |
| `src/cli.py` | Add `agent` subcommand group (eval, list-suites, report, export, list-runs) |
| `tests/conftest.py` | Add agent-related fixtures |
| `tests/test_db.py` | Add migration v3 tests |
| `tests/test_cli.py` | Add agent command tests |
| `pyproject.toml` | Update description, version bump to 0.3.0 |
| `CHANGELOG.md` | Add Phase 2B entry |

---

*Plan created June 24 2026. Research complete. Ready for build agent execution.*
