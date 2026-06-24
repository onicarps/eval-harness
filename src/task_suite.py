"""Task suite loading, registry, and built-in suites for eval-harness agent evaluation.

Provides:
- load_suite_from_dict / load_suite_from_yaml: Deserialize task suites.
- TaskSuiteRegistry: Generic registry for looking up suites by ID.
- BuiltinSuiteRegistry: The 5 built-in task suites (echo, math, file-read, string-reversal, multi-step).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.agent_models import TaskStep, TaskStepType, TaskSuite

logger = logging.getLogger(__name__)


# ── Loading functions ───────────────────────────────────────────────────────────


def load_suite_from_dict(data: dict[str, Any]) -> TaskSuite:
    """Load a :class:`TaskSuite` from a dictionary.

    Args:
        data: Dictionary with keys ``suite_id``, ``name``, ``description``,
            ``steps``, and optional ``metadata``.

    Returns:
        A fully constructed TaskSuite.

    Raises:
        KeyError: If required fields are missing.
    """
    steps_data = data["steps"]
    steps = []
    for s in steps_data:
        step = TaskStep(
            id=s["id"],
            prompt=s["prompt"],
            expected_output=s.get("expected_output", ""),
            step_type=TaskStepType(s.get("step_type", "echo")),
            timeout_seconds=s.get("timeout_seconds", 60.0),
            metadata=s.get("metadata", {}),
        )
        steps.append(step)

    return TaskSuite(
        suite_id=data["suite_id"],
        name=data["name"],
        description=data.get("description", ""),
        steps=steps,
        metadata=data.get("metadata", {}),
    )


def load_suite_from_yaml(path: str | Path) -> TaskSuite:
    """Load a :class:`TaskSuite` from a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        A fully constructed TaskSuite.

    Raises:
        yaml.YAMLError: If the file is not valid YAML.
        KeyError: If required fields are missing.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping at top level, got {type(data)}")
    return load_suite_from_dict(data)


# ── TaskSuiteRegistry ───────────────────────────────────────────────────────────


class TaskSuiteRegistry:
    """A registry for looking up :class:`TaskSuite` objects by ID.

    Supports registering custom suites and querying them at runtime.
    """

    def __init__(self) -> None:
        self._suites: dict[str, TaskSuite] = {}

    def register(self, suite: TaskSuite) -> None:
        """Register a task suite.

        Args:
            suite: The TaskSuite to register.
        """
        logger.debug("Registering suite %s (%s)", suite.suite_id, suite.name)
        self._suites[suite.suite_id] = suite

    def get(self, suite_id: str) -> TaskSuite | None:
        """Return the suite with ``suite_id`` or None."""
        return self._suites.get(suite_id)

    def list_ids(self) -> list[str]:
        """Return all registered suite IDs."""
        return list(self._suites.keys())

    def __contains__(self, suite_id: str) -> bool:
        return suite_id in self._suites

    def __len__(self) -> int:
        return len(self._suites)


# ── Built-in Suites ─────────────────────────────────────────────────────────────

# Each suite has exactly 5 tasks.

_ECHO_SUITE = {
    "suite_id": "echo-v1",
    "name": "Echo Tests",
    "description": "Tests basic echo/repeat functionality.",
    "steps": [
        {"id": "echo-1", "prompt": "hello", "expected_output": "hello", "step_type": "echo"},
        {"id": "echo-2", "prompt": "world", "expected_output": "world", "step_type": "echo"},
        {"id": "echo-3", "prompt": "foo bar baz", "expected_output": "foo bar baz", "step_type": "echo"},
        {"id": "echo-4", "prompt": "12345", "expected_output": "12345", "step_type": "echo"},
        {"id": "echo-5", "prompt": "The quick brown fox", "expected_output": "The quick brown fox", "step_type": "echo"},
    ],
}

_MATH_SUITE = {
    "suite_id": "math-v1",
    "name": "Math Tests",
    "description": "Tests basic arithmetic capabilities.",
    "steps": [
 {"id": "math-1", "prompt": "What is 2+2?", "expected_output": "4", "step_type": "math"},
        {"id": "math-2", "prompt": "What is 10-3?", "expected_output": "7", "step_type": "math"},
        {"id": "math-3", "prompt": "What is 3*4?", "expected_output": "12", "step_type": "math"},
        {"id": "math-4", "prompt": "What is 15/3?", "expected_output": "5", "step_type": "math"},
        {"id": "math-5", "prompt": "What is 7+8?", "expected_output": "15", "step_type": "math"},
    ],
}

_FILE_READ_SUITE = {
    "suite_id": "file-read-v1",
    "name": "File Read Tests",
    "description": "Tests file reading capabilities.",
    "steps": [
        {
            "id": "file-1",
            "prompt": "Read the file /tmp/test_file_1.txt",
            "expected_output": "alpha",
            "step_type": "file_read",
            "metadata": {"path": "/tmp/test_file_1.txt"},
        },
        {
            "id": "file-2",
            "prompt": "Read the file /tmp/test_file_2.txt",
            "expected_output": "beta",
            "step_type": "file_read",
            "metadata": {"path": "/tmp/test_file_2.txt"},
        },
        {
            "id": "file-3",
            "prompt": "Read the file /tmp/test_file_3.txt",
            "expected_output": "gamma",
            "step_type": "file_read",
            "metadata": {"path": "/tmp/test_file_3.txt"},
        },
        {
            "id": "file-4",
            "prompt": "Read the file /tmp/test_file_4.txt",
            "expected_output": "delta",
            "step_type": "file_read",
            "metadata": {"path": "/tmp/test_file_4.txt"},
        },
        {
            "id": "file-5",
            "prompt": "Read the file /tmp/test_file_5.txt",
            "expected_output": "epsilon",
            "step_type": "file_read",
            "metadata": {"path": "/tmp/test_file_5.txt"},
        },
    ],
}

_STRING_REVERSAL_SUITE = {
    "suite_id": "string-reversal-v1",
    "name": "String Reversal Tests",
    "description": "Tests string manipulation (reversal).",
    "steps": [
        {
            "id": "rev-1",
            "prompt": "Reverse the string 'hello'",
            "expected_output": "olleh",
            "step_type": "string_reversal",
        },
        {
            "id": "rev-2",
            "prompt": "Reverse the string 'world'",
            "expected_output": "dlrow",
            "step_type": "string_reversal",
        },
        {
            "id": "rev-3",
            "prompt": "Reverse the string 'abc'",
            "expected_output": "cba",
            "step_type": "string_reversal",
        },
        {
            "id": "rev-4",
            "prompt": "Reverse the string 'test'",
            "expected_output": "tset",
            "step_type": "string_reversal",
        },
        {
            "id": "rev-5",
            "prompt": "Reverse the string 'xyz'",
            "expected_output": "zyx",
            "step_type": "string_reversal",
        },
    ],
}

_MULTI_STEP_SUITE = {
    "suite_id": "multi-step-v1",
    "name": "Multi-Step Tests",
    "description": "Tests multi-step reasoning capabilities.",
    "steps": [
        {
            "id": "multi-1",
            "prompt": "First say 'step1', then say 'step2'",
            "expected_output": "step1 step2",
            "step_type": "multi_step",
        },
        {
            "id": "multi-2",
            "prompt": "Count from 1 to 3",
            "expected_output": "1 2 3",
            "step_type": "multi_step",
        },
        {
            "id": "multi-3",
            "prompt": "Say 'hello' and then 'goodbye'",
            "expected_output": "hello goodbye",
            "step_type": "multi_step",
        },
        {
            "id": "multi-4",
            "prompt": "Output 'a', 'b', 'c' in sequence",
            "expected_output": "a b c",
            "step_type": "multi_step",
        },
        {
            "id": "multi-5",
            "prompt": "First compute 1+1, then say 'done'",
            "expected_output": "2 done",
            "step_type": "multi_step",
        },
    ],
}

# ── BuiltinSuiteRegistry ────────────────────────────────────────────────────────


class BuiltinSuiteRegistry:
    """Provides access to the 5 built-in task suites.

    The built-in suites are lazily loaded on first access.
    """

    _cache: dict[str, TaskSuite] | None = None

    @classmethod
    def _load_all(cls) -> dict[str, TaskSuite]:
        if cls._cache is None:
            cls._cache = {}
            for suite_data in [
                _ECHO_SUITE,
                _MATH_SUITE,
                _FILE_READ_SUITE,
                _STRING_REVERSAL_SUITE,
                _MULTI_STEP_SUITE,
            ]:
                suite = load_suite_from_dict(suite_data)
                cls._cache[suite.suite_id] = suite
        return cls._cache

    @classmethod
    def get(cls, suite_id: str) -> TaskSuite | None:
        """Return the built-in suite with ``suite_id`` or None."""
        return cls._load_all().get(suite_id)

    @classmethod
    def list_ids(cls) -> list[str]:
        """Return all built-in suite IDs."""
        return list(cls._load_all().keys())

    @classmethod
    def all(cls) -> list[TaskSuite]:
        """Return all built-in suites."""
        return list(cls._load_all().values())


# ── Convenience function ────────────────────────────────────────────────────────


def get_suite_by_id(suite_id: str) -> TaskSuite | None:
    """Return a built-in suite by ID, or None if not found.

    Args:
        suite_id: The suite identifier (e.g., 'echo-v1').

    Returns:
        The TaskSuite or None.
    """
    return BuiltinSuiteRegistry.get(suite_id)


# ── Module-level constant for quick access ─────────────────────────────────────

BUILTIN_SUITES: list[TaskSuite] = BuiltinSuiteRegistry.all()
