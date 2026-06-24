"""Tests for src/task_suite.py — Task suite loading and built-in suites."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.agent_models import TaskStep, TaskStepType, TaskSuite
from src.task_suite import (
    BUILTIN_SUITES,
    BuiltinSuiteRegistry,
    TaskSuiteRegistry,
    get_suite_by_id,
    load_suite_from_dict,
    load_suite_from_yaml,
)


# ── load_suite_from_dict tests ─────────────────────────────────────────────────


class TestLoadSuiteFromDict:
    """Tests for loading a TaskSuite from a dictionary."""

    def test_load_basic_suite(self) -> None:
        data = {
            "suite_id": "test-v1",
            "name": "Test Suite",
            "description": "A test suite",
            "steps": [
                {"id": "s1", "prompt": "Say hello", "expected_output": "hello"},
                {"id": "s2", "prompt": "Say world", "expected_output": "world"},
            ],
        }
        suite = load_suite_from_dict(data)
        assert suite.suite_id == "test-v1"
        assert suite.name == "Test Suite"
        assert len(suite.steps) == 2
        assert suite.steps[0].id == "s1"
        assert suite.steps[0].step_type == TaskStepType.ECHO

    def test_load_suite_with_step_types(self) -> None:
        data = {
            "suite_id": "typed",
            "name": "Typed Suite",
            "steps": [
                {
                    "id": "m1",
                    "prompt": "Compute 1+1",
                    "expected_output": "2",
                    "step_type": "math",
                },
                {
                    "id": "f1",
                    "prompt": "Read /tmp/x",
                    "expected_output": "content",
                    "step_type": "file_read",
                },
            ],
        }
        suite = load_suite_from_dict(data)
        assert suite.steps[0].step_type == TaskStepType.MATH
        assert suite.steps[1].step_type == TaskStepType.FILE_READ

    def test_load_suite_with_metadata(self) -> None:
        data = {
            "suite_id": "meta",
            "name": "Meta",
            "steps": [
                {
                    "id": "s1",
                    "prompt": "test",
                    "expected_output": "ok",
                    "metadata": {"key": "value"},
                },
            ],
        }
        suite = load_suite_from_dict(data)
        assert suite.steps[0].metadata["key"] == "value"

    def test_load_suite_empty_steps(self) -> None:
        data = {"suite_id": "empty", "name": "Empty", "steps": []}
        suite = load_suite_from_dict(data)
        assert len(suite.steps) == 0

    def test_load_suite_missing_required_field(self) -> None:
        with pytest.raises(KeyError):
            load_suite_from_dict({"name": "No ID", "steps": []})


# ── load_suite_from_yaml tests ─────────────────────────────────────────────────


class TestLoadSuiteFromYaml:
    """Tests for loading a TaskSuite from YAML."""

    def test_load_basic_yaml(self, tmp_path: Path) -> None:
        """Load a TaskSuite from a YAML file."""
        data = {
            "suite_id": "yaml-test",
            "name": "YAML Test",
            "description": "Testing YAML loading",
            "steps": [
                {"id": "y1", "prompt": "hello", "expected_output": "world"},
            ],
        }
        yaml_path = tmp_path / "suite.yaml"
        yaml_path.write_text(yaml.dump(data))
        suite = load_suite_from_yaml(yaml_path)
        assert suite.suite_id == "yaml-test"
        assert len(suite.steps) == 1

    def test_load_invalid_yaml(self, tmp_path: pytest.Path) -> None:
        """Loading invalid YAML raises an error."""
        yaml_path = tmp_path / "bad.yaml"
        yaml_path.write_text("{{invalid yaml")
        with pytest.raises(yaml.YAMLError):
            load_suite_from_yaml(yaml_path)


# ── BuiltinSuiteRegistry tests ─────────────────────────────────────────────────


class TestBuiltinSuiteRegistry:
    """Tests for the built-in task suite registry."""

    def test_registry_contains_echo_suite(self) -> None:
        suite = get_suite_by_id("echo-v1")
        assert suite is not None
        assert suite.suite_id == "echo-v1"
        assert len(suite.steps) > 0

    def test_registry_contains_math_suite(self) -> None:
        suite = get_suite_by_id("math-v1")
        assert suite is not None
        assert suite.suite_id == "math-v1"
        assert len(suite.steps) > 0

    def test_registry_contains_file_read_suite(self) -> None:
        suite = get_suite_by_id("file-read-v1")
        assert suite is not None
        assert suite.suite_id == "file-read-v1"
        assert len(suite.steps) > 0

    def test_registry_contains_string_reversal_suite(self) -> None:
        suite = get_suite_by_id("string-reversal-v1")
        assert suite is not None
        assert suite.suite_id == "string-reversal-v1"
        assert len(suite.steps) > 0

    def test_registry_contains_multi_step_suite(self) -> None:
        suite = get_suite_by_id("multi-step-v1")
        assert suite is not None
        assert suite.suite_id == "multi-step-v1"
        assert len(suite.steps) > 0

    def test_registry_returns_none_for_unknown(self) -> None:
        suite = get_suite_by_id("nonexistent-v99")
        assert suite is None

    def test_registry_list_all(self) -> None:
        suite_ids = BuiltinSuiteRegistry.list_ids()
        assert "echo-v1" in suite_ids
        assert "math-v1" in suite_ids
        assert "file-read-v1" in suite_ids
        assert "string-reversal-v1" in suite_ids
        assert "multi-step-v1" in suite_ids

    def test_registry_all_have_5_steps(self) -> None:
        """Each built-in suite should have exactly 5 tasks."""
        for suite_id in BuiltinSuiteRegistry.list_ids():
            suite = get_suite_by_id(suite_id)
            assert suite is not None
            assert len(suite.steps) == 5, f"Suite {suite_id} has {len(suite.steps)} steps, expected 5"

    def test_registry_echo_suite_content(self) -> None:
        """Echo suite should test basic echo functionality."""
        suite = get_suite_by_id("echo-v1")
        assert suite is not None
        for step in suite.steps:
            assert step.step_type == TaskStepType.ECHO

    def test_registry_math_suite_content(self) -> None:
        """Math suite should test arithmetic."""
        suite = get_suite_by_id("math-v1")
        assert suite is not None
        for step in suite.steps:
            assert step.step_type == TaskStepType.MATH

    def test_registry_file_read_suite_content(self) -> None:
        """File-read suite should test file reading."""
        suite = get_suite_by_id("file-read-v1")
        assert suite is not None
        for step in suite.steps:
            assert step.step_type == TaskStepType.FILE_READ

    def test_registry_string_reversal_suite_content(self) -> None:
        """String reversal suite should test string manipulation."""
        suite = get_suite_by_id("string-reversal-v1")
        assert suite is not None
        for step in suite.steps:
            assert step.step_type == TaskStepType.STRING_REVERSAL

    def test_registry_multi_step_suite_content(self) -> None:
        """Multi-step suite should test multi-step tasks."""
        suite = get_suite_by_id("multi-step-v1")
        assert suite is not None
        for step in suite.steps:
            assert step.step_type == TaskStepType.MULTI_STEP


# ── TaskSuiteRegistry tests ────────────────────────────────────────────────────


class TestTaskSuiteRegistry:
    """Tests for the generic TaskSuiteRegistry."""

    def test_register_and_get(self) -> None:
        registry = TaskSuiteRegistry()
        suite = TaskSuite(suite_id="custom", name="Custom", steps=[])
        registry.register(suite)
        assert registry.get("custom") is not None

    def test_register_duplicate(self) -> None:
        """Registering the same suite_id twice overwrites."""
        registry = TaskSuiteRegistry()
        s1 = TaskSuite(suite_id="dup", name="V1", steps=[])
        s2 = TaskSuite(suite_id="dup", name="V2", steps=[])
        registry.register(s1)
        registry.register(s2)
        assert registry.get("dup").name == "V2"

    def test_list_empty(self) -> None:
        registry = TaskSuiteRegistry()
        assert registry.list_ids() == []

    def test_list_multiple(self) -> None:
        registry = TaskSuiteRegistry()
        registry.register(TaskSuite(suite_id="a", name="A", steps=[]))
        registry.register(TaskSuite(suite_id="b", name="B", steps=[]))
        ids = registry.list_ids()
        assert "a" in ids
        assert "b" in ids
