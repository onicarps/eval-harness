"""Tests for src/agent.py — Abstract Agent base class and adapters."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from src.agent import Agent, PythonAgent, SubprocessAgent
from src.agent_models import AgentResult

# ── Abstract Agent tests ────────────────────────────────────────────────────────


class TestAgent:
    """Tests for the abstract Agent base class."""

    def test_agent_is_abstract(self) -> None:
        """Agent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Agent(name="test")  # type: ignore[abstract]

    def test_agent_subclass_must_implement(self) -> None:
        """Subclasses must implement start, act, stop."""
        class IncompleteAgent(Agent):
            pass

        with pytest.raises(TypeError):
            IncompleteAgent(name="test")  # type: ignore[abstract]

    def test_agent_name(self) -> None:
        """Agent stores its name."""
        class MyAgent(Agent):
            async def start(self) -> None:
                pass
            async def act(self, task: str) -> str:
                return task
            async def stop(self) -> None:
                pass

        agent = MyAgent(name="my-agent")
        assert agent.name == "my-agent"

    def test_agent_context_manager_not_implemented_by_default(self) -> None:
        """Agent supports async context manager protocol."""
        class MyAgent(Agent):
            async def start(self) -> None:
                pass
            async def act(self, task: str) -> str:
                return task
            async def stop(self) -> None:
                pass

        agent = MyAgent(name="ctx-test")
        assert hasattr(agent, "__aenter__")
        assert hasattr(agent, "__aexit__")


# ── SubprocessAgent tests ────────────────────────────────────────────────────────


class TestSubprocessAgent:
    """Tests for the SubprocessAgent adapter."""

    @pytest.fixture()
    def echo_script(self, tmp_path: Path) -> Path:
        """Create a simple echo script for testing."""
        script = tmp_path / "echo_agent.py"
        script.write_text(
            'import sys\n'
            'for line in sys.stdin:\n'
            '    sys.stdout.write(line.strip() + "\\n")\n'
            '    sys.stdout.flush()\n'
        )
        return script

    @pytest.mark.asyncio
    async def test_subprocess_agent_start_and_stop(self, echo_script: Path) -> None:
        agent = SubprocessAgent(
            name="echo",
            command=[sys.executable, str(echo_script)],
        )
        await agent.start()
        assert agent._process is not None  # type: ignore[attr-defined]
        await agent.stop()
        # After stop(), _process is set to None
        assert agent._process is None  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_subprocess_agent_act(self, echo_script: Path) -> None:
        agent = SubprocessAgent(
            name="echo",
            command=[sys.executable, str(echo_script)],
        )
        await agent.start()
        result = await agent.act("hello world")
        await agent.stop()
        assert result is not None
        assert "hello world" in result

    @pytest.mark.asyncio
    async def test_subprocess_agent_timeout(self, tmp_path: Path) -> None:
        """SubprocessAgent respects timeout."""
        script = tmp_path / "slow_agent.py"
        script.write_text(
            'import time\n'
            'time.sleep(10)\n'
            'print("done")\n'
        )
        agent = SubprocessAgent(
            name="slow",
            command=[sys.executable, str(script)],
            timeout=0.5,
        )
        await agent.start()
        with pytest.raises(asyncio.TimeoutError):
            await agent.act("test")
        await agent.stop()

    @pytest.mark.asyncio
    async def test_subprocess_agent_not_started_error(self) -> None:
        """Calling act() before start() raises RuntimeError."""
        agent = SubprocessAgent(
            name="not-started",
            command=[sys.executable, "-c", "print('hi')"],
        )
        with pytest.raises(RuntimeError, match="not started"):
            await agent.act("test")

    @pytest.mark.asyncio
    async def test_subprocess_agent_context_manager(self, echo_script: Path) -> None:
        """SubprocessAgent works as async context manager."""
        async with SubprocessAgent(
            name="ctx-echo",
            command=[sys.executable, str(echo_script)],
        ) as agent:
            result = await agent.act("test prompt")
            assert "test prompt" in result


# ── PythonAgent tests ────────────────────────────────────────────────────────────


class TestPythonAgent:
    """Tests for the PythonAgent adapter."""

    def test_python_agent_with_echo_function(self) -> None:
        """PythonAgent calls a Python function."""
        def echo_fn(task: str) -> str:
            return f"echo: {task}"

        agent = PythonAgent(name="py-echo", handler=echo_fn)
        assert agent.name == "py-echo"

    @pytest.mark.asyncio
    async def test_python_agent_start(self) -> None:
        """PythonAgent start is a no-op (no resources to acquire)."""
        def echo_fn(task: str) -> str:
            return task

        agent = PythonAgent(name="py", handler=echo_fn)
        await agent.start()
        await agent.stop()

    @pytest.mark.asyncio
    async def test_python_agent_act(self) -> None:
        """PythonAgent calls the handler function."""
        def echo_fn(task: str) -> str:
            return f"result: {task}"

        agent = PythonAgent(name="py", handler=echo_fn)
        await agent.start()
        result = await agent.act("hello")
        await agent.stop()
        assert result == "result: hello"

    @pytest.mark.asyncio
    async def test_python_agent_act_with_math(self) -> None:
        """PythonAgent handles math tasks."""
        def math_fn(task: str) -> str:
            if "2+2" in task:
                return "4"
            return "unknown"

        agent = PythonAgent(name="math", handler=math_fn)
        await agent.start()
        result = await agent.act("What is 2+2?")
        await agent.stop()
        assert result == "4"

    @pytest.mark.asyncio
    async def test_python_agent_handler_exception(self) -> None:
        """PythonAgent propagates handler exceptions."""
        def failing_fn(task: str) -> str:
            raise ValueError("handler error")

        agent = PythonAgent(name="fail", handler=failing_fn)
        await agent.start()
        with pytest.raises(ValueError, match="handler error"):
            await agent.act("test")
        await agent.stop()

    @pytest.mark.asyncio
    async def test_python_agent_context_manager(self) -> None:
        """PythonAgent works as async context manager."""
        def fn(task: str) -> str:
            return task.upper()

        async with PythonAgent(name="ctx-py", handler=fn) as agent:
            result = await agent.act("hello")
            assert result == "HELLO"

    def test_python_agent_handler_required(self) -> None:
        """PythonAgent requires a callable handler."""
        with pytest.raises(TypeError):
            PythonAgent(name="bad", handler="not callable")  # type: ignore[arg-type]


# ── AgentResult mapping tests ───────────────────────────────────────────────────


class TestAgentResultMapping:
    """Tests for converting agent outputs to AgentResult objects."""

    def test_successful_result(self) -> None:
        """AgentResult for a successful step."""
        result = AgentResult(
            step_id="step-1",
            agent_output="hello",
            success=True,
            score=1.0,
        )
        assert result.success is True
        assert result.score == 1.0

    def test_failed_result(self) -> None:
        """AgentResult for a failed step."""
        result = AgentResult(
            step_id="step-1",
            agent_output="",
            success=False,
            score=0.0,
            error="timeout",
        )
        assert result.success is False
        assert result.error == "timeout"
