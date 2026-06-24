"""Abstract Agent base class and concrete adapters for eval-harness.

Provides:
- Agent: Abstract base class with async start/act/stop lifecycle.
- SubprocessAgent: Adapter that runs a CLI subprocess via stdin/stdout.
- PythonAgent: Adapter that calls a Python function directly.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from src.agent_models import AgentResult, TaskStep

logger = logging.getLogger(__name__)


class Agent(ABC):
    """Abstract base class for agent adapters.

    All agents follow a lifecycle: start() -> act(task) -> stop().
    Agents can also be used as async context managers::

        async with MyAgent(name="x") as agent:
            result = await agent.act("hello")
    """

    def __init__(self, name: str, **kwargs: Any) -> None:
        """Initialize the agent.

        Args:
            name: Human-readable agent identifier.
            **kwargs: Adapter-specific configuration.
        """
        self.name = name
        self._started = False

    @abstractmethod
    async def start(self) -> None:
        """Start the agent, acquiring any necessary resources.

        Must be called before act().  For stateless agents (e.g. PythonAgent),
        this may be a no-op.
        """
        ...

    @abstractmethod
    async def act(self, task: str) -> str:
        """Perform a task and return the agent's output.

        Args:
            task: The task prompt/input to give the agent.

        Returns:
            The agent's response as a string.

        Raises:
            RuntimeError: If the agent has not been started.
            asyncio.TimeoutError: If the task exceeds the configured timeout.
        """
        ...

    @abstractmethod
    async def stop(self) -> None:
        """Stop the agent, releasing any resources.

        After stop(), act() should not be called until start() is called again.
        """
        ...

    async def execute_step(self, step: TaskStep) -> AgentResult:
        """Execute a TaskStep and return an AgentResult.

        Wraps act() with error handling, timing, and result construction.

        Args:
            step: The TaskStep to execute.

        Returns:
            An AgentResult with the outcome of this step.
        """
        import time

        start = time.monotonic()
        try:
            output = await self.act(step.prompt)
            duration = time.monotonic() - start
            return AgentResult(
                step_id=step.id,
                agent_output=output,
                success=True,
                score=self._score_output(output, step.expected_output),
                duration_seconds=duration,
            )
        except TimeoutError:
            duration = time.monotonic() - start
            return AgentResult(
                step_id=step.id,
                agent_output="",
                success=False,
                score=0.0,
                error="timeout",
                duration_seconds=duration,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            return AgentResult(
                step_id=step.id,
                agent_output="",
                success=False,
                score=0.0,
                error=str(exc),
                duration_seconds=duration,
            )

    @staticmethod
    def _score_output(output: str, expected: str) -> float:
        """Score an agent's output against the expected output.

        Uses exact match (1.0) or case-insensitive match (0.8),
        otherwise returns a simple overlap score.

        Args:
            output: The agent's output.
            expected: The expected correct output.

        Returns:
            A score in [0.0, 1.0].
        """
        if not expected:
            return 1.0
        if output.strip() == expected.strip():
            return 1.0
        if output.strip().lower() == expected.strip().lower():
            return 0.8
        # Simple overlap scoring
        expected_words = set(expected.lower().split())
        output_words = set(output.lower().split())
        if not expected_words:
            return 1.0
        overlap = len(expected_words & output_words) / len(expected_words)
        return max(0.0, min(1.0, overlap))

    async def __aenter__(self) -> Agent:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()


class SubprocessAgent(Agent):
    """Agent adapter that runs a CLI subprocess via stdin/stdout.

    Communicates with the subprocess by writing tasks to stdin and
    reading responses from stdout (line-based).

    Args:
        name: Agent identifier.
        command: The command to execute (list of strings).
        timeout: Default timeout in seconds for each act() call.
        cwd: Working directory for the subprocess.
        env: Environment variables for the subprocess.
    """

    def __init__(
        self,
        name: str,
        command: list[str],
        timeout: float = 30.0,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self._command = command
        self._timeout = timeout
        self._cwd = cwd
        self._env = env
        self._process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        """Start the subprocess."""
        logger.info("Starting subprocess agent %s: %s", self.name, self._command)
        self._process = await asyncio.create_subprocess_exec(
            *self._command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self._cwd,
            env=self._env,
        )
        self._started = True
        logger.info("Subprocess agent %s started (PID %d)", self.name, self._process.pid)

    async def act(self, task: str) -> str:
        """Send a task to the subprocess via stdin and read the response."""
        if not self._started or self._process is None:
            raise RuntimeError(f"agent {self.name} not started")
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError(f"agent {self.name} pipes not available")

        logger.debug("Sending task to %s: %s", self.name, task[:80])
        self._process.stdin.write((task + "\n").encode())
        await self._process.stdin.drain()

        try:
            line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self._timeout,
            )
        except TimeoutError:
            logger.warning("Subprocess agent %s timed out", self.name)
            raise

        if not line:
            raise RuntimeError(f"subprocess {self.name} closed stdout")

        result = line.decode().strip()
        logger.debug("Received response from %s: %s", self.name, result[:80])
        return result

    async def stop(self) -> None:
        """Terminate the subprocess."""
        if self._process is not None:
            logger.info("Stopping subprocess agent %s (PID %d)", self.name, self._process.pid)
            try:
                self._process.terminate()
                try:
                    await asyncio.wait_for(self._process.wait(), timeout=5.0)
                except TimeoutError:
                    logger.warning("Subprocess %s did not terminate, killing", self.name)
                    self._process.kill()
                    await self._process.wait()
            except ProcessLookupError:
                pass  # Already dead
            finally:
                self._process = None
                self._started = False


class PythonAgent(Agent):
    """Agent adapter that calls a Python function directly.

    The handler function must accept a single string argument (the task)
    and return a string response.

    Args:
        name: Agent identifier.
        handler: Callable that takes a task string and returns a response string.
        timeout: Timeout in seconds (applied via asyncio.wait_for if possible).
    """

    def __init__(
        self,
        name: str,
        handler: Callable[[str], str],
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        if not callable(handler):
            raise TypeError(f"handler must be callable, got {type(handler)}")
        self._handler = handler
        self._timeout = timeout

    async def start(self) -> None:
        """No-op for PythonAgent (no resources to acquire)."""
        self._started = True
        logger.info("PythonAgent %s ready", self.name)

    async def act(self, task: str) -> str:
        """Call the handler function with the task."""
        if not self._started:
            raise RuntimeError(f"agent {self.name} not started")

        logger.debug("PythonAgent %s handling task: %s", self.name, task[:80])

        # Run the handler in a thread to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, self._handler, task),
            timeout=self._timeout,
        )
        return str(result)

    async def stop(self) -> None:
        """No-op for PythonAgent."""
        self._started = False
        logger.info("PythonAgent %s stopped", self.name)
