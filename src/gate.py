"""CI/CD quality gate for eval-harness runs.

The ``gate`` command checks whether a completed run meets a pass-rate
threshold, or suggests a baseline from historical runs.  It is designed
for CI/CD pipelines:

.. code-block:: bash

   eval-harness gate --run-id abc123 --threshold 0.8   # exits 0 or 1
   eval-harness gate --suggest-baseline                # prints suggestion
   eval-harness gate --run-id abc123 --json            # machine-readable

Exit codes
----------
0 — run passes the threshold
1 — run fails the threshold
2 — error (run not found, DB issue, etc.)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.db import Database
from src.models import EvalRun, RunStatus

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class CheckGateResult:
    """Result of a single gate check."""
    run_id: str
    pass_rate: float
    threshold: float
    passed: bool
    mean_score: float | None
    record_count: int

    def to_json(self) -> str:
        """Serialize to JSON for CI/CD consumption."""
        return json.dumps({
            "run_id": self.run_id,
            "pass_rate": self.pass_rate,
            "threshold": self.threshold,
            "passed": self.passed,
            "mean_score": self.mean_score,
            "record_count": self.record_count,
            "exit_code": self.exit_code,
        }, indent=2)

    def to_text(self) -> str:
        """Render as a human-readable table string."""
        status = "PASS" if self.passed else "FAIL"
        color = "green" if self.passed else "red"
        lines = [
            f"Gate result for run {self.run_id}",
            "",
            f"  Pass rate:      {self.pass_rate:.1%}  [{color}]{status}[/{color}]",
            f"  Threshold:      {self.threshold:.1%}",
            f"  Mean score:     {self.mean_score:.3f}" if self.mean_score is not None else "  Mean score:     -",
            f"  Records:        {self.record_count}",
            "",
            f"  Exit code:      {self.exit_code} ({'pass' if self.passed else 'fail'})",
        ]
        return "\n".join(lines)

    @property
    def exit_code(self) -> int:
        """Return the appropriate exit code."""
        return 0 if self.passed else 1


# ── GateRunner ───────────────────────────────────────────────────────────────

class GateRunner:
    """Runs gate checks against stored evaluation runs."""

    def __init__(self, db: Database) -> None:
        """Initialize with a database connection.

        Args:
            db: Database instance connected to the eval database.
        """
        self.db = db

    def check(self, run_id: str, threshold: float = 0.7) -> CheckGateResult:
        """Check if a run meets the pass-rate threshold.

        Args:
            run_id: The run to check.
            threshold: Required pass rate (0.0-1.0).

        Returns:
            CheckGateResult with pass/fail status.

        Raises:
            ValueError: If the run is not found.
        """
        run = self.db.get_run(run_id)
        if run is None:
            raise ValueError(f"run not found: {run_id}")

        if run.pass_rate is None:
            raise ValueError(f"run {run_id} has no pass_rate (was it completed?)")

        passed = run.pass_rate >= threshold
        result = CheckGateResult(
            run_id=run.run_id,
            pass_rate=run.pass_rate,
            threshold=threshold,
            passed=passed,
            mean_score=run.mean_score,
            record_count=run.record_count,
        )
        logger.info(
            "Gate check %s: pass_rate=%.1f%% threshold=%.1f%% %s",
            run_id, run.pass_rate * 100, threshold * 100,
            "PASS" if passed else "FAIL",
        )
        return result

    def suggest_baseline(self) -> dict[str, Any] | None:
        """Analyze historical runs and suggest a baseline threshold.

        Computes the median pass rate of completed runs and suggests
        using the 25th percentile as a baseline (conservative estimate).

        Returns:
            Dict with suggestion details, or None if no data.
        """
        runs = self.db.list_runs(limit=100)
        completed = [r for r in runs if r.status == RunStatus.COMPLETED and r.pass_rate is not None]

        if not completed:
            logger.info("No completed runs with pass_rate to suggest baseline")
            return None

        rates: list[float] = [r.pass_rate for r in completed if r.pass_rate is not None]
        if not rates:
            return None
        rates.sort()
        median = rates[len(rates) // 2]

        # Suggest 25th percentile as baseline (conservative)
        q25_idx = max(0, len(rates) // 4)
        q25 = rates[q25_idx]

        suggestion: dict[str, Any] = {
            "recommended_baseline": q25,
            "median_pass_rate": median,
            "runs_analyzed": len(completed),
            "min_pass_rate": rates[0],
            "max_pass_rate": rates[-1],
            "note": "Baseline is the 25th percentile — conservative estimate.",
        }
        logger.info(
            "Suggested baseline: %.1f%% (median=%.1f%%, %d runs)",
            q25 * 100, median * 100, len(completed),
        )
        return suggestion