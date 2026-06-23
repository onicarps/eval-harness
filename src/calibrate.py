"""Judge calibration: measure inter-judge agreement.

The ``calibrate`` command runs every record through all available judges
and reports how much the judges disagree.  High disagreement signals that
the judge prompt needs calibration (too ambiguous, too strict, etc.).

Usage::

    eval-harness calibrate data.jsonl          # default: all judges
    eval-harness calibrate data.jsonl --sample 10
    eval-harness calibrate data.jsonl --json --output-file calibrate.json
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from src.db import Database
from src.ingest import IngestOptions, ingest_file, ingest_stdin
from src.judges import JudgeRegistry
from src.models import (
    EvalRecord,
    EvalResult,
    EvalRun,
    PassFail,
    RubricTemplate,
    RunStatus,
    _new_id,
)

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CalibrationSummary:
    """Aggregate calibration metrics across all records."""

    run_id: str
    total_records: int = 0
    total_judges: int = 0
    mean_std_dev: float = 0.0
    max_std_dev: float = 0.0
    median_std_dev: float = 0.0
    mean_faithfulness: float = 0.0
    mean_task_completion: float = 0.0
    pass_agreement_rate: float | None = None  # None when no data
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    judge_agreement: dict[str, dict[str, float]] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @classmethod
    def from_results(
        cls,
        results: list[EvalResult],
        run_id: str,
        judges: list[str],
        *,
        disagreement_threshold: float = 0.1,
    ) -> "CalibrationSummary":
        """Build a summary from raw ``EvalResult`` entries."""
        if not results:
            return cls(run_id=run_id, total_records=0, total_judges=len(judges))

        # Group results by record_id
        records: dict[str, list[EvalResult]] = {}
        for r in results:
            records.setdefault(r.record_id, []).append(r)

        total_records = len(records)
        all_std_devs: list[float] = []
        all_faith: list[float] = []
        all_task: list[float] = []
        all_pass_agree: list[bool] = []
        disagreements: list[dict[str, Any]] = []

        for rec_id, rec_results in records.items():
            combined = [r.combined_score for r in rec_results]
            faiths = [r.faithfulness for r in rec_results]
            tasks = [r.task_completion for r in rec_results]
            passes = [r.pass_fail for r in rec_results]

            # Use population stdev — we have ALL judges, not a sample
            std_dev = (
                statistics.pstdev(combined)
                if len(combined) >= 2
                else 0.0
            )
            all_std_devs.append(std_dev)
            all_faith.extend(faiths)
            all_task.extend(tasks)

            # Pass/fail agreement: all judges agree?
            if passes:
                unique_passes = set(passes)
                all_pass_agree.append(len(unique_passes) == 1)

            if std_dev >= disagreement_threshold:
                disagreements.append({
                    "record_id": rec_id,
                    "std_dev": round(std_dev, 4),
                    "mean_score": round(statistics.mean(combined), 4),
                    "min_score": round(min(combined), 4),
                    "max_score": round(max(combined), 4),
                    "n_judges": len(combined),
                })

        # Sort disagreements by std_dev descending
        disagreements.sort(key=lambda d: d["std_dev"], reverse=True)

        # Judge-pair agreement: for each pair, what % of records agree on pass/fail?
        judge_agreement = _compute_pair_agreement(results, judges)

        return cls(
            run_id=run_id,
            total_records=total_records,
            total_judges=len(judges),
            mean_std_dev=round(statistics.mean(all_std_devs), 4) if all_std_devs else 0.0,
            max_std_dev=round(max(all_std_devs), 4) if all_std_devs else 0.0,
            median_std_dev=round(statistics.median(all_std_devs), 4) if all_std_devs else 0.0,
            mean_faithfulness=round(statistics.mean(all_faith), 4) if all_faith else 0.0,
            mean_task_completion=round(statistics.mean(all_task), 4) if all_task else 0.0,
            pass_agreement_rate=(
                round(statistics.mean(all_pass_agree), 4)
                if all_pass_agree
                else None
            ),
            disagreements=disagreements,
            judge_agreement=judge_agreement,
        )


def compute_agreement_metrics(
    scores: list[float],
    threshold: float = 0.7,
) -> dict[str, Any]:
    """Compute agreement metrics for a list of scores.

    Uses population standard deviation (pstdev) since calibration
    evaluates all available judges, not a sample.

    Args:
        scores: Combined scores from multiple judges for one record.
        threshold: Pass/fail threshold.

    Returns:
        Dictionary with std_dev, mean_score, min_score, max_score,
        pass_agreement, pass_count, fail_count.
    """
    if not scores:
        return {
            "std_dev": 0.0,
            "mean_score": 0.0,
            "min_score": 0.0,
            "max_score": 0.0,
            "pass_agreement": True,
            "pass_count": 0,
            "fail_count": 0,
        }

    # Population stdev — all judges, not a sample
    std_dev = (
        statistics.pstdev(scores)
        if len(scores) >= 2
        else 0.0
    )
    mean_score = statistics.mean(scores)
    min_score = min(scores)
    max_score = max(scores)

    passes = sum(1 for s in scores if s >= threshold)
    fails = len(scores) - passes
    pass_agreement = (passes == len(scores)) or (fails == len(scores))

    return {
        "std_dev": round(std_dev, 4),
        "mean_score": round(mean_score, 4),
        "min_score": round(min_score, 4),
        "max_score": round(max_score, 4),
        "pass_agreement": pass_agreement,
        "pass_count": passes,
        "fail_count": fails,
    }


def _compute_pair_agreement(
    results: list[EvalResult],
    judges: list[str],
) -> dict[str, dict[str, float]]:
    """Compute pass/fail agreement rate between each pair of judges.

    For each pair of judges, count how many records they agree on
    (both PASS or both FAIL) and return the agreement rate.

    Args:
        results: All calibration results.
        judges: Ordered list of all judge model IDs (ensures consistent keys).

    Returns:
        Nested dict: {judge_a: {judge_b: agreement_rate}}.
    """
    # Group by record → {record_id: {judge: pass_fail}}
    record_judges: dict[str, dict[str, PassFail]] = {}
    for r in results:
        record_judges.setdefault(r.record_id, {})[r.judge_model] = r.pass_fail

    # Use passed judges list for consistent key ordering
    agreement: dict[str, dict[str, float]] = {}
    for j1 in judges:
        agreement[j1] = {}
        for j2 in judges:
            if j1 == j2:
                agreement[j1][j2] = 1.0
                continue
            agree_count = 0
            total = 0
            for rec_id, judges_map in record_judges.items():
                if j1 in judges_map and j2 in judges_map:
                    total += 1
                    if judges_map[j1] == judges_map[j2]:
                        agree_count += 1
            agreement[j1][j2] = round(agree_count / total, 4) if total > 0 else 0.0

    return agreement


# ── CalibrationRunner ────────────────────────────────────────────────────────

class CalibrationRunner:
    """Runs calibration: evaluates records through all judges and computes metrics."""

    def __init__(
        self,
        db: Database,
        api_key: str,
        judges: list[str],
        *,
        concurrency: int = 4,
        timeout: float = 60.0,
        rpm_limit: int | None = None,
        use_cache: bool = True,
    ) -> None:
        """Initialize the calibration runner.

        Args:
            db: Database for caching.
            api_key: API key for judge calls.
            judges: Ordered list of judge model IDs to use.
            concurrency: Max concurrent judge calls.
            timeout: Per-call timeout in seconds.
            rpm_limit: Optional requests-per-minute limit.
            use_cache: Whether to use the judge response cache.
        """
        if not judges:
            raise ValueError("at least one judge model is required")
        self.db = db
        self.api_key = api_key
        self.judges = judges
        self.concurrency = concurrency
        self.timeout = timeout
        self.rpm_limit = rpm_limit
        self.use_cache = use_cache

    def run(
        self,
        records: list[EvalRecord],
        run_id: str | None = None,
        rubric: RubricTemplate | None = None,
    ) -> CalibrationSummary:
        """Run calibration on ``records`` using all judges.

        Each record is evaluated through every judge independently.
        Results are stored in the DB under the calibration run_id.

        Args:
            records: Records to evaluate.
            run_id: Optional run identifier (defaults to new UUID).
            rubric: Optional rubric template (defaults to built-in v1).

        Returns:
            CalibrationSummary with agreement metrics.
        """
        from src.evaluator import EvaluatorConfig, LLMEvaluator  # avoid circular

        if rubric is None:
            rubric = cast(RubricTemplate, RubricTemplate(
                rubric_id="faithfulness-v1",
                version="1.0",
                description="Dual-dimension rubric: faithfulness + task completion.",
                prompt_template=(
                    "You are an impartial evaluator. Score the assistant's output on two "
                    "dimensions: FAITHFULNESS (does it stay grounded in the input/reference "
                    "without hallucination?) and TASK_COMPLETION (does it satisfy what was "
                    "asked?). Each dimension is a float in [0.0, 1.0].\n\n"
                    "Return STRICT JSON only, with the following keys: "
                    '{"faithfulness": float, "task_completion": float, '
                    '"faithfulness_reasoning": str, "task_completion_reasoning": str, '
                    '"reasoning": str}.\n\n'
                    "INPUT:\n{input}\n\nOUTPUT:\n{output}\n\nREFERENCE:\n{reference}\n"
                ),
            ))

        run_id = run_id or _new_id()
        run = EvalRun(
            run_id=run_id,
            status=RunStatus.RUNNING,
            record_count=len(records),
            config={
                "file": "<calibration>",
                "format": "internal",
                "judges": self.judges,
                "pass_threshold": 0.7,
            },
            rubric_id="faithfulness-v1",
            judge_model=",".join(self.judges),
        )
        self.db.insert_run(run)

        # CRITICAL FIX: max_fallbacks = len(judges) - 1 so ALL judges are tried.
        # The evaluator truncates to max(1, max_fallbacks + 1) judges.
        config = EvaluatorConfig(
            api_key=self.api_key,
            judges=self.judges,
            rubric=rubric,
            concurrency=self.concurrency,
            timeout=self.timeout,
            rpm_limit=self.rpm_limit,
            use_cache=self.use_cache,
            pass_threshold=0.7,
            max_fallbacks=len(self.judges) - 1,  # all judges
            no_fallback=False,
        )

        evaluator = LLMEvaluator(db=self.db, config=config)
        results = asyncio_run(evaluator.evaluate(run, records))

        # Persist run completion
        run.status = RunStatus.COMPLETED
        summary = CalibrationSummary.from_results(
            results, run.run_id, self.judges,
        )
        run.mean_score = summary.mean_faithfulness
        run.pass_rate = (
            summary.pass_agreement_rate
            if summary.pass_agreement_rate is not None
            else 0.0
        )
        self.db.update_run(run)

        logger.info(
            "Calibration complete: %d records, %d judges, "
            "mean_std_dev=%.4f, pass_agreement=%.1f%%",
            summary.total_records,
            summary.total_judges,
            summary.mean_std_dev,
            (summary.pass_agreement_rate or 0.0) * 100,
        )
        return summary


def asyncio_run(coro):
    """Run an async coroutine, handling both fresh and nested event loops."""
    try:
        asyncio.get_running_loop()
        # Already in a loop — create a new one
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    except RuntimeError:
        return asyncio.run(coro)


# ── Rendering ────────────────────────────────────────────────────────────────

def render_calibration_summary(summary: CalibrationSummary) -> str:
    """Render a CalibrationSummary as a human-readable table string."""
    lines = [
        f"Calibration Summary for run {summary.run_id}",
        "",
        f"  Records evaluated:    {summary.total_records}",
        f"  Judges used:          {summary.total_judges}",
        "",
        "  ── Overall Agreement ──",
        f"  Mean std deviation:   {summary.mean_std_dev:.4f}",
        f"  Median std deviation: {summary.median_std_dev:.4f}",
        f"  Max std deviation:    {summary.max_std_dev:.4f}",
        f"  Mean faithfulness:    {summary.mean_faithfulness:.4f}",
        f"  Mean task completion: {summary.mean_task_completion:.4f}",
        f"  Pass/fail agreement:  {(summary.pass_agreement_rate or 0.0):.1%}",
    ]

    if summary.disagreements:
        lines.append("")
        lines.append("  ── Disagreements (std dev ≥ 0.1) ──")
        for d in summary.disagreements[:10]:
            lines.append(
                f"  {d['record_id'][:12]:<14} "
                f"std_dev={d['std_dev']:.4f} "
                f"mean={d['mean_score']:.4f} "
                f"[{d['min_score']:.4f}, {d['max_score']:.4f}] "
                f"({d['n_judges']} judges)"
            )
        if len(summary.disagreements) > 10:
            lines.append(f"  ... and {len(summary.disagreements) - 10} more")

    if summary.judge_agreement:
        lines.append("")
        lines.append("  ── Judge Pair Agreement ──")
        judges = sorted(summary.judge_agreement.keys())
        header = f"  {'':>12}" + "".join(f"{j[:10]:>12}" for j in judges)
        lines.append(header)
        for j1 in judges:
            row = f"  {j1[:12]:>12}"
            for j2 in judges:
                rate = summary.judge_agreement[j1][j2]
                color = "green" if rate >= 0.9 else "yellow" if rate >= 0.7 else "red"
                row += f" \x1b[{32 + (90 if color == 'green' else 33 if color == 'yellow' else 31)}m{rate:.2%}\x1b[0m"
            lines.append(row)

    return "\n".join(lines)


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def render_calibration_json(summary: CalibrationSummary) -> str:
    """Render a CalibrationSummary as JSON."""
    data = {
        "run_id": summary.run_id,
        "total_records": summary.total_records,
        "total_judges": summary.total_judges,
        "mean_std_dev": summary.mean_std_dev,
        "max_std_dev": summary.max_std_dev,
        "median_std_dev": summary.median_std_dev,
        "mean_faithfulness": summary.mean_faithfulness,
        "mean_task_completion": summary.mean_task_completion,
        "pass_agreement_rate": summary.pass_agreement_rate,
        "disagreements": summary.disagreements,
        "judge_agreement": summary.judge_agreement,
    }
    return json.dumps(data, indent=2, default=str)