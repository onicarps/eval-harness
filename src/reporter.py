"""Reporting helpers: summaries, Rich tables, ASCII histograms, and exports."""

from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path
from typing import cast

from rich.console import Console
from rich.table import Table

from src.models import EvalRecord, EvalResult, EvalRun, EvalSummary, PassFail


def build_summary(run: EvalRun, results: list[EvalResult]) -> EvalSummary:
    """Aggregate per-record results into a single :class:`EvalSummary`."""
    total = len(results)
    if total == 0:
        return EvalSummary(
            run_id=run.run_id,
            total=0,
            passed=0,
            failed=0,
            pass_rate=0.0,
            mean_faithfulness=0.0,
            mean_task_completion=0.0,
            mean_combined=0.0,
            eval_time_seconds=run.eval_time_seconds or 0.0,
            judge_usage={},
            errors=0,
        )
    passed = sum(1 for r in results if r.pass_fail == PassFail.PASS)
    failed = total - passed
    errors = sum(1 for r in results if r.error)
    mean_faith = sum(r.faithfulness for r in results) / total
    mean_task = sum(r.task_completion for r in results) / total
    mean_combined = sum(r.combined_score for r in results) / total
    return EvalSummary(
        run_id=run.run_id,
        total=total,
        passed=passed,
        failed=failed,
        pass_rate=passed / total,
        mean_faithfulness=mean_faith,
        mean_task_completion=mean_task,
        mean_combined=mean_combined,
        eval_time_seconds=run.eval_time_seconds or 0.0,
        judge_usage=judge_usage(results),
        errors=errors,
    )


def judge_usage(results: list[EvalResult]) -> dict[str, int]:
    """Return a histogram of judge_model -> count across results."""
    return dict(Counter(r.judge_model for r in results if r.judge_model))


def ascii_histogram(values: list[float], buckets: int = 10, width: int = 40) -> str:
    """Render a simple ASCII histogram of ``values`` distributed across ``buckets``."""
    if not values:
        return "(no data)\n"
    counts = [0] * buckets
    for v in values:
        v = max(0.0, min(1.0, v))
        idx = min(buckets - 1, int(v * buckets))
        counts[idx] += 1
    peak = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        lo = i / buckets
        hi = (i + 1) / buckets
        bar = "#" * int((c / peak) * width)
        lines.append(f"[{lo:.2f}-{hi:.2f}) {bar} {c}")
    return "\n".join(lines) + "\n"


def render_table(summary: EvalSummary) -> str:
    """Render the EvalSummary as a Rich table string."""
    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    table = Table(title=f"Summary for run {summary.run_id}")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("total", str(summary.total))
    table.add_row("passed", str(summary.passed))
    table.add_row("failed", str(summary.failed))
    table.add_row("pass_rate", f"{summary.pass_rate:.2%}")
    table.add_row("mean_faithfulness", f"{summary.mean_faithfulness:.3f}")
    table.add_row("mean_task_completion", f"{summary.mean_task_completion:.3f}")
    table.add_row("mean_combined", f"{summary.mean_combined:.3f}")
    table.add_row("eval_time_seconds", f"{summary.eval_time_seconds:.2f}")
    table.add_row("errors", str(summary.errors))
    console.print(table)
    if summary.judge_usage:
        usage = Table(title="Judge usage")
        usage.add_column("model")
        usage.add_column("count")
        for k, v in summary.judge_usage.items():
            usage.add_row(k, str(v))
        console.print(usage)
    # Rich uses StringIO internally; IO[str] stub lacks getvalue()
    return cast(str, console.file.getvalue())  # type: ignore[attr-defined]


def render_comparison_table(
    records: list[EvalRecord],
    results: list[EvalResult],
    judges: list[str],
) -> str:
    """Render a side-by-side judge comparison table.

    Args:
        records: The evaluated records.
        results: All results (from multiple judges).
        judges: Ordered list of judge model names.

    Returns:
        Rich-formatted table string.
    """
    console = Console(file=io.StringIO(), force_terminal=False, width=120)
    table = Table(title="Judge Comparison")
    table.add_column("Record", max_width=30)
    for judge in judges:
        short = judge.split("/")[-1][:16] if "/" in judge else judge[:16]
        table.add_column(short, justify="center")
    table.add_column("Std Dev", justify="center")
    table.add_column("Agree", justify="center")

    rec_map = {r.record_id: r for r in records}
    for record in records:
        rec_results = [r for r in results if r.record_id == record.record_id]
        if not rec_results:
            continue
        scores = []
        row = [record.input_text[:28] + "..." if len(record.input_text) > 28 else record.input_text]
        for judge in judges:
            jr = next((r for r in rec_results if r.judge_model == judge), None)
            if jr:
                scores.append(jr.combined_score)
                row.append(f"{jr.combined_score:.2f}")
            else:
                row.append("-")
        # Compute std dev
        import statistics
        if len(scores) >= 2:
            std = statistics.pstdev(scores)
            row.append(f"{std:.3f}")
        else:
            row.append("-")
        # Pass/fail agreement
        passes = sum(1 for s in scores if s >= 0.7)
        agrees = "yes" if passes == 0 or passes == len(scores) else "no"
        row.append(agrees)
        table.add_row(*row)

    console.print(table)
    return cast(str, console.file.getvalue())  # type: ignore[attr-defined]


def print_table(summary: EvalSummary, console: Console | None = None) -> None:
    """Print the summary table to the supplied or default Rich console."""
    c = console or Console()
    c.print(render_table(summary))


def export_results(
    run: EvalRun, results: list[EvalResult], out_path: str | Path, fmt: str = "json"
) -> Path:
    """Export results to JSON or CSV.

    Args:
        run: The associated run.
        results: Iterable of EvalResult.
        out_path: Destination file path.
        fmt: 'json' or 'csv'.
    """
    out_path = Path(out_path)
    if fmt == "json":
        payload = {
            "run_id": run.run_id,
            "results": [r.model_dump(mode="json") for r in results],
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
    elif fmt == "csv":
        fields = [
            "result_id",
            "record_id",
            "run_id",
            "faithfulness",
            "task_completion",
            "combined_score",
            "pass_fail",
            "judge_model",
            "judge_fallbacks",
            "error",
        ]
        with out_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in results:
                w.writerow(
                    {
                        "result_id": r.result_id,
                        "record_id": r.record_id,
                        "run_id": r.run_id,
                        "faithfulness": r.faithfulness,
                        "task_completion": r.task_completion,
                        "combined_score": r.combined_score,
                        "pass_fail": r.pass_fail.value,
                        "judge_model": r.judge_model,
                        "judge_fallbacks": r.judge_fallbacks,
                        "error": r.error or "",
                    }
                )
    else:
        raise ValueError(f"unsupported export format: {fmt}")
    return out_path
