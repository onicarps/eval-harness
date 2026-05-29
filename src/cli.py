"""Typer CLI for eval-harness."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import typer
from rich.console import Console

from src.db import Database
from src.evaluator import EvaluatorConfig, LLMEvaluator
from src.ingest import IngestOptions, ingest_file, ingest_stdin
from src.judges import JudgeRegistry
from src.models import BUILTIN_RUBRIC_V1, EvalRun, RunStatus
from src.reporter import (
    build_summary,
    export_results,
    print_table,
    render_table,
)

DEFAULT_DB_PATH = Path.home() / ".eval-harness" / "eval.db"

app = typer.Typer(
    name="eval-harness",
    help="Evaluate LLM outputs against a dual-dimension rubric.",
    add_completion=False,
    no_args_is_help=True,
)


def _get_api_key() -> str:
    """Return the OPENRIXER_API_KEY env value or empty string."""
    return os.environ.get("OPENRIXER_API_KEY", "")


def _judge_list(
    explicit: str | None, registry: JudgeRegistry, no_fallback: bool, max_fallbacks: int
) -> list[str]:
    """Resolve the ordered judge list to attempt for each record."""
    available = [m.id for m in registry.list()]
    if explicit:
        ordered = [explicit] + [m for m in available if m != explicit]
    else:
        ordered = available
    if no_fallback:
        return ordered[:1]
    return ordered[: max(1, max_fallbacks + 1)]


@app.command("run", help="Ingest a file, evaluate it, and report.")
def run_cmd(
    file: Path = typer.Argument(..., help="Input file path (use '-' for stdin)."),
    format: str = typer.Option("jsonl", "--format", help="jsonl or csv"),
    input_col: str = typer.Option("input", "--input-col"),
    output_col: str = typer.Option("output", "--output-col"),
    reference_col: str = typer.Option("reference", "--reference-col"),
    sample: int | None = typer.Option(None, "--sample"),
    since: str | None = typer.Option(None, "--since"),
    limit: int | None = typer.Option(None, "--limit"),
    judge: str | None = typer.Option(None, "--judge"),
    no_fallback: bool = typer.Option(False, "--no-fallback"),
    max_fallbacks: int = typer.Option(3, "--max-fallbacks"),
    pass_threshold: float = typer.Option(0.7, "--pass-threshold"),
    output: str = typer.Option("table", "--output", help="json or table"),
    output_file: Path | None = typer.Option(None, "--output-file"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    resume: bool = typer.Option(False, "--resume"),
    timeout: float = typer.Option(60.0, "--timeout"),
    rpm_limit: int | None = typer.Option(None, "--rpm-limit"),
    yes: bool = typer.Option(False, "--yes"),
    verbose: bool = typer.Option(False, "--verbose"),
    quiet: bool = typer.Option(False, "--quiet"),
    config: Path | None = typer.Option(None, "--config"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
    judges_cache: Path | None = typer.Option(None, "--judges-cache"),
) -> None:
    """Run ingest + evaluation pipeline."""
    console = Console(quiet=quiet)
    options = IngestOptions(
        input_col=input_col,
        output_col=output_col,
        reference_col=reference_col,
        sample=sample,
        since=since,
        limit=limit,
    )
    if str(file) == "-":
        records = list(ingest_stdin(sys.stdin, fmt=format, options=options))
    else:
        if not file.exists():
            console.print(f"[red]file not found: {file}[/red]")
            raise typer.Exit(code=2)
        records = list(ingest_file(file, fmt=format, options=options))

    if not records:
        console.print("[yellow]no records to evaluate[/yellow]")
        raise typer.Exit(code=2)

    if dry_run:
        console.print(f"[green]dry-run: {len(records)} record(s) parsed.[/green]")
        for r in records[:5]:
            console.print(f" - {r.input_text[:80]} -> {r.output_text[:80]}")
        raise typer.Exit(code=0)

    api_key = _get_api_key()
    if not api_key:
        console.print("[red]OPENRIXER_API_KEY is not set. Aborting.[/red]")
        raise typer.Exit(code=2)

    if not yes and not quiet:
        console.print(f"Will evaluate {len(records)} record(s). Continue? [y/N]")

    registry = JudgeRegistry(cache_path=judges_cache) if judges_cache else JudgeRegistry()
    judges = _judge_list(judge, registry, no_fallback, max_fallbacks)
    if not judges:
        console.print("[red]no judges available[/red]")
        raise typer.Exit(code=2)

    db = Database(db_path)
    try:
        run = EvalRun(
            config={
                "file": str(file),
                "format": format,
                "judges": judges,
                "pass_threshold": pass_threshold,
            },
            judge_model=judges[0],
        )
        db.insert_run(run)
        for rec in records:
            rec.run_id = run.run_id
            db.insert_record(rec)
        run.record_count = len(records)
        db.update_run(run)

        evaluator = LLMEvaluator(
            db=db,
            config=EvaluatorConfig(
                api_key=api_key,
                judges=judges,
                rubric=BUILTIN_RUBRIC_V1,
                concurrency=4,
                timeout=timeout,
                rpm_limit=rpm_limit,
                pass_threshold=pass_threshold,
                max_fallbacks=max_fallbacks,
                no_fallback=no_fallback,
            ),
        )

        progress_cb = None
        if len(records) > 10 and not quiet:

            def progress_cb(done: int, total: int) -> None:
                if done == total or done % max(1, total // 20) == 0:
                    console.print(f"[dim]progress: {done}/{total}[/dim]")

        start = time.monotonic()
        try:
            results = asyncio.run(
                evaluator.evaluate(run, records, resume=resume, progress_cb=progress_cb)
            )
        except Exception as exc:
            console.print(f"[red]evaluator error: {exc}[/red]")
            run.status = RunStatus.FAILED
            db.update_run(run)
            raise typer.Exit(code=2) from exc
        elapsed = time.monotonic() - start

        for r in results:
            db.insert_result(r)
        run.status = RunStatus.COMPLETED
        run.eval_time_seconds = elapsed
        summary = build_summary(run, results)
        run.mean_score = summary.mean_combined
        run.pass_rate = summary.pass_rate
        db.update_run(run)

        if output == "json":
            payload = summary.model_dump(mode="json")
            text = json.dumps(payload, indent=2, default=str)
            if output_file:
                Path(output_file).write_text(text)
            else:
                console.print(text)
        else:
            if output_file:
                Path(output_file).write_text(render_table(summary))
            else:
                print_table(summary, console=console)

        if summary.failed == 0:
            raise typer.Exit(code=0)
        raise typer.Exit(code=1)
    finally:
        db.close()


@app.command("judges", help="List free judge models.")
def judges_cmd(
    refresh: bool = typer.Option(False, "--refresh"),
    json_out: bool = typer.Option(False, "--json"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
    judges_cache: Path | None = typer.Option(None, "--judges-cache"),
) -> None:
    """List or refresh the cached judge registry."""
    registry = JudgeRegistry(cache_path=judges_cache) if judges_cache else JudgeRegistry()
    if refresh:
        try:
            registry.fetch(refresh=True, api_key=_get_api_key() or None)
        except Exception as exc:  # noqa: BLE001
            typer.echo(f"refresh failed: {exc}", err=True)
            raise typer.Exit(code=2) from exc
    models = registry.list()
    if json_out:
        typer.echo(json.dumps([m.model_dump() for m in models], indent=2))
        return
    console = Console()
    for m in models:
        console.print(f"{m.id}\tctx={m.context_length}\t{m.name}")


@app.command("report", help="Show a previously stored run.")
def report_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    output: str = typer.Option("table", "--output", help="json, table, or csv"),
    output_file: Path | None = typer.Option(None, "--output-file"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
) -> None:
    """Display a previously stored run."""
    db = Database(db_path)
    try:
        run = db.get_run(run_id)
        if run is None:
            typer.echo(f"run not found: {run_id}", err=True)
            raise typer.Exit(code=1)
        results = db.get_results(run_id)
        summary = build_summary(run, results)
        if output == "json":
            text = json.dumps(summary.model_dump(mode="json"), indent=2, default=str)
        elif output == "csv":
            tmp = Path(output_file) if output_file else Path(f"{run_id}.csv")
            export_results(run, results, tmp, fmt="csv")
            typer.echo(f"wrote {tmp}")
            return
        else:
            text = render_table(summary)
        if output_file:
            Path(output_file).write_text(text)
        else:
            typer.echo(text)
    finally:
        db.close()


@app.command("export", help="Export a run to JSON or CSV.")
def export_cmd(
    run_id: str = typer.Option(..., "--run-id"),
    format: str = typer.Option("json", "--format", help="json or csv"),
    output_file: Path = typer.Option(..., "--output-file"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
) -> None:
    """Export a run's full payload."""
    db = Database(db_path)
    try:
        run = db.get_run(run_id)
        if run is None:
            typer.echo(f"run not found: {run_id}", err=True)
            raise typer.Exit(code=1)
        db.export_run(run_id, output_file, fmt=format)
        typer.echo(f"wrote {output_file}")
    finally:
        db.close()


@app.command("cache", help="Inspect or clear the judge response cache.")
def cache_cmd(
    clear: bool = typer.Option(False, "--clear"),
    stats: bool = typer.Option(False, "--stats"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
) -> None:
    """Inspect or clear the judge response cache."""
    db = Database(db_path)
    try:
        if clear:
            n = db.clear_cache()
            typer.echo(f"cleared {n} cache entries")
        if stats or not clear:
            s = db.cache_stats()
            typer.echo(json.dumps(s, indent=2))
    finally:
        db.close()


if __name__ == "__main__":  # pragma: no cover
    app()
