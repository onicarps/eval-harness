"""Typer CLI for eval-harness."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from src.db import Database
from src.evaluator import EvaluatorConfig, LLMEvaluator
from src.ingest import IngestOptions, _parse_since, ingest_file, ingest_stdin
from src.judges import JudgeRegistry
from src.models import BUILTIN_RUBRIC_V1, EvalRun, RunStatus
from src.reporter import (
    build_summary,
    export_results,
    print_table,
    render_table,
)
from src.rubric import RubricManager
from src.calibrate import (
    CalibrationRunner,
    CalibrationSummary,
    render_calibration_json,
    render_calibration_summary,
)
from src.gate import CheckGateResult, GateRunner
from src.rubric import RubricTemplate
from src.trend import MIN_RUNS_DISPLAY, compute_trends

# Load environment variables from .env file
# Check for custom env path via OPENROUTER_ENV_PATH, then fallback to default
env_path = os.environ.get("OPENROUTER_ENV_PATH")
if env_path:
    load_dotenv(Path(env_path))
else:
    # Default to .env in current directory or parent directories
    load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    """Configure root logger level based on --verbose flag."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


DEFAULT_DB_PATH = Path.home() / ".eval-harness" / "eval.db"

app = typer.Typer(
    name="eval-harness",
    help="Evaluate LLM outputs against a dual-dimension rubric.",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _global_opts(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    """Global options for all eval-harness commands."""
    _setup_logging(verbose)


def _get_api_key() -> str:
    """Return the OPENROUTER_API_KEY env value or empty string."""
    return os.environ.get("OPENROUTER_API_KEY", "")


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
    quiet: bool = typer.Option(False, "--quiet"),
    feedback: bool = typer.Option(False, "--feedback", help="Generate improvement suggestions for low-scoring records"),
    compare_judges: bool = typer.Option(False, "--compare-judges", help="Show side-by-side judge comparison"),
    degrade: bool = typer.Option(False, "--degrade", help="Use local heuristic fallback when judge API is unreachable"),
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
        records = list(ingest_stdin(cast(io.TextIOBase, sys.stdin), fmt=format, options=options))
    else:
        if not file.exists():
            console.print(f"[red]file not found: {file}[/red]")
            raise typer.Exit(code=2)
        records = list(ingest_file(file, fmt=format, options=options))

    if not records:
        console.print("[yellow]no records to evaluate[/yellow]")
        logger.warning("No records to evaluate from %s", file)
        raise typer.Exit(code=2)

    if sample is not None and sample <= 0:
        console.print(f"[red]--sample must be a positive integer, got {sample}[/red]")
        raise typer.Exit(code=2)

    if since:
        try:
            _parse_since(since)
        except (ValueError, TypeError):
            console.print(f"[red]invalid --since date: {since}[/red]")
            raise typer.Exit(code=2)

    if dry_run:
        console.print(f"[green]dry-run: {len(records)} record(s) parsed.[/green]")
        logger.info("Dry run: %d records parsed from %s", len(records), file)
        for r in records[:5]:
            console.print(f" - {r.input_text[:80]} -> {r.output_text[:80]}")
        raise typer.Exit(code=0)

    logger.info("Starting evaluation of %d records from %s", len(records), file)
    logger.debug("Options: format=%s, judge=%s, pass_threshold=%s",
                 format, judge, pass_threshold)

    api_key = _get_api_key()
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY is not set. Aborting.[/red]")
        logger.error("OPENROUTER_API_KEY environment variable is not set")
        raise typer.Exit(code=2)

    logger.debug("Using API key (first 8 chars): %s...", api_key[:8] if len(api_key) >= 8 else api_key)

    if not yes and not quiet:
        if not typer.confirm(
            f"Will evaluate {len(records)} record(s). Continue?",
            default=True,
        ):
            console.print("[yellow]aborted by user[/yellow]")
            logger.info("User aborted evaluation")
            raise typer.Exit(code=2)
        logger.info("User confirmed evaluation of %d records", len(records))

    registry = JudgeRegistry(cache_path=judges_cache) if judges_cache else JudgeRegistry()
    judges = _judge_list(judge, registry, no_fallback, max_fallbacks)
    if not judges:
        console.print("[red]no judges available[/red]")
        logger.error("No judges available for evaluation")
        raise typer.Exit(code=2)
    logger.info("Selected judges: %s", judges)

    logger.info("Opening database at %s", db_path)
    db = Database(db_path)
    try:
        logger.debug("Creating EvalRun")
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
        logger.debug("Inserted run %s", run.run_id)
        for rec in records:
            rec.run_id = run.run_id
            db.insert_record(rec)
        run.record_count = len(records)
        db.update_run(run)
        logger.debug("Updated run %s with %d records", run.run_id, len(records))

        logger.info("Initializing LLMEvaluator with %d judges", len(judges))
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
                degrade=degrade,
            ),
        )

        progress_cb = None
        if len(records) > 10 and not quiet:

            def progress_cb(done: int, total: int) -> None:
                # Print at most 20 updates: at completion and evenly spaced
                step = max(1, total // 20)
                if done == total or done % step == 0:
                    console.print(f"[dim]progress: {done}/{total}[/dim]")
                    logger.debug("Evaluation progress: %d/%d", done, total)

        start = time.monotonic()
        logger.info("Starting evaluation of %d records", len(records))
        try:
            results = asyncio.run(
                evaluator.evaluate(run, records, resume=resume, progress_cb=progress_cb)
            )
        except Exception as exc:
            console.print(f"[red]evaluator error: {exc}[/red]")
            logger.error("Evaluator error: %s", exc, exc_info=True)
            run.status = RunStatus.FAILED
            db.update_run(run)
            raise typer.Exit(code=2) from exc
        elapsed = time.monotonic() - start
        logger.info("Evaluation completed in %.2f seconds", elapsed)

        # Generate feedback for failed records if requested
        if feedback and results:
            logger.info("Generating feedback for failed records")
            console.print("[dim]Generating improvement suggestions...[/dim]")
            try:
                asyncio.run(
                    evaluator.generate_all_feedback(run, records, results)
                )
            except Exception as exc:
                console.print(f"[yellow]Feedback generation failed: {exc}[/yellow]")
                logger.error("Feedback generation error: %s", exc, exc_info=True)

        for result in results:
            db.insert_result(result)
        run.status = RunStatus.COMPLETED
        run.eval_time_seconds = elapsed
        summary = build_summary(run, results)
        run.mean_score = summary.mean_combined
        run.pass_rate = summary.pass_rate
        db.update_run(run)
        logger.info("Updated run %s: mean_score=%.3f, pass_rate=%.3f",
                   run.run_id, run.mean_score, run.pass_rate)

        # Display feedback section if any was generated
        if feedback and results:
            feedback_results = [r for r in results if r.feedback]
            if feedback_results:
                console.print("\n[bold cyan]═══ Improvement Suggestions ═══[/bold cyan]")
                for r in feedback_results:
                    rec = next((rec for rec in records if rec.record_id == r.record_id), None)
                    if rec:
                        console.print(f"\n[bold]Record:[/bold] {rec.input_text[:80]}...")
                        console.print(f"[dim]Score: {r.combined_score:.2f}[/dim]")
                        try:
                            fb_data = json.loads(r.feedback)
                            for i, s in enumerate(fb_data.get("suggestions", []), 1):
                                console.print(f"  {i}. {s}")
                        except (json.JSONDecodeError, TypeError):
                            console.print(f"  {r.feedback}")

        if output == "json":
            payload = summary.model_dump(mode="json")
            text = json.dumps(payload, indent=2, default=str)
            if output_file:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                Path(output_file).write_text(text)
            else:
                console.print(text)
        else:
            if output_file:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                Path(output_file).write_text(render_table(summary))
            else:
                print_table(summary, console=console)

        # Display judge comparison if requested
        if compare_judges and results:
            from src.reporter import render_comparison_table
            judges_used = list(dict.fromkeys(r.judge_model for r in results if r.judge_model))
            if len(judges_used) >= 2:
                comparison = render_comparison_table(records, results, judges_used)
                console.print("")
                console.print("[bold cyan]═══ Judge Comparison ═══[/bold cyan]")
                console.print(comparison)
                # Summary stats
                total_recs = len(set(r.record_id for r in results))
                disagree = sum(
                    1 for rec in records
                    if len({r.combined_score for r in results if r.record_id == rec.record_id}) > 1
                )
                console.print("")
                console.print(f"[dim]Records evaluated: {total_recs}[/dim]")
                judges_str = ", ".join(judges_used)
                console.print(f"[dim]Judges used: {judges_str}[/dim]")
                console.print(f"[dim]Records with disagreement: {disagree}/{total_recs}[/dim]")
            else:
                console.print("[yellow]Need 2+ judges for comparison (use --judge to specify multiple)[/yellow]")

        if summary.failed == 0:
            logger.info("Evaluation passed: %d/%d records passed",
                       summary.passed, summary.total)
            raise typer.Exit(code=0)
        logger.info("Evaluation failed: %d/%d records passed",
                   summary.passed, summary.total)
        raise typer.Exit(code=1)
    finally:
        logger.debug("Closing database connection")
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
            tmp.parent.mkdir(parents=True, exist_ok=True)
            export_results(run, results, tmp, fmt="csv")
            typer.echo(f"wrote {tmp}")
            return
        else:
            text = render_table(summary)
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
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
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
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


@app.command("list-runs", help="List previous evaluation runs.")
def list_runs_cmd(
    limit: int = typer.Option(20, "--limit", help="Maximum number of runs to show."),
    json_out: bool = typer.Option(False, "--json"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
) -> None:
    """List previous evaluation runs."""
    db = Database(db_path)
    try:
        runs = db.list_runs(limit=limit)
        if json_out:
            typer.echo(json.dumps([r.model_dump() for r in runs], indent=2, default=str))
            return
        console = Console()
        if not runs:
            console.print("[yellow]no runs found[/yellow]")
            return
        table = Table(title="Evaluation Runs")
        table.add_column("Run ID", style="cyan", no_wrap=True)
        table.add_column("Created", style="dim")
        table.add_column("Records", justify="right")
        table.add_column("Status", style="bold")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Mean Score", justify="right")
        table.add_column("Judge", style="dim")
        for r in runs:
            status_style = "green" if r.status == RunStatus.COMPLETED else "red" if r.status == RunStatus.FAILED else "yellow"
            table.add_row(
                r.run_id[:8],
                r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "",
                str(r.record_count),
                f"[{status_style}]{r.status.value}[/{status_style}]",
                f"{r.pass_rate:.1%}" if r.pass_rate is not None else "-",
                f"{r.mean_score:.3f}" if r.mean_score is not None else "-",
                r.judge_model or "-",
            )
        console.print(table)
    finally:
        db.close()


@app.command("rubric", help="Manage rubric templates.")
def rubric_cmd(
    list_templates: bool = typer.Option(False, "--list", help="List all rubric templates."),
    show: str | None = typer.Option(None, "--show", help="Show a template by ID."),
    create_name: str | None = typer.Option(None, "--create-name", help="Name for new template."),
    create_file: Path | None = typer.Option(None, "--create-file", help="YAML file for new template."),
    delete_id: str | None = typer.Option(None, "--delete", help="Delete a template by ID."),
    json_out: bool = typer.Option(False, "--json"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
) -> None:
    """Manage rubric templates."""
    db = Database(db_path)
    try:
        manager = RubricManager(db)
        if list_templates:
            templates = manager.list_templates()
            if json_out:
                typer.echo(json.dumps([{"template_id": t.template_id, "name": t.name, "is_builtin": t.is_builtin, "created_at": t.created_at.isoformat() if t.created_at else None, "dimensions": [{"name": d.get("name"), "weight": d.get("weight")} for d in t.dimensions]} for t in templates], indent=2))
                return
            console = Console()
            if not templates:
                console.print("[yellow]no rubric templates found[/yellow]")
                return
            table = Table(title="Rubric Templates")
            table.add_column("ID", style="cyan", no_wrap=True)
            table.add_column("Name")
            table.add_column("Built-in", justify="center")
            table.add_column("Dimensions", justify="right")
            table.add_column("Created", style="dim")
            for t in templates:
                table.add_row(
                    t.template_id,
                    t.name,
                    "yes" if t.is_builtin else "no",
                    str(len(t.dimensions)),
                    t.created_at.strftime("%Y-%m-%d") if t.created_at else "-",
                )
            console.print(table)
        elif show:
            template = manager.get_template(show)
            if template is None:
                typer.echo(f"template not found: {show}", err=True)
                raise typer.Exit(code=1)
            assert template is not None
            if json_out:
                typer.echo(json.dumps({"template_id": template.template_id, "name": template.name, "yaml_content": template.yaml_content, "is_builtin": template.is_builtin, "dimensions": template.dimensions, "scoring": template.scoring}, indent=2))
            else:
                console = Console()
                console.print(f"[bold]{template.name}[/bold] ({template.template_id})")
                console.print(f"Built-in: {'yes' if template.is_builtin else 'no'}")
                console.print(f"Dimensions: {len(template.dimensions)}")
                console.print()
                console.print(template.yaml_content)
        elif create_name and create_file:
            if not create_file.exists():
                typer.echo(f"file not found: {create_file}", err=True)
                raise typer.Exit(code=2)
            yaml_content = create_file.read_text()
            try:
                t = manager.create_template(create_name, yaml_content)
                console = Console()
                console.print(f"[green]created template: {t.template_id}[/green]")
            except ValueError as exc:
                typer.echo(f"error: {exc}", err=True)
                raise typer.Exit(code=1) from exc
        elif delete_id:
            try:
                if manager.delete_template(delete_id):
                    console = Console()
                    console.print(f"[green]deleted template: {delete_id}[/green]")
                else:
                    typer.echo(f"template not found: {delete_id}", err=True)
                    raise typer.Exit(code=1) from None
            except ValueError as exc:
                typer.echo(f"error: {exc}", err=True)
                raise typer.Exit(code=1) from exc
        else:
            typer.echo("Use --list, --show <id>, --create-name <name> --create-file <path>, or --delete <id>", err=True)
            raise typer.Exit(code=1) from None
    finally:
        db.close()


@app.command("trend", help="Show score timeline and regression detection.")
def trend_cmd(
    rubric_template_id: str | None = typer.Option(None, "--rubric", help="Filter by rubric template ID."),
    judge_model: str | None = typer.Option(None, "--judge", help="Filter by judge model."),
    since: str | None = typer.Option(None, "--since", help="Only show runs after this date (ISO-8601)."),
    json_out: bool = typer.Option(False, "--json"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
) -> None:
    """Show score timeline and regression detection."""
    db = Database(db_path)
    try:
        since_dt = None
        if since:
            try:
                since_dt = datetime.fromisoformat(since)
                if since_dt.tzinfo is None:
                    since_dt = since_dt.replace(tzinfo=UTC)
            except ValueError:
                typer.echo(f"invalid date: {since}", err=True)
                raise typer.Exit(code=2) from None

        result = compute_trends(
            db,
            rubric_template_id=rubric_template_id,
            judge_model=judge_model,
            since=since_dt,
        )

        if json_out:
            typer.echo(json.dumps({
                "total_runs": result.total_runs,
                "has_regression": result.has_regression,
                "mean_score_overall": result.mean_score_overall,
                "latest_score": result.latest_score,
                "earliest_score": result.earliest_score,
                "points": [
                    {
                        "run_id": p.run_id[:8],
                        "created_at": p.created_at.isoformat() if p.created_at else None,
                        "mean_score": p.mean_score,
                        "pass_rate": p.pass_rate,
                        "record_count": p.record_count,
                        "is_regression": p.is_regression,
                    }
                    for p in result.points
                ],
            }, indent=2, default=str))
            return

        console = Console()
        if result.total_runs < MIN_RUNS_DISPLAY:
            console.print(
                f"[yellow]need at least {MIN_RUNS_DISPLAY} completed runs for trend display, "
                f"found {result.total_runs}[/yellow]"
            )
            return

        # Summary
        console.print(f"[bold]Score Trend[/bold] — {result.total_runs} runs")
        if result.mean_score_overall is not None:
            console.print(f"  Mean score: {result.mean_score_overall:.3f}")
        if result.latest_score is not None and result.earliest_score is not None:
            delta = result.latest_score - result.earliest_score
            direction = "up" if delta >= 0 else "down"
            console.print(
                f"  Earliest: {result.earliest_score:.3f} → Latest: {result.latest_score:.3f} "
                f"({direction} {abs(delta):.3f})"
            )
        if result.has_regression:
            console.print("  [red]⚠ Regression detected[/red]")

        # Timeline table
        table = Table(title="Run Timeline")
        table.add_column("Run ID", style="cyan", no_wrap=True)
        table.add_column("Date", style="dim")
        table.add_column("Records", justify="right")
        table.add_column("Mean Score", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("", justify="center")

        for p in result.points:
            score_str = f"{p.mean_score:.3f}" if p.mean_score is not None else "-"
            pass_str = f"{p.pass_rate:.1%}" if p.pass_rate is not None else "-"
            regression_marker = "[red]▼[/red]" if p.is_regression else ""
            date_str = p.created_at.strftime("%Y-%m-%d") if p.created_at != datetime.min else "-"
            table.add_row(
                p.run_id[:8],
                date_str,
                str(p.record_count),
                score_str,
                pass_str,
                regression_marker,
            )
        console.print(table)
    finally:
        db.close()


@app.command("calibrate", help="Measure inter-judge agreement on a dataset.")
def calibrate_cmd(
    file: Path = typer.Argument(..., help="Input file path (use '-' for stdin)."),
    format: str = typer.Option("jsonl", "--format", help="jsonl or csv"),
    input_col: str = typer.Option("input", "--input-col"),
    output_col: str = typer.Option("output", "--output-col"),
    reference_col: str = typer.Option("reference", "--reference-col"),
    sample: int | None = typer.Option(None, "--sample"),
    since: str | None = typer.Option(None, "--since"),
    limit: int | None = typer.Option(None, "--limit"),
    output_file: Path | None = typer.Option(None, "--output-file"),
    json_out: bool = typer.Option(False, "--json"),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
    judges_cache: Path | None = typer.Option(None, "--judges-cache"),
) -> None:
    """Run all records through every judge and report disagreement."""
    console = Console()
    options = IngestOptions(
        input_col=input_col,
        output_col=output_col,
        reference_col=reference_col,
        sample=sample,
        since=since,
        limit=limit,
    )
    if str(file) == "-":
        import io
        records = list(ingest_stdin(cast(io.TextIOBase, sys.stdin), fmt=format, options=options))
    else:
        if not file.exists():
            console.print(f"[red]file not found: {file}[/red]")
            raise typer.Exit(code=2)
        records = list(ingest_file(file, fmt=format, options=options))

    if not records:
        console.print("[yellow]no records to calibrate[/yellow]")
        raise typer.Exit(code=2)

    api_key = _get_api_key()
    if not api_key:
        console.print("[red]OPENROUTER_API_KEY is not set. Aborting.[/red]")
        raise typer.Exit(code=2)

    registry = JudgeRegistry(cache_path=judges_cache) if judges_cache else JudgeRegistry()
    judge_models = [m.id for m in registry.list()]
    if not judge_models:
        console.print("[red]no judges available[/red]")
        raise typer.Exit(code=2)

    db = Database(db_path)
    try:
        runner = CalibrationRunner(
            db=db,
            api_key=api_key,
            judges=judge_models,
        )

        summary = runner.run(records)

        if json_out:
            text = render_calibration_json(summary)
            if output_file:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                Path(output_file).write_text(text)
            else:
                console.print(text)
        else:
            text = render_calibration_summary(summary)
            if output_file:
                Path(output_file).parent.mkdir(parents=True, exist_ok=True)
                Path(output_file).write_text(text)
            else:
                console.print(text)

    finally:
        db.close()


@app.command("gate", help="CI/CD quality gate — check if a run meets a threshold.")
def gate_cmd(
    run_id: str | None = typer.Option(None, "--run-id", help="Run ID to check against threshold."),
    threshold: float = typer.Option(0.7, "--threshold", help="Pass rate threshold (0.0–1.0)."),
    suggest_baseline: bool = typer.Option(False, "--suggest-baseline", help="Analyze history and suggest a baseline."),
    json_out: bool = typer.Option(False, "--json", help="Output result as JSON."),
    output_file: Path | None = typer.Option(None, "--output-file", help="Write output to file."),
    db_path: Path = typer.Option(DEFAULT_DB_PATH, "--db"),
) -> None:
    """CI/CD quality gate — check if a run meets a threshold, or suggest one from history."""
    db = Database(db_path)
    try:
        if suggest_baseline:
            runner = GateRunner(db)
            suggestion = runner.suggest_baseline()
            if suggestion is None:
                typer.echo("no completed runs found to suggest baseline from", err=True)
                raise typer.Exit(code=2) from None
            if json_out:
                typer.echo(json.dumps(suggestion, indent=2))
            else:
                console = Console()
                console.print("[bold]Suggested Baseline[/bold]")
                console.print(f"  Recommended: {suggestion['recommended_baseline']:.1%}")
                console.print(f"  Median pass rate: {suggestion['median_pass_rate']:.1%}")
                console.print(f"  Runs analyzed: {suggestion['runs_analyzed']}")
                console.print(f"  Range: {suggestion['min_pass_rate']:.1%} – {suggestion['max_pass_rate']:.1%}")
                if suggestion.get("note"):
                    console.print(f"  [dim]{suggestion['note']}[/dim]")
            return

        if run_id is None:
            typer.echo("use --run-id <id> to check a run, or --suggest-baseline", err=True)
            raise typer.Exit(code=2) from None

        runner = GateRunner(db)
        result = runner.check(run_id, threshold=threshold)

        if json_out:
            text = result.to_json()
        else:
            text = result.to_text()

        if output_file:
            Path(output_file).write_text(text)
        else:
            typer.echo(text)

        raise typer.Exit(code=result.exit_code) from None
    except ValueError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from None
    finally:
        db.close()


if __name__ == "__main__":  # pragma: no cover
    app()
