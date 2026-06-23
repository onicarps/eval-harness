# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `idx_results_record` index on `eval_results(record_id)` to speed up `--resume` lookups.
- Targeted tests for rate limiter, judges cache corruption fallback, CLI confirmation
  abort, CLI `report --output csv`, and DB lookup helpers.

### Changed
- `run` command now prompts for actual confirmation when neither `--yes` nor
  `--quiet` is set (previously printed a question but did not wait for input).
- `JudgeRegistry.list` now falls back to built-in defaults when the on-disk
  cache file is missing, corrupt, empty, or contains invalid entries.
- README pip install command updated to the published PyPI name `llm-eval-harness`.

### Fixed
- Replaced deprecated `datetime.utcnow()` with timezone-aware `datetime.now(UTC)`
  in the database migration writer (Python 3.12 deprecation warning).
- Corrected `progress_cb` type annotation from `callable[[int, int], None]` to
  `Callable[[int, int], None]` in the evaluator.
- Replaced `evaluated_at or 0` sort key with a tz-aware `datetime` sentinel so
  the fallback path can never raise `TypeError` when ordering results.

## [0.1.1] - 2026-06-23

### Added
- `list-runs` CLI command with Rich table output and `--json` flag.
- `RubricTemplate` field validator ensuring `{input}`, `{output}`, `{reference}` placeholders are present.
- CSV column validation: ingest now checks that configured columns exist in the CSV header and logs a clear warning if missing.
- Per-line debug logging in `ingest_jsonl` and `ingest_csv` for skipped rows (empty lines, invalid JSON, missing fields, date filtering).
- `_RateLimiter` now uses a bounded `deque` instead of an unbounded list.
- Schema migration system: `_MIGRATIONS` dict with versioned incremental SQL diffs, replacing the monolithic `_SCHEMA_V1` list.
- `CONTRIBUTING.md` with development setup, code quality guidelines, and PR workflow.
- `mypy` type checking in CI and `[tool.mypy]` config in `pyproject.toml`.
- `_setup_logging(verbose)` helper: `--verbose` flag now enables `DEBUG`-level log output.

### Changed
- `typer.confirm` default changed from `False` to `True` for the evaluation confirmation prompt (pressing Enter now proceeds instead of aborting).
- `estimate_tokens` fallback heuristic changed from word count to `len(text) // 4` (roughly 4 chars per token).
- `_FALLBACK_OBJ_RE` regex changed from greedy `.*` to non-greedy `.*?` to avoid over-matching long LLM reasoning paragraphs.
- `PASS_THRESHOLD` constant documented with inline comment clarifying it is the default for `EvaluatorConfig`.
- `combine_scores` docstring now documents the 50/50 weighting rationale.
- `list_runs` DB query optimized from O(n) round-trips to a single query.

### Fixed
- Removed phantom `--config` CLI flag (declared but never parsed).
- Fixed `__init__.py` version mismatch (`0.1.0` -> `0.1.1`).
- Fixed `AGENTS.md` and `GENERATION_PROMPT.md` env var name: `openrouter_API_KEY` -> `OPENROUTER_API_KEY`.
- Fixed duplicate `list_runs` method definition in `db.py`.

## [0.1.0] - 2026-05-29

### Added
- Typer CLI: `run`, `judges`, `report`, `export`, `cache`
- Pydantic v2 models: `EvalRecord`, `EvalResult`, `EvalRun`, `JudgeCacheEntry`, `RubricTemplate`, `EvalSummary`
- SQLite persistence layer with WAL mode and schema migrations
- JSONL/CSV/stdin ingestion with sampling, since-date filtering, and lenient parsing
- Async LLM-as-judge evaluator with round-robin fallback and response caching
- Rich-based reporter with summary table, ASCII histogram, and JSON/CSV export
- OpenRouter free-model fetcher and on-disk cache
