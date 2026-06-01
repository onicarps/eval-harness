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

## [0.1.0] - 2026-05-29

### Added
- Typer CLI: `run`, `judges`, `report`, `export`, `cache`
- Pydantic v2 models: `EvalRecord`, `EvalResult`, `EvalRun`, `JudgeCacheEntry`, `RubricTemplate`, `EvalSummary`
- SQLite persistence layer with WAL mode and schema migrations
- JSONL/CSV/stdin ingestion with sampling, since-date filtering, and lenient parsing
- Async LLM-as-judge evaluator with round-robin fallback and response caching
- Rich-based reporter with summary table, ASCII histogram, and JSON/CSV export
- OpenRouter free-model fetcher and on-disk cache
