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

## [0.2.1] - 2026-06-25

### Added
- **Phase 2B: Agent Evaluation** — environment-based agent evaluation against task suites.
- `eval-harness agent eval` command — run agents against task suites with trajectory scoring and LLM-judge evaluation.
- `eval-harness agent-report` command — view past agent evaluation runs in table, JSON, or CSV format.
- `eval-harness agent-export` command — export agent evaluation results to file (JSON/CSV).
- `eval-harness agent-list-suites` command — list all available built-in task suites.
- `src/agent.py` — Abstract Agent ABC with async `start`/`act`/`stop` lifecycle; `SubprocessAgent` (CLI subprocess via stdin/stdout NDJSON); `PythonAgent` (in-process async function).
- `src/agent_evaluator.py` — `AgentEvaluator` with trajectory scoring (exact-match + efficiency + LLM-judge), `EvaluatorConfig`.
- `src/agent_models.py` — Pydantic models: `TaskSuite`, `TaskStep`, `AgentRun`, `AgentResult`, `ScoringSummary`, and supporting enums.
- `src/task_suite.py` — 5 built-in task suites (echo-v1, math-v1, file-read-v1, string-reversal-v1, multi-step-v1) plus `llm-judge-v1` suite.
- LLM-judge scoring integration — `score_step()` method in `LLMEvaluator` reuses existing judge infrastructure for open-ended task scoring.
- Database migration v4 — new `agent_runs`, `agent_results`, `agent_task_suites` tables; schema version bumped to 4.
- 102 new tests across 5 test files covering agent lifecycle, evaluator, models, task suites, and CLI commands.
- GitHub OIDC publish workflow (`.github/workflows/publish.yml`) for automated PyPI releases.

### Changed
- `CURRENT_SCHEMA_VERSION` bumped to 4.

## [0.2.0] - 2026-06-23

### Added
- `eval-harness calibrate` command — measure inter-judge agreement by running all records through every available judge model. Options: `--format`, `--sample`, `--since`, `--limit`, `--output-file`, `--json`.
- `eval-harness gate` command — CI/CD quality gate that checks a run's pass rate against a threshold. Options: `--run-id`, `--threshold`, `--suggest-baseline`, `--json`, `--output-file`.
- `--feedback` flag on `run` — generates improvement suggestions for low-scoring records using the judge model.
- `--compare-judges` flag on `run` — displays a side-by-side comparison table of scores from multiple judges.
- `--degrade` flag on `run` — uses a local heuristic fallback when the judge API is unreachable, allowing evaluations to continue offline.
- `src/calibrate.py` module — `CalibrationRunner` and `CalibrationSummary` for inter-judge agreement measurement.
- `src/gate.py` module — `GateRunner` and `CheckGateResult` for CI/CD quality gates with baseline suggestion.
- `src/rubric.py` module — `RubricTemplate` and `RubricManager` classes for rubric template CRUD.
- `src/trend.py` module — `TrendPoint`, `TrendResult`, `compute_trends()` with regression detection.
- `rubric_templates` database table with 5 built-in templates: `faithfulness-v1`, `safety-v1`, `accuracy-v1`, `conciseness-v1`, `custom-v1`.
- `rubric_template_id` column on `eval_runs` for tracking which rubric was used per run.
- `Database.rollback(target_version)` method for migration downgrades.
- `pyyaml>=6.0` dependency for YAML parsing of rubric templates.
- 18 new tests in `test_rubric.py` for template CRUD and validation.
- 12 new tests in `test_calibrate.py` for inter-judge agreement measurement.
- 10 new tests in `test_gate.py` for CI/CD quality gate logic.

### Changed
- `CURRENT_SCHEMA_VERSION` bumped to 2.
- `insert_run`, `update_run`, `get_run`, `list_runs` updated to handle `rubric_template_id`.
- `--verbose` is now a global option (via `@app.callback()`) available on all commands.
- README expanded with badges, quickstart, full command reference with examples, configuration, troubleshooting, and CI/CD example.
- `.env.example` updated with all relevant environment variables.
- CLI help text improved across all commands with clearer descriptions and usage guidance.

### Fixed
- Judge registry now falls back to built-in defaults when the on-disk cache file is missing, corrupt, empty, or contains invalid entries.
- Fixed `evaluated_at` sort key to use a tz-aware `datetime` sentinel, preventing `TypeError` on mixed-timezone databases.
- Replaced deprecated `datetime.utcnow()` with `datetime.now(UTC)` for Python 3.12 compatibility.
- Fixed `progress_cb` type annotation from `callable` to `Callable` for proper type checking.

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
- Fixed `AGENTS.md` and `GENERATION_PROMPT.md` env var name: `OPENROUTER_API_KEY` -> `OPENROUTER_API_KEY`.
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
