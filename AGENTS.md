# AGENTS.md — Eval Harness Project

## Project
Python CLI tool that evaluates LLM outputs from production logs AND agent behavior in environments.

## Phase 2 Plan
See `PLAN_PHASE2.md` for the full Phase 2 implementation plan.

**Phase 2A (v0.2.0) — Production Intelligence:**
- `eval-harness trend` — score timeline + regression detection
- `eval-harness rubric` — domain-specific rubric templates
- `eval-harness gate` — CI/CD quality gate
- Multi-judge comparison mode

**Phase 2B (v0.3.0) — Agent Evaluation:**
- `eval-harness agent eval` — environment-based agent evaluation
- Task suite format + built-in suites
- Environment adapters (python-repl, bash-sandbox, mock)

## Tech Stack
- Python 3.11+
- hatchling build backend (pyproject.toml)
- Typer + Rich (CLI + terminal output)
- Pydantic v2 (data validation)
- sqlite3 stdlib (sync, WAL mode)
- httpx (async HTTP for judge API calls)
- tiktoken (token estimation)
- pytest + pytest-httpx + ruff

## Project Structure
```
eval-harness/
├── src/
│   ├── __init__.py
│   ├── cli.py              # Typer app: run, judges, report, export, cache
│   ├── models.py           # Pydantic: EvalRecord, EvalResult, EvalRun, JudgeCacheEntry, RubricTemplate
│   ├── db.py               # SQLite: schema versioning, CRUD, migrations, export
│   ├── ingest.py           # JSONL + CSV + stdin parser: --sample, --since, lenient parsing
│   ├── evaluator.py        # LLM-as-judge: async batch, round-robin fallback, response caching
│   ├── reporter.py         # Rich terminal tables, ASCII histogram, JSON/CSV export
│   └── judges.py           # OpenRouter free model fetcher/cache
├── tests/
│   ├── conftest.py         # Fixtures, mock judge responses, temp DB setup
│   ├── test_models.py
│   ├── test_db.py
│   ├── test_ingest.py      # VCR cassettes for mock LLM responses
│   ├── test_evaluator.py   # pytest-httpx for mock HTTP
│   ├── test_reporter.py
│   ├── test_cli.py         # Typer CliRunner, end-to-end
│   └── test_judges.py
├── .github/workflows/ci.yml
├── pyproject.toml
├── README.md
├── .gitignore
├── .env.example
├── LICENSE (MIT)
└── CHANGELOG.md
```

## Database Schema (v1)
```sql
CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);

CREATE TABLE eval_runs (
    run_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, config_json TEXT NOT NULL,
    record_count INTEGER DEFAULT 0, rubric_id TEXT DEFAULT 'faithfulness-v1',
    judge_model TEXT, status TEXT DEFAULT 'running', completed_at TEXT,
    mean_score REAL, pass_rate REAL, eval_time_seconds REAL
);

CREATE TABLE eval_records (
    record_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES eval_runs(run_id),
    input_text TEXT NOT NULL, output_text TEXT NOT NULL, reference_text TEXT,
    source_file TEXT, metadata_json TEXT, created_at TEXT NOT NULL
);
CREATE INDEX idx_records_run ON eval_records(run_id);

CREATE TABLE eval_results (
    result_id TEXT PRIMARY KEY, record_id TEXT NOT NULL REFERENCES eval_records(record_id),
    run_id TEXT NOT NULL REFERENCES eval_runs(run_id), rubric_id TEXT DEFAULT 'faithfulness-v1',
    rubric_version TEXT DEFAULT '1.0', faithfulness REAL NOT NULL, task_completion REAL NOT NULL,
    combined_score REAL NOT NULL, pass_fail TEXT NOT NULL, reasoning TEXT DEFAULT '',
    faithfulness_reasoning TEXT DEFAULT '', task_completion_reasoning TEXT DEFAULT '',
    judge_model TEXT NOT NULL, judge_fallbacks INTEGER DEFAULT 0, judge_tried TEXT DEFAULT '[]',
    tokens_estimated INTEGER, evaluated_at TEXT NOT NULL, error TEXT
);
CREATE INDEX idx_results_run ON eval_results(run_id);

CREATE TABLE judge_cache (
    cache_key TEXT PRIMARY KEY, model_id TEXT NOT NULL, rubric_version TEXT NOT NULL,
    response_json TEXT NOT NULL, created_at TEXT NOT NULL, hits INTEGER DEFAULT 1
);
```

## Input Schema (JSONL)
```json
{"input": "user prompt", "output": "model response", "reference": "optional ground truth"}
```

## Judge Output Schema
```json
{"faithfulness": 0.0-1.0, "task_completion": 0.0-1.0, "reasoning": "str", "faithfulness_reasoning": "str", "task_completion_reasoning": "str"}
```
Combined: 0.5 * faithfulness + 0.5 * task_completion. Pass/fail threshold: 0.7.

## CLI Commands
- `eval-harness run <file>` — primary: ingest + evaluate + report. Flags: --format jsonl|csv, --input-col, --output-col, --reference-col, --sample N, --since DATE, --limit N, --judge MODEL, --no-fallback, --max-fallbacks N, --pass-threshold FLOAT, --output json|table, --output-file PATH, --dry-run, --resume, --timeout SECONDS, --rpm-limit INT, --yes, --verbose, --quiet
- `eval-harness judges` — list free judge models. Flags: --refresh, --json
- `eval-harness list-runs [--limit N] [--json]` — list previous evaluation runs
- `eval-harness report --run-id UUID` — show results. Flags: --output json|table|csv, --output-file PATH
- `eval-harness export --run-id UUID --format json|csv --output-file PATH`
- `eval-harness cache [--clear] [--stats]`

Exit codes: 0=all pass, 1=any failures, 2=evaluator error

## API Key
Read from OPENROUTER_API_KEY env var (not hardcoded).

## Rules
- TDD: write failing test BEFORE implementation
- Type hints everywhere
- Google-style docstrings
- ruff check + ruff format before every commit
- Commit after every task
