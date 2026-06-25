# eval-harness

[![CI](https://github.com/onicarps/eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/onicarps/eval-harness/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llm-eval-harness)](https://pypi.org/project/llm-eval-harness/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

**Evaluate LLM outputs from production logs AND agent behavior in environments against a dual-dimension rubric (faithfulness + task completion).**

eval-harness ingests JSONL/CSV logs, scores each record using LLM-as-judge via OpenRouter, and produces a rich terminal report — or exports JSON/CSV for CI pipelines. The `agent` command runs agents in sandboxed environments against task suites with trajectory scoring. Built for teams who need fast, repeatable quality checks on production LLM traffic and agent behavior.

## Features

- **Dual-dimension scoring** — faithfulness + task completion, combined 50/50
- **Multi-judge support** — round-robin fallback across free OpenRouter models
- **Regression detection** — `trend` command tracks score changes over time
- **CI/CD ready** — `gate` command with pass/fail exit codes and threshold suggestions
- **Rubric templates** — manage reusable rubric definitions (faithfulness, safety, accuracy, conciseness)
- **Response caching** — judge responses cached in SQLite; skip re-evaluation of unchanged records
- **Calibration mode** — measure inter-judge agreement across all available models
- **Degraded mode** — local heuristic fallback when the judge API is unreachable
- **Agent evaluation** — `agent` command runs agents in environment adapters (subprocess, Python REPL) against task suites with trajectory scoring
- **Task suites** — built-in suites (echo, math, file-read, string-reversal, multi-step) plus LLM-judge scoring for open-ended tasks

## Install

```bash
# From PyPI
pip install llm-eval-harness

# From source (development)
pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. Set your API key
export OPENROUTER_API_KEY=sk-or-...

# 2. Run an evaluation
eval-harness run path/to/logs.jsonl --judge meta-llama/llama-3.1-8b-instruct:free

# 3. View a previous run
eval-harness report --run-id <RUN_ID>

# 4. Export results
eval-harness export --run-id <RUN_ID> --format json --output-file results.json

# 5. Run an agent evaluation
eval-harness agent --suite echo-v1 --agent-subprocess "python my_agent.py"

# 6. List available task suites
eval-harness agent-list-suites

# 7. View a previous agent run
eval-harness agent-report --run-id <AGENT_RUN_ID>
```

### Input Format

JSONL (one JSON object per line) or CSV. Fields default to `input`, `output`, `reference` (reference is optional):

```json
{"input": "user prompt", "output": "model response", "reference": "optional ground truth"}
```

## Commands

### `run` — Ingest, evaluate, and report

```bash
eval-harness run <file> [options]
```

| Option | Default | Description |
|---|---|---|
| `--format` | `jsonl` | Input format: `jsonl` or `csv` |
| `--input-col` | `input` | Column name for the user prompt |
| `--output-col` | `output` | Column name for the model response |
| `--reference-col` | `reference` | Column name for the ground truth |
| `--sample` | — | Randomly sample N records for evaluation |
| `--since` | — | Only evaluate records after this date (ISO-8601, e.g. `2026-06-01`) |
| `--limit` | — | Maximum number of records to evaluate |
| `--judge` | — | Judge model ID (e.g. `meta-llama/llama-3.1-8b-instruct:free`). If omitted, uses all available free models |
| `--no-fallback` | `false` | Disable round-robin fallback to other judges |
| `--max-fallbacks` | `3` | Maximum number of fallback judges to try |
| `--pass-threshold` | `0.7` | Score threshold for pass/fail (0.0–1.0) |
| `--output` | `table` | Output format: `table` (rich terminal) or `json` |
| `--output-file` | — | Write output to a file instead of stdout |
| `--dry-run` | `false` | Parse input and show record count without evaluating |
| `--resume` | `false` | Skip records that were already evaluated in a previous run |
| `--timeout` | `60` | Per-request timeout in seconds |
| `--rpm-limit` | — | Rate limit (requests per minute) for judge API calls |
| `--yes` / `-y` | `false` | Skip confirmation prompt |
| `--quiet` | `false` | Suppress progress output |
| `--feedback` | `false` | Generate improvement suggestions for low-scoring records |
| `--compare-judges` | `false` | Show side-by-side judge comparison table |
| `--degrade` | `false` | Use local heuristic fallback when judge API is unreachable |
| `--db` | `~/.eval-harness/eval.db` | Path to SQLite database |
| `--judges-cache` | — | Path to judge registry cache file |

**Examples:**

```bash
# Basic evaluation with default settings
eval-harness run logs.jsonl

# Use a specific judge with a higher pass threshold
eval-harness run logs.jsonl --judge meta-llama/llama-3.1-8b-instruct:free --pass-threshold 0.8

# Quick dry-run to validate input
eval-harness run logs.jsonl --dry-run

# Evaluate a random sample of 50 records
eval-harness run logs.jsonl --sample 50

# CSV with custom column names
eval-harness run logs.csv --format csv --input-col prompt --output-col response --reference-col expected

# Export JSON results to a file
eval-harness run logs.jsonl --output json --output-file results.json

# Resume a previously interrupted run
eval-harness run logs.jsonl --resume

# Generate improvement suggestions for failed records
eval-harness run logs.jsonl --feedback

# Compare scores across multiple judges
eval-harness run logs.jsonl --compare-judges

# Pipe from stdin
cat logs.jsonl | eval-harness run -
```

### `judges` — List free judge models

```bash
eval-harness judges [--refresh] [--json]
```

Lists available free judge models from OpenRouter. Results are cached locally; use `--refresh` to force an update.

```bash
eval-harness judges                  # list models in a table
eval-harness judges --json           # output as JSON
eval-harness judges --refresh        # force refresh from API
```

### `list-runs` — List previous evaluation runs

```bash
eval-harness list-runs [--limit N] [--json]
```

| Option | Default | Description |
|---|---|---|
| `--limit` | `20` | Maximum number of runs to show |
| `--json` | `false` | Output as JSON |
| `--db` | `~/.eval-harness/eval.db` | Path to SQLite database |

```bash
eval-harness list-runs               # show last 20 runs
eval-harness list-runs --limit 50    # show last 50 runs
eval-harness list-runs --json        # output as JSON
```

### `report` — Show results for a previous run

```bash
eval-harness report --run-id <ID> [--output table|json|csv] [--output-file <path>]
```

| Option | Default | Description |
|---|---|---|
| `--run-id` | *required* | Run ID to display |
| `--output` | `table` | Output format: `table`, `json`, or `csv` |
| `--output-file` | — | Write output to a file |
| `--db` | `~/.eval-harness/eval.db` | Path to SQLite database |

```bash
eval-harness report --run-id abc123
eval-harness report --run-id abc123 --output json
eval-harness report --run-id abc123 --output csv --output-file report.csv
```

### `export` — Export run results

```bash
eval-harness export --run-id <ID> --format json|csv --output-file <path>
```

| Option | Default | Description |
|---|---|---|
| `--run-id` | *required* | Run ID to export |
| `--format` | `json` | Export format: `json` or `csv` |
| `--output-file` | *required* | Output file path |
| `--db` | `~/.eval-harness/eval.db` | Path to SQLite database |

```bash
eval-harness export --run-id abc123 --format json --output-file results.json
eval-harness export --run-id abc123 --format csv --output-file results.csv
```

### `cache` — Manage the judge response cache

```bash
eval-harness cache [--stats] [--clear]
```

| Option | Description |
|---|---|
| `--stats` | Show cache statistics (entry count, hit rate, size) |
| `--clear` | Remove all cached judge responses |

```bash
eval-harness cache --stats            # show cache stats
eval-harness cache --clear            # clear all cached responses
eval-harness cache                    # default: show stats
```

### `trend` — Score timeline and regression detection

```bash
eval-harness trend [--rubric <id>] [--judge <model>] [--since <date>] [--json]
```

| Option | Description |
|---|---|
| `--rubric` | Filter by rubric template ID |
| `--judge` | Filter by judge model ID |
| `--since` | Only show runs after this date (ISO-8601, e.g. `2026-06-01`) |
| `--json` | Output as JSON |
| `--db` | Path to SQLite database |

Requires at least 2 completed runs to display trends.

```bash
eval-harness trend                              # show all runs
eval-harness trend --rubric faithfulness-v1     # filter by rubric
eval-harness trend --since 2026-06-01           # recent runs only
eval-harness trend --json                       # output as JSON
```

### `rubric` — Manage rubric templates

```bash
eval-harness rubric [--list] [--show <id>] [--create-name <n> --create-file <path>] [--delete <id>] [--json]
```

| Option | Description |
|---|---|
| `--list` | List all rubric templates |
| `--show` | Show a specific template by ID |
| `--create-name` | Name for a new template (use with `--create-file`) |
| `--create-file` | Path to a YAML file defining the template (use with `--create-name`) |
| `--delete` | Delete a template by ID (built-in templates cannot be deleted) |
| `--json` | Output as JSON |
| `--db` | Path to SQLite database |

```bash
eval-harness rubric --list                              # list all templates
eval-harness rubric --show faithfulness-v1              # show a template
eval-harness rubric --list --json                       # list as JSON
eval-harness rubric --create-name "my-rubric" --create-file rubric.yaml
eval-harness rubric --delete my-rubric                  # delete a custom template
```

### `calibrate` — Measure inter-judge agreement

```bash
eval-harness calibrate <file> [--format jsonl|csv] [--sample N] [--since DATE] [--limit N] [--output-file <path>] [--json]
```

Runs every record through every available judge model and reports disagreement statistics. Useful for validating that your chosen judge model agrees with others.

| Option | Default | Description |
|---|---|---|
| `--format` | `jsonl` | Input format: `jsonl` or `csv` |
| `--input-col` | `input` | Column name for the user prompt |
| `--output-col` | `output` | Column name for the model response |
| `--reference-col` | `reference` | Column name for the ground truth |
| `--sample` | — | Randomly sample N records |
| `--since` | — | Only evaluate records after this date |
| `--limit` | — | Maximum number of records |
| `--output-file` | — | Write output to a file |
| `--json` | `false` | Output as JSON |
| `--db` | `~/.eval-harness/eval.db` | Path to SQLite database |
| `--judges-cache` | — | Path to judge registry cache file |

```bash
eval-harness calibrate logs.jsonl
eval-harness calibrate logs.jsonl --sample 20 --json
```

### `gate` — CI/CD quality gate

```bash
eval-harness gate --run-id <ID> [--threshold 0.7] [--suggest-baseline] [--json] [--output-file <path>]
```

Checks whether a run's pass rate meets a threshold. Returns exit code 0 (pass) or 1 (fail), making it suitable for CI pipelines.

| Option | Default | Description |
|---|---|---|
| `--run-id` | — | Run ID to check (required unless `--suggest-baseline`) |
| `--threshold` | `0.7` | Pass rate threshold (0.0–1.0) |
| `--suggest-baseline` | `false` | Analyze run history and suggest a threshold |
| `--json` | `false` | Output as JSON |
| `--output-file` | — | Write output to a file |
| `--db` | `~/.eval-harness/eval.db` | Path to SQLite database |

```bash
# Check a run against the default threshold (0.7)
eval-harness gate --run-id abc123

# Use a custom threshold
eval-harness gate --run-id abc123 --threshold 0.8

# Get a suggested baseline from historical runs
eval-harness gate --suggest-baseline

# Use in CI
eval-harness gate --run-id abc123 --json --output-file gate-result.json
```

### `agent` — Evaluate agent behavior in environments

```bash
eval-harness agent --suite <suite> [--agent-subprocess <cmd> | --agent-python <path>] [--max-steps N] [--timeout N] [--pass-threshold 0.7] [--output table|json] [--output-file <path>] [--db <path>]
```

Runs an agent against a task suite, recording trajectories and scoring each step. Provides either `--agent-subprocess` (CLI subprocess via NDJSON over stdin/stdout) or `--agent-python` (in-process async function).

| Option | Default | Description |
|---|---|---|
| `--suite` | `echo-v1` | Task suite ID to run (use `agent-list-suites` to see available) |
| `--agent-subprocess` | — | Command to launch a subprocess agent (NDJSON over stdin/stdout) |
| `--agent-python` | — | Python agent module path (e.g. `mymodule:my_agent_func`) |
| `--max-steps` | `10` | Maximum number of steps per task before marking as failed |
| `--timeout` | `60` | Per-step timeout in seconds |
| `--pass-threshold` | `0.7` | Score threshold (0.0–1.0) for pass/fail |
| `--output` | `table` | Output format: `table` or `json` |
| `--output-file` | — | Write results to a file |
| `--db` | `~/.eval-harness/eval.db` | Path to SQLite database |

```bash
# Run the echo-v1 suite with a subprocess agent
eval-harness agent --suite echo-v1 --agent-subprocess "python my_agent.py"

# Use a Python agent with custom timeout
eval-harness agent --suite math-v1 --agent-python "my_agent:solve" --max-steps 20

# Export JSON results to a file
eval-harness agent --suite echo-v1 --agent-subprocess "python agent.py" --output json --output-file agent-results.json
```

### `agent-list-suites` — List available task suites

```bash
eval-harness agent-list-suites [--json]
```

Lists all built-in task suites available for agent evaluation.

| Option | Default | Description |
|---|---|---|
| `--json` | `false` | Output as JSON |

```bash
eval-harness agent-list-suites              # list suites in a table
eval-harness agent-list-suites --json       # list as JSON
```

### `agent-report` — Show results for a previous agent run

```bash
eval-harness agent-report --run-id <ID> [--output table|json|csv] [--output-file <path>] [--db <path>]
```

Displays detailed trajectory results from a stored agent evaluation run, including per-step scores, success/failure, and timing.

| Option | Default | Description |
|---|---|---|
| `--run-id` | *required* | Agent run ID to display |
| `--output` | `table` | Output format: `table`, `json`, or `csv` |
| `--output-file` | — | Write report to a file |
| `--db` | `~/.eval-harness/eval.db` | Path to SQLite database |

```bash
eval-harness agent-report --run-id abc123                   # table output
eval-harness agent-report --run-id abc123 --output json     # JSON output
eval-harness agent-report --run-id abc123 --output csv --output-file report.csv
```

### `agent-export` — Export agent run results

```bash
eval-harness agent-export --run-id <ID> --format json|csv --output-file <path> [--db <path>]
```

Exports full trajectory data from a completed agent run. JSON export includes per-step detail; CSV export is a flat table.

| Option | Default | Description |
|---|---|---|
| `--run-id` | *required* | Agent run ID to export |
| `--format` | `json` | Export format: `json` or `csv` |
| `--output-file` | *required* | Output file path |
| `--db` | `~/.eval-harness/eval.db` | Path to SQLite database |

```bash
eval-harness agent-export --run-id abc123 --format json --output-file results.json
eval-harness agent-export --run-id abc123 --format csv --output-file results.csv
```

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | All records passed (score >= threshold) |
| `1` | One or more records failed |
| `2` | Evaluator error (API key missing, file not found, etc.) |

## Configuration

Configuration is handled via environment variables. See `.env.example` for all options.

| Variable | Required | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | Yes | Your OpenRouter API key ([get one](https://openrouter.ai/keys)) |
| `OPENROUTER_ENV_PATH` | No | Path to a `.env` file (default: `.env` in current directory) |

## CI/CD Example

```yaml
# .github/workflows/eval.yml
name: LLM Evaluation
on:
  push:
    branches: [main]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install llm-eval-harness
      - run: eval-harness run eval/cases.jsonl --pass-threshold 0.7 --yes
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
      - run: eval-harness gate --run-id $(eval-harness list-runs --json | jq -r '.[0].run_id') --threshold 0.7
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
```

## Troubleshooting

### `OPENROUTER_API_KEY is not set`
Set the environment variable before running any command:
```bash
export OPENROUTER_API_KEY=sk-or-...
```
Or create a `.env` file in your project root (see `.env.example`).

### `no judges available`
The judge registry cache may be empty or corrupt. Force a refresh:
```bash
eval-harness judges --refresh
```
If the API is unreachable, the built-in judge list is used as a fallback.

### `file not found` or `no records to evaluate`
- Check that the file path is correct.
- For stdin input, use `-` as the file argument: `cat data.jsonl | eval-harness run -`
- Verify the input format matches `--format` (jsonl or csv).
- If using CSV, confirm column names match `--input-col` / `--output-col`.

### `evaluator error: ...` (exit code 2)
- Check your API key is valid and has available credits.
- Try `--degrade` to use a local heuristic fallback.
- Use `--verbose` (`-v`) for debug-level logging.

### Slow evaluations
- Use `--sample N` to evaluate a subset.
- Use `--rpm-limit N` to rate-limit API calls.
- Use `--limit N` to cap the total number of records.

### `need at least 2 completed runs for trend display`
The `trend` command requires 2+ completed runs. Run more evaluations first.

## Development

```bash
# Setup
git clone https://github.com/onicarps/eval-harness.git
cd eval-harness
pip install -e ".[dev]"

# Lint and format
ruff check src tests
ruff format --check src tests

# Type check
mypy --config-file pyproject.toml src

# Test
pytest tests/ -v --cov=src --cov-report=term-missing

# Run the CLI locally
python -m src.cli run sample.jsonl --dry-run
```
