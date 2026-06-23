# Droid Mission: Phase 2A — Batch 1 (ONI-51, ONI-47, ONI-45)

## Mission Overview
Implement 3 Phase 2A features for eval-harness v0.2.0: DB migration v2 (rubric_templates), rubric CLI command, and trend tracking command. Follow TDD strictly — write failing tests before implementation.

## Pre-requisites
- Read AGENTS.md, PLAN_PHASE2.md, and all existing source files before starting
- Current version: v0.1.1 at commit def1df5
- All existing tests pass (101 tests, ruff clean, mypy clean)
- Working directory: ~/.hermes/profiles/eval-harness/workspace/eval-harness/

## Execution Order

### Task 1: ONI-51 — DB Migration v2 (rubric_templates + seed)

**What:** Add rubric_templates table and seed 5 built-in templates.

**Schema (from PLAN_PHASE2.md):**
```sql
CREATE TABLE rubric_templates (
    template_id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    yaml_content TEXT NOT NULL,
    is_builtin INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);
ALTER TABLE eval_runs ADD COLUMN rubric_template_id TEXT;
```

**Steps:**
1. In `src/db.py`, add migration `2` to `_MIGRATIONS` dict with the CREATE TABLE and ALTER TABLE statements
2. Add rollback `2` to `_ROLLBACKS` dict: `ALTER TABLE eval_runs DROP COLUMN rubric_template_id; DROP TABLE IF EXISTS rubric_templates;`
3. Bump `CURRENT_SCHEMA_VERSION` to 2
4. Write a seed function `_seed_rubric_templates()` that inserts 5 built-in templates if the table is empty. Call it from `_migrate()` after applying migration 2. The 5 templates:
   - `faithfulness-v1` — existing dual-dimension rubric (faithfulness + task completion, 50/50 weighting)
   - `safety-v1` — safety-focused rubric (harm avoidance + helpfulness, 60/40 weighting)
   - `accuracy-v1` — accuracy-focused rubric (factual correctness + completeness, 70/30 weighting)
   - `conciseness-v1` — conciseness rubric (brevity + clarity, 50/50 weighting)
   - `custom-v1` — empty template for user customization
5. Each template's `yaml_content` should be a YAML string with: `dimensions` (list of {name, weight, description}), `scoring` (scale: 0-1), `output_format` (JSON schema)
6. Add `rubric_template_id` field to `EvalRun` model in `models.py` (Optional[str] = None)
7. Update `db.py` `insert_run` and `get_run` to handle the new column
8. Write tests in `test_db.py`: test migration v2 applies cleanly, test seed inserts 5 templates, test rollback works
9. Run `ruff check src/`, `mypy --config-file pyproject.toml src/`, `pytest tests/ -v` — all must pass
10. Commit: `feat: add rubric_templates table with 5 built-in seeds (ONI-51)`

### Task 2: ONI-47 — `eval-harness rubric` Command

**What:** Add `eval-harness rubric` subcommand with `--list`, `--show <id>`, `--create` operations.

**Steps:**
1. Create `src/rubric.py` module with:
   - `RubricManager` class: list_templates(), get_template(template_id), create_template(name, yaml_content), delete_template(template_id)
   - Uses `Database` for persistence
   - Validates YAML content has required fields (dimensions, scoring, output_format)
2. Add `rubric` subcommand to `cli.py`:
   - `eval-harness rubric --list` — Rich table of all templates (ID, name, is_builtin, created_at)
   - `eval-harness rubric --show <template_id>` — shows full YAML content
   - `eval-harness rubric --create --name <name> --file <path>` — creates from YAML file
   - `eval-harness rubric --delete <template_id>` — deletes (refuses if is_builtin=1)
3. Update `AGENTS.md` CLI Commands section to include `eval-harness rubric`
4. Write tests in `test_rubric.py`: test list, show, create, delete, validation errors, builtin protection
5. Run all checks, commit: `feat: add eval-harness rubric command (ONI-47)`

### Task 3: ONI-45 — `eval-harness trend` Command

**What:** Add `eval-harness trend` subcommand showing score timeline and regression detection.

**Steps:**
1. Create `src/trend.py` module with:
   - `compute_trends(db, rubric_template_id=None, judge_model=None, since=None)` function
   - Queries `eval_runs` filtered by optional params, ordered by created_at
   - Returns list of {run_id, created_at, mean_score, pass_rate, record_count}
   - Regression detection: flag runs where mean_score dropped >10% from previous
   - Sparse data guard: require ≥3 runs for display, ≥5 for regression detection
2. Add `trend` subcommand to `cli.py`:
   - `eval-harness trend` — shows ASCII timeline of mean_score over runs
   - `eval-harness trend --rubric <template_id>` — filter by rubric
   - `eval-harness trend --judge <model>` — filter by judge
   - `eval-harness trend --since <date>` — filter by date
   - `eval-harness trend --json` — JSON output
   - Rich output: table with run_id, date, records, mean_score, pass_rate, regression_flag
3. Write tests in `test_trend.py`: test with sparse data (<3 runs returns message), test with sufficient data, test regression detection, test filters, test JSON output
4. Run all checks, commit: `feat: add eval-harness trend command with regression detection (ONI-45)`

## Quality Gates (after ALL 3 tasks)
- `ruff check src/` — zero errors
- `ruff format --check src/` — zero errors
- `mypy --config-file pyproject.toml src/` — zero errors
- `pytest tests/ -v --cov=src --cov-report=term-missing` — all tests pass, 90%+ coverage
- `CHANGELOG.md` updated with [0.2.0] section
- `README.md` updated with new commands

## Important Rules
- ONE feature per commit (3 commits total)
- TDD: write failing test BEFORE implementation for each feature
- Follow existing patterns: judges.py for CLI subcommand pattern, reporter.py for Rich output, db.py for DB operations
- Google-style docstrings on all public functions
- Type hints everywhere
- Do NOT modify existing tests — only add new ones
- If a test fails after your changes, fix the code (not the test) unless the test is wrong
