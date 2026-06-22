Generate the complete eval-harness Python CLI project. Follow the AGENTS.md spec in the current directory exactly.

Create ALL files listed in the project structure. Use TDD: write each test file before its corresponding source file.

Files to create:
1. pyproject.toml (hatchling build, all deps, ruff config, test config)
2. .github/workflows/ci.yml (ruff + pytest)
3. README.md (install, quickstart, CI/CD example)
4. .gitignore
5. .env.example
6. LICENSE (MIT)
7. CHANGELOG.md
8. src/__init__.py
9. src/models.py (Pydantic v2: EvalRecord, EvalResult, EvalRun, JudgeCacheEntry, RubricTemplate, BUILTIN_RUBRIC_V1, PassFail/RunStatus enums, EvalSummary)
10. src/db.py (SQLite WAL mode, schema versioning, full CRUD for all 5 tables, export JSON/CSV)
11. src/ingest.py (JSONL + CSV + stdin parser: --sample N, --since DATE, lenient parsing, no auto-detect)
12. src/evaluator.py (async batch eval, round-robin fallback, sqlite response cache, token tracking, markdown-wrapped JSON handling)
13. src/reporter.py (Rich terminal tables, ASCII histogram, JSON/CSV export, judge usage stats)
14. src/cli.py (Typer app: run/judges/report/export/cache commands, all flags from AGENTS.md, exit codes 0/1/2, --dry-run, --resume, progress bar for batches > 10)
15. src/judges.py (OpenRouter model fetch/cache, rank by context length, --refresh, --json)
16. tests/conftest.py (temp DB per test, sample records, mock judge responses)
17. tests/test_models.py, test_db.py, test_ingest.py, test_evaluator.py, test_reporter.py, test_cli.py, test_judges.py

Requirements:
- Python 3.11+, hatchling build backend
- Dependencies: typer>=0.12, pydantic>=2.0, rich>=13.0, httpx>=0.27, tiktoken>=0.7
- Dev deps: pytest>=8, pytest-httpx>=0.30, pytest-cov>=5, ruff>=0.5
- openrouter_API_KEY env var (never hardcoded)
- TDD for every module: test file before source file, each public function tested
- Type hints everywhere (mypy-compatible)
- Google-style docstrings for all public functions
- The project must work: pip install -e . && eval-harness --help
- Database: sqlite3 stdlib (NOT aiosqlite), WAL mode, foreign keys, schema migration system
- Judge API: OpenRouter (openrouter.ai/api/v1/chat/completions), round-robin fallback through cached free models
