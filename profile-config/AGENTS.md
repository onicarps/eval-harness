# AGENTS.md — Eval Harness Profile

## Session Startup (MANDATORY)
1. Read SOUL.md
2. Read workspace/eval-harness/plan.md (if exists)
3. Check Linear for active tasks
4. Check Notion for recent design decisions

## Build Rules
- TDD: write failing test → run → implement → run → commit
- Type hints everywhere (mypy-compatible)
- Google-style docstrings for all public functions
- ruff check + ruff format before every commit
- Commit after every task (git add -A && git commit -m "type: description")

## File Organization
- Source: workspace/eval-harness/src/
- Tests: workspace/eval-harness/tests/
- Docs: workspace/eval-harness/docs/
- Spike scripts: workspace/eval-harness/spikes/

## Testing
- pytest tests/ -v --cov=src --cov-report=term-missing
- VCR cassettes for mock LLM responses (tests/cassettes/)
- No real API calls in CI
