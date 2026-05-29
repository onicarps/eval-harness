#!/bin/bash
set -e

cd /home/oni/.hermes/profiles/eval-harness/workspace/eval-harness

# Load env
eval $(grep -v '^#' /home/oni/.hermes/profiles/eval-harness/.env | grep -v '^$' | sed 's/^/export /')

echo "FACTORY_API_KEY set: ${FACTORY_API_KEY:0:8}..."

# Write prompt to temp file
cat > /tmp/droid_prompt.txt << 'PROMPT'
Read the AGENTS.md and GENERATION_PROMPT.md files in the current directory. Then generate the complete eval-harness Python CLI project as specified. Create ALL files listed in the project structure. Use TDD: write each test file before its corresponding source file. Start with pyproject.toml, then src/__init__.py, src/models.py, tests/test_models.py, src/db.py, tests/test_db.py, src/ingest.py, tests/test_ingest.py, src/evaluator.py, tests/test_evaluator.py, src/reporter.py, tests/test_reporter.py, src/cli.py, tests/test_cli.py, src/judges.py, tests/test_judges.py, .github/workflows/ci.yml, README.md, .gitignore, .env.example, LICENSE, CHANGELOG.md. Make sure pip install -e . works and eval-harness --help shows all commands.
PROMPT

echo "Starting droid..."
droid exec --auto high -f /tmp/droid_prompt.txt
