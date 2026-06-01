# eval-harness

A Python CLI that evaluates LLM outputs from production logs against a
dual-dimension rubric (faithfulness + task completion).

## Install

```bash
pip install -e ".[dev]"
```

## Quickstart

```bash
export OPENRIXER_API_KEY=sk-or-...
eval-harness run path/to/logs.jsonl --judge meta-llama/llama-3.1-8b-instruct:free
```

Input JSONL schema:

```json
{"input": "user prompt", "output": "model response", "reference": "optional ground truth"}
```

## Commands

- `eval-harness run <file>` — ingest, evaluate, and report
- `eval-harness judges` — list free judge models (cached in `~/.eval-harness/judges.json`)
- `eval-harness report --run-id UUID` — show a stored run
- `eval-harness export --run-id UUID --format json|csv --output-file PATH`
- `eval-harness cache [--stats] [--clear]`

Exit codes: `0` all pass, `1` any failures, `2` evaluator error.

## CI/CD example

```yaml
- run: pip install llm-eval-harness
- run: OPENRIXER_API_KEY=${{ secrets.OPENRIXER_API_KEY }} eval-harness run eval/cases.jsonl --pass-threshold 0.7
```

## Development

```bash
pip install -e ".[dev]"
ruff check src tests && ruff format --check src tests
pytest tests/ -v --cov=src
```
