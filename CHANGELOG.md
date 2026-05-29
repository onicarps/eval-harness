# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-29

### Added
- Typer CLI: `run`, `judges`, `report`, `export`, `cache`
- Pydantic v2 models: `EvalRecord`, `EvalResult`, `EvalRun`, `JudgeCacheEntry`, `RubricTemplate`, `EvalSummary`
- SQLite persistence layer with WAL mode and schema migrations
- JSONL/CSV/stdin ingestion with sampling, since-date filtering, and lenient parsing
- Async LLM-as-judge evaluator with round-robin fallback and response caching
- Rich-based reporter with summary table, ASCII histogram, and JSON/CSV export
- OpenRouter free-model fetcher and on-disk cache
