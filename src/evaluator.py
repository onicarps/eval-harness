"""LLM-as-judge evaluation engine.

Provides:
- Async batched evaluation over an HTTPX client.
- Round-robin fallback through a list of judges.
- SQLite-backed response caching keyed by (model, rubric_version, record).
- Tolerant JSON extraction for markdown-wrapped or prose-padded responses.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from src.db import Database
from src.models import (
    BUILTIN_RUBRIC_V1,
    EvalRecord,
    EvalResult,
    EvalRun,
    JudgeCacheEntry,
    PassFail,
    RubricTemplate,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PASS_THRESHOLD = 0.7


@dataclass
class EvaluatorConfig:
    """Configuration for :class:`LLMEvaluator`."""

    api_key: str
    judges: list[str]
    rubric: RubricTemplate = field(default_factory=lambda: BUILTIN_RUBRIC_V1)
    concurrency: int = 4
    timeout: float = 60.0
    rpm_limit: int | None = None
    pass_threshold: float = PASS_THRESHOLD
    max_fallbacks: int = 3
    no_fallback: bool = False
    use_cache: bool = True


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FALLBACK_OBJ_RE = re.compile(r"(\{.*\})", re.DOTALL)


def extract_judge_json(content: str) -> dict[str, Any]:
    """Extract a JSON object from raw judge text content.

    Handles markdown-fenced blocks and JSON embedded in surrounding prose.

    Args:
        content: Raw assistant content from the judge.

    Returns:
        Parsed JSON dictionary.

    Raises:
        ValueError: When no parseable JSON object is found.
    """
    s = content.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = _FALLBACK_OBJ_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError("could not parse JSON from judge response")


def estimate_tokens(text: str) -> int:
    """Estimate token count for ``text``.

    Falls back to a simple word-based heuristic when tiktoken is unavailable.
    """
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text.split()))


def cache_key_for(
    model_id: str,
    rubric_version: str,
    input_text: str,
    output_text: str,
    reference_text: str | None,
) -> str:
    """Return a stable hash key for caching judge responses."""
    h = hashlib.sha256()
    h.update(model_id.encode())
    h.update(b"|")
    h.update(rubric_version.encode())
    h.update(b"|")
    h.update(input_text.encode())
    h.update(b"|")
    h.update(output_text.encode())
    h.update(b"|")
    h.update((reference_text or "").encode())
    return h.hexdigest()


def combine_scores(faithfulness: float, task_completion: float) -> float:
    """Compute combined score as a balanced average."""
    return 0.5 * float(faithfulness) + 0.5 * float(task_completion)


def pass_fail_from(score: float, threshold: float) -> PassFail:
    """Return PASS when ``score`` >= ``threshold``, otherwise FAIL."""
    return PassFail.PASS if score >= threshold else PassFail.FAIL


def _render_prompt(rubric: RubricTemplate, record: EvalRecord) -> str:
    """Render the rubric prompt with the record's text.

    Uses placeholder substitution (not str.format) so the rubric's JSON example
    braces don't need to be escaped.
    """
    text = rubric.prompt_template
    text = text.replace("{input}", record.input_text)
    text = text.replace("{output}", record.output_text)
    text = text.replace("{reference}", record.reference_text or "(none)")
    return text


class _RateLimiter:
    """Simple async RPM rate limiter."""

    def __init__(self, rpm: int | None) -> None:
        self.rpm = rpm
        self._lock = asyncio.Lock()
        self._timestamps: list[float] = []

    async def wait(self) -> None:
        if not self.rpm:
            return
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < 60.0]
            if len(self._timestamps) >= self.rpm:
                sleep_for = 60.0 - (now - self._timestamps[0])
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
            self._timestamps.append(time.monotonic())


class LLMEvaluator:
    """Async batched LLM-as-judge evaluator backed by OpenRouter."""

    def __init__(self, db: Database, config: EvaluatorConfig) -> None:
        """Initialize the evaluator.

        Args:
            db: Database used for caching and persistence.
            config: Configuration object.
        """
        self.db = db
        self.config = config
        if not config.judges:
            raise ValueError("at least one judge model is required")
        self._rate = _RateLimiter(config.rpm_limit)

    async def evaluate(
        self,
        run: EvalRun,
        records: list[EvalRecord],
        resume: bool = False,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[EvalResult]:
        """Evaluate records, returning a list of :class:`EvalResult`.

        Args:
            run: The active EvalRun (used for run_id and persistence).
            records: Records to evaluate.
            resume: When True, skip records that already have a stored result.
            progress_cb: Optional callback invoked after each record.
        """
        to_eval: list[EvalRecord] = []
        for r in records:
            if resume and self.db.get_result_for_record(r.record_id) is not None:
                continue
            to_eval.append(r)

        sem = asyncio.Semaphore(max(1, self.config.concurrency))
        results: list[EvalResult] = []
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            tasks = [self._evaluate_one(client, sem, run, rec, i) for i, rec in enumerate(to_eval)]
            done = 0
            for coro in asyncio.as_completed(tasks):
                res = await coro
                results.append(res)
                done += 1
                if progress_cb is not None:
                    progress_cb(done, len(to_eval))
        results.sort(key=lambda r: r.evaluated_at or datetime.min.replace(tzinfo=UTC))
        return results

    async def _evaluate_one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        run: EvalRun,
        record: EvalRecord,
        index: int,
    ) -> EvalResult:
        """Evaluate a single record using round-robin fallback."""
        async with sem:
            prompt = _render_prompt(self.config.rubric, record)
            tokens = estimate_tokens(prompt)
            tried: list[str] = []
            last_error: str | None = None
            judges = list(self.config.judges)
            if self.config.no_fallback:
                judges = judges[:1]
            else:
                judges = judges[: max(1, self.config.max_fallbacks + 1)]

            for judge in judges:
                tried.append(judge)
                key = cache_key_for(
                    judge,
                    self.config.rubric.version,
                    record.input_text,
                    record.output_text,
                    record.reference_text,
                )
                cached = self.db.get_cache(key) if self.config.use_cache else None
                if cached is not None:
                    self.db.touch_cache(key)
                    return self._build_result(
                        record=record,
                        run=run,
                        judge=judge,
                        tried=tried,
                        data=cached.response,
                        tokens=tokens,
                    )
                await self._rate.wait()
                try:
                    data = await self._call_judge(client, judge, prompt)
                except Exception as exc:
                    last_error = f"{judge}: {exc}"
                    continue
                if self.config.use_cache:
                    self.db.put_cache(
                        JudgeCacheEntry(
                            cache_key=key,
                            model_id=judge,
                            rubric_version=self.config.rubric.version,
                            response=data,
                        )
                    )
                return self._build_result(
                    record=record,
                    run=run,
                    judge=judge,
                    tried=tried,
                    data=data,
                    tokens=tokens,
                )

            return EvalResult(
                record_id=record.record_id,
                run_id=run.run_id,
                rubric_id=self.config.rubric.rubric_id,
                rubric_version=self.config.rubric.version,
                faithfulness=0.0,
                task_completion=0.0,
                combined_score=0.0,
                pass_fail=PassFail.FAIL,
                judge_model=tried[-1] if tried else "",
                judge_fallbacks=max(0, len(tried) - 1),
                judge_tried=tried,
                tokens_estimated=tokens,
                error=last_error or "no judges available",
            )

    async def _call_judge(
        self, client: httpx.AsyncClient, model_id: str, prompt: str
    ) -> dict[str, Any]:
        """Issue a single OpenRouter chat completion call."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": "You are an impartial evaluator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {resp.status_code}", request=resp.request, response=resp
            )
        body = resp.json()
        content = body["choices"][0]["message"]["content"]
        return extract_judge_json(content)

    def _build_result(
        self,
        record: EvalRecord,
        run: EvalRun,
        judge: str,
        tried: list[str],
        data: dict[str, Any],
        tokens: int,
    ) -> EvalResult:
        """Construct an EvalResult from judge data, clamping invalid scores."""
        faith = float(data.get("faithfulness", 0.0))
        task = float(data.get("task_completion", 0.0))
        faith = max(0.0, min(1.0, faith))
        task = max(0.0, min(1.0, task))
        combined = combine_scores(faith, task)
        return EvalResult(
            record_id=record.record_id,
            run_id=run.run_id,
            rubric_id=self.config.rubric.rubric_id,
            rubric_version=self.config.rubric.version,
            faithfulness=faith,
            task_completion=task,
            combined_score=combined,
            pass_fail=pass_fail_from(combined, self.config.pass_threshold),
            reasoning=str(data.get("reasoning", "")),
            faithfulness_reasoning=str(data.get("faithfulness_reasoning", "")),
            task_completion_reasoning=str(data.get("task_completion_reasoning", "")),
            judge_model=judge,
            judge_fallbacks=max(0, len(tried) - 1),
            judge_tried=tried,
            tokens_estimated=tokens,
        )
