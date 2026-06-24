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
import logging
import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

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
PASS_THRESHOLD = 0.7  # Default for EvaluatorConfig; override via --pass-threshold

# HTTP status codes that indicate a permanent (non-retryable) error.
# These should NOT trigger judge fallback — the request itself is invalid.
_PERMANENT_STATUS_CODES: frozenset[int] = frozenset({400, 401, 402, 403, 404})

# Setup logging
logger = logging.getLogger(__name__)


def _is_permanent_http_error(status_code: int) -> bool:
    """Return True when ``status_code`` indicates a non-retryable error.

    Authentication (401), authorization (403), payment (402), and bad-request
    (400/404) errors will not succeed with a different judge, so fallback is
    wasteful.  Transient errors (429, 5xx) should trigger fallback.
    """
    return status_code in _PERMANENT_STATUS_CODES


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
    degrade: bool = False


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_FALLBACK_OBJ_RE = re.compile(r"(\{.*?\})", re.DOTALL)


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
        return cast(dict[str, Any], json.loads(s))
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK_RE.search(s)
    if m:
        try:
            return cast(dict[str, Any], json.loads(m.group(1)))
        except json.JSONDecodeError:
            pass
    m = _FALLBACK_OBJ_RE.search(s)
    if m:
        try:
            return cast(dict[str, Any], json.loads(m.group(1)))
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
        return max(1, len(text) // 4)


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


def _build_feedback_prompt(combined_score: float, faithfulness: float, task_completion: float, input_text: str, output_text: str, reference_text: str) -> str:
    """Build the feedback prompt for a low-scoring record."""
    return (
        "You are an evaluation assistant. A model produced the following output for "
        "the given input. The output scored " + f"{combined_score:.2f} "
        + "(faithfulness: " + f"{faithfulness:.2f}" + ", task_completion: " + f"{task_completion:.2f}" + "). "
        + "Provide 2-3 specific, actionable suggestions to improve the score. "
        + "Return STRICT JSON only with the key 'suggestions' as a list of strings. "
        + 'Example: {"suggestions": ["suggestion1", "suggestion2"]}. '
        + "\nINPUT:\n" + input_text + "\n\nOUTPUT:\n" + output_text + "\n\nREFERENCE:\n" + (reference_text or "(none)") + "\n"
    )


def combine_scores(faithfulness: float, task_completion: float) -> float:
    """Compute combined score as a balanced 50/50 average.

    The equal weighting reflects that both faithfulness (avoiding
    hallucination) and task completion (satisfying the request) are
    equally important for a quality response.  Phase 2 rubrics may
    expose a configurable weighting.
    """
    return 0.5 * float(faithfulness) + 0.5 * float(task_completion)


def pass_fail_from(score: float, threshold: float) -> PassFail:
    """Return PASS when ``score`` >= ``threshold``, otherwise FAIL."""
    return PassFail.PASS if score >= threshold else PassFail.FAIL


def local_heuristic_score(record: EvalRecord) -> dict[str, Any]:
    """Compute a local heuristic score when the judge API is unavailable.

    Uses keyword overlap and response length as rough quality signals.
    Returns a dict matching the judge output schema.
    """
    input_words = set(record.input_text.lower().split())
    output_words = set(record.output_text.lower().split())
    # Faithfulness: overlap between input and output (Jaccard-like)
    if input_words:
        overlap = len(input_words & output_words) / len(input_words)
        faithfulness = min(1.0, overlap * 2.0)  # scale up a bit
    else:
        overlap = 0.0
        faithfulness = 0.5
    # Task completion: length-based heuristic (longer = more complete, capped)
    output_len = len(record.output_text.split())
    task_completion = min(1.0, output_len / 50.0)  # 50 words = "complete"
    return {
        "faithfulness": round(faithfulness, 2),
        "task_completion": round(task_completion, 2),
        "reasoning": "Local heuristic fallback (API unavailable)",
        "faithfulness_reasoning": f"Keyword overlap: {overlap:.1%} of input words found in output",
        "task_completion_reasoning": f"Output length: {output_len} words (heuristic threshold: 50)",
    }


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
    """Simple async RPM rate limiter using a bounded deque."""

    def __init__(self, rpm: int | None) -> None:
        self.rpm = rpm
        self._lock = asyncio.Lock()
        self._timestamps: deque[float] = deque(maxlen=rpm or 60)

    async def wait(self) -> None:
        if not self.rpm:
            return
        async with self._lock:
            now = time.monotonic()
            # Drop timestamps older than 60s
            while self._timestamps and now - self._timestamps[0] >= 60.0:
                self._timestamps.popleft()
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
        logger.info("Starting evaluation of %d records (resume=%s)", len(records), resume)
        to_eval: list[EvalRecord] = []
        for r in records:
            if resume and self.db.get_result_for_record(r.record_id) is not None:
                logger.debug("Skipping record %s (already evaluated)", r.record_id)
                continue
            to_eval.append(r)
        logger.info("Evaluating %d records (skipped %d cached)", len(to_eval), len(records) - len(to_eval))

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
                    logger.debug("Progress callback: %d/%d", done, len(to_eval))
        results.sort(key=lambda r: r.evaluated_at or datetime.min.replace(tzinfo=UTC))
        logger.info("Evaluation completed for %d records", len(results))
        return results

    async def generate_all_feedback(
        self,
        run: EvalRun,
        records: list[EvalRecord],
        results: list[EvalResult],
    ) -> None:
        """Generate improvement suggestions for all low-scoring records.

        Evaluates feedback only for records below pass_threshold.
        Updates results in-place and persists to DB.
        """
        logger.info("Generating feedback for low-scoring records")
        rec_map = {r.record_id: r for r in records}
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            for result in results:
                if result.pass_fail == PassFail.PASS:
                    continue
                if result.error:
                    continue
                record = rec_map.get(result.record_id)
                if record is None:
                    continue
                feedback = await self.generate_feedback(client, record, result)
                if feedback:
                    result.feedback = feedback
        logger.info("Feedback generation completed")

    async def _evaluate_one(
        self,
        client: httpx.AsyncClient,
        sem: asyncio.Semaphore,
        run: EvalRun,
        record: EvalRecord,
        index: int,
    ) -> EvalResult:
        """Evaluate a single record using round-robin fallback."""
        logger.debug("Starting evaluation of record %s (index %d)", record.record_id, index)
        async with sem:
            prompt = _render_prompt(self.config.rubric, record)
            tokens = estimate_tokens(prompt)
            logger.debug("Prompt tokens estimated: %d", tokens)
            tried: list[str] = []
            last_error: str | None = None
            judges = list(self.config.judges)
            if self.config.no_fallback:
                judges = judges[:1]
            else:
                judges = judges[: max(1, self.config.max_fallbacks + 1)]
            logger.debug("Will try judges in order: %s", judges)

            for judge in judges:
                tried.append(judge)
                logger.debug("Trying judge %s (%d/%d)", judge, len(tried), len(judges))
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
                    logger.debug("Using cached result for judge %s", judge)
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
                    logger.debug("Calling judge %s", judge)
                    data = await self._call_judge(client, judge, prompt)
                    logger.debug("Judge %s returned result", judge)
                except httpx.HTTPStatusError as exc:
                    last_error = f"{judge}: HTTP {exc.response.status_code}"
                    logger.warning("Judge %s failed: HTTP %d", judge, exc.response.status_code)
                    if _is_permanent_http_error(exc.response.status_code):
                        logger.info("Permanent error (%d) — skipping fallback judges", exc.response.status_code)
                        break
                    continue
                except Exception as exc:
                    last_error = f"{judge}: {exc}"
                    logger.warning("Judge %s failed: %s", judge, exc)
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
                    logger.debug("Cached result for judge %s", judge)
                return self._build_result(
                    record=record,
                    run=run,
                    judge=judge,
                    tried=tried,
                    data=data,
                    tokens=tokens,
                )

            # If degrade mode is enabled, use local heuristic fallback
            if self.config.degrade:
                logger.warning("All judges failed for record %s, using local heuristic", record.record_id)
                heuristic = local_heuristic_score(record)
                data = heuristic
                return self._build_result(
                    record=record,
                    run=run,
                    judge="local-heuristic",
                    tried=tried + ["local-heuristic"],
                    data=data,
                    tokens=tokens,
                )
            logger.error("All judges failed for record %s", record.record_id)
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
        logger.debug("Calling OpenRouter API for model %s", model_id)
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
        resp = await client.post(
            OPENROUTER_URL,
            headers=headers,
            json=payload,
            timeout=httpx.Timeout(self.config.timeout, connect=10.0),
        )
        logger.debug("OpenRouter API response status: %d", resp.status_code)
        if resp.status_code >= 400:
            logger.error("OpenRouter API HTTP error: %d", resp.status_code)
            raise httpx.HTTPStatusError(
                f"HTTP {resp.status_code}", request=resp.request, response=resp
            )
        body = resp.json()
        # Check for OpenRouter application-level error format
        if "error" in body:
            error_msg = body["error"].get("message", "Unknown error from OpenRouter")
            logger.error("OpenRouter API application error: %s", error_msg)
            raise RuntimeError(f"OpenRouter API error: {error_msg}")
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError("OpenRouter API returned no choices")
        content = choices[0]["message"]["content"]
        logger.debug("OpenRouter API returned content length: %d", len(content))
        return extract_judge_json(content)

    async def generate_feedback(
        self,
        client: httpx.AsyncClient,
        record: EvalRecord,
        result: EvalResult,
    ) -> str | None:
        """Generate improvement suggestions for a low-scoring record.

        Calls the primary judge with a feedback prompt and extracts
        suggestions from the JSON response.

        Returns:
            Feedback string (JSON with suggestions) or None on error.
        """
        logger.debug("Generating feedback for record %s", record.record_id)
        prompt = _build_feedback_prompt(
            combined_score=result.combined_score,
            faithfulness=result.faithfulness,
            task_completion=result.task_completion,
            input_text=record.input_text,
            output_text=record.output_text,
            reference_text=record.reference_text or "(none)",
        )
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.judges[0],
            "messages": [
                {"role": "system", "content": "You are a helpful improvement advisor."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
        }
        try:
            resp = await client.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=httpx.Timeout(self.config.timeout, connect=10.0),
            )
            if resp.status_code >= 400:
                logger.warning("Feedback API error %d for record %s", resp.status_code, record.record_id)
                return None
            body = resp.json()
            content = body["choices"][0]["message"]["content"]
            data = extract_judge_json(content)
            suggestions = data.get("suggestions", [])
            if suggestions:
                feedback = json.dumps({"suggestions": suggestions})
                logger.debug("Generated %d suggestions for record %s", len(suggestions), record.record_id)
                return feedback
            logger.debug("No suggestions in feedback response for record %s", record.record_id)
            return None
        except Exception as exc:
            logger.warning("Feedback generation failed for record %s: %s", record.record_id, exc)
            return None

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
        logger.debug("Building result for record %s from judge %s", record.record_id, judge)
        faith = float(data.get("faithfulness", 0.0))
        task = float(data.get("task_completion", 0.0))
        faith = max(0.0, min(1.0, faith))
        task = max(0.0, min(1.0, task))
        logger.debug("Scores: faithfulness=%.3f, task_completion=%.3f", faith, task)
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
