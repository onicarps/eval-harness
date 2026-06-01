"""Tests for src/evaluator.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest_httpx import HTTPXMock

from src.db import Database
from src.evaluator import (
    EvaluatorConfig,
    LLMEvaluator,
    _RateLimiter,
    cache_key_for,
    estimate_tokens,
    extract_judge_json,
)
from src.models import BUILTIN_RUBRIC_V1, EvalRecord, EvalRun


def test_extract_judge_json_plain() -> None:
    s = '{"faithfulness": 0.5, "task_completion": 0.5}'
    out = extract_judge_json(s)
    assert out["faithfulness"] == 0.5


def test_extract_judge_json_markdown_wrapped() -> None:
    s = '```json\n{"faithfulness": 0.7, "task_completion": 0.9}\n```'
    out = extract_judge_json(s)
    assert out["faithfulness"] == 0.7


def test_extract_judge_json_inside_prose() -> None:
    s = 'Reasoning here.\n{"faithfulness": 0.4, "task_completion": 0.6}\nEnd.'
    out = extract_judge_json(s)
    assert out["task_completion"] == 0.6


def test_extract_judge_json_invalid() -> None:
    with pytest.raises(ValueError):
        extract_judge_json("no json here")


def test_estimate_tokens_returns_positive() -> None:
    assert estimate_tokens("hello world") > 0


def test_cache_key_stable() -> None:
    a = cache_key_for("m", "1.0", "i", "o", "r")
    b = cache_key_for("m", "1.0", "i", "o", "r")
    assert a == b
    c = cache_key_for("m", "1.0", "i", "o", "different")
    assert a != c


@pytest.fixture()
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "eval.db")


@pytest.mark.asyncio
async def test_evaluate_single_success(
    db: Database,
    httpx_mock: HTTPXMock,
    openrouter_payload: dict,
) -> None:
    run = EvalRun(config={}, judge_model="judge/free")
    db.insert_run(run)
    rec = EvalRecord(input_text="hi", output_text="hello", run_id=run.run_id)
    db.insert_record(rec)
    httpx_mock.add_response(
        url="https://openrouter.ai/api/v1/chat/completions",
        method="POST",
        json=openrouter_payload,
    )
    ev = LLMEvaluator(
        db=db,
        config=EvaluatorConfig(
            api_key="test",
            judges=["judge/free"],
            rubric=BUILTIN_RUBRIC_V1,
            concurrency=1,
        ),
    )
    results = await ev.evaluate(run, [rec])
    assert len(results) == 1
    assert results[0].judge_model == "judge/free"
    assert results[0].faithfulness == 0.9
    assert results[0].combined_score == pytest.approx(0.5 * 0.9 + 0.5 * 0.85)


@pytest.mark.asyncio
async def test_evaluate_uses_cache(
    db: Database, httpx_mock: HTTPXMock, openrouter_payload: dict
) -> None:
    run = EvalRun(config={}, judge_model="judge/free")
    db.insert_run(run)
    rec = EvalRecord(input_text="hi", output_text="hello", run_id=run.run_id)
    db.insert_record(rec)
    httpx_mock.add_response(
        url="https://openrouter.ai/api/v1/chat/completions",
        method="POST",
        json=openrouter_payload,
    )
    ev = LLMEvaluator(
        db=db,
        config=EvaluatorConfig(
            api_key="test",
            judges=["judge/free"],
            rubric=BUILTIN_RUBRIC_V1,
            concurrency=1,
        ),
    )
    first = await ev.evaluate(run, [rec])
    assert first[0].judge_fallbacks == 0
    second = await ev.evaluate(run, [rec])
    assert second[0].faithfulness == 0.9


@pytest.mark.asyncio
async def test_evaluate_fallback(
    db: Database, httpx_mock: HTTPXMock, openrouter_payload: dict
) -> None:
    run = EvalRun(config={}, judge_model="judge/primary")
    db.insert_run(run)
    rec = EvalRecord(input_text="hi", output_text="hello", run_id=run.run_id)
    db.insert_record(rec)
    httpx_mock.add_response(
        url="https://openrouter.ai/api/v1/chat/completions",
        method="POST",
        status_code=500,
    )
    httpx_mock.add_response(
        url="https://openrouter.ai/api/v1/chat/completions",
        method="POST",
        json=openrouter_payload,
    )
    ev = LLMEvaluator(
        db=db,
        config=EvaluatorConfig(
            api_key="test",
            judges=["judge/primary", "judge/secondary"],
            rubric=BUILTIN_RUBRIC_V1,
            concurrency=1,
            max_fallbacks=2,
        ),
    )
    results = await ev.evaluate(run, [rec])
    assert results[0].judge_fallbacks == 1
    assert "judge/primary" in results[0].judge_tried
    assert "judge/secondary" in results[0].judge_tried


@pytest.mark.asyncio
async def test_evaluate_no_fallback_records_error(db: Database, httpx_mock: HTTPXMock) -> None:
    run = EvalRun(config={}, judge_model="judge/primary")
    db.insert_run(run)
    rec = EvalRecord(input_text="hi", output_text="hello", run_id=run.run_id)
    db.insert_record(rec)
    httpx_mock.add_response(
        url="https://openrouter.ai/api/v1/chat/completions",
        method="POST",
        status_code=500,
    )
    ev = LLMEvaluator(
        db=db,
        config=EvaluatorConfig(
            api_key="test",
            judges=["judge/primary"],
            rubric=BUILTIN_RUBRIC_V1,
            concurrency=1,
            max_fallbacks=0,
            no_fallback=True,
        ),
    )
    results = await ev.evaluate(run, [rec])
    assert results[0].error is not None
    assert results[0].faithfulness == 0.0


@pytest.mark.asyncio
async def test_evaluate_resume_skips_existing(
    db: Database, httpx_mock: HTTPXMock, openrouter_payload: dict
) -> None:
    run = EvalRun(config={}, judge_model="judge/free")
    db.insert_run(run)
    rec1 = EvalRecord(input_text="i1", output_text="o1", run_id=run.run_id)
    rec2 = EvalRecord(input_text="i2", output_text="o2", run_id=run.run_id)
    db.insert_record(rec1)
    db.insert_record(rec2)
    httpx_mock.add_response(
        url="https://openrouter.ai/api/v1/chat/completions",
        method="POST",
        json=openrouter_payload,
    )
    ev = LLMEvaluator(
        db=db,
        config=EvaluatorConfig(
            api_key="test",
            judges=["judge/free"],
            rubric=BUILTIN_RUBRIC_V1,
            concurrency=1,
        ),
    )
    results = await ev.evaluate(run, [rec1])
    assert len(results) == 1
    for r in results:
        db.insert_result(r)

    httpx_mock.add_response(
        url="https://openrouter.ai/api/v1/chat/completions",
        method="POST",
        json=openrouter_payload,
    )
    results2 = await ev.evaluate(run, [rec1, rec2], resume=True)
    assert len(results2) == 1
    assert results2[0].record_id == rec2.record_id


def test_pass_fail_threshold() -> None:
    from src.evaluator import combine_scores, pass_fail_from

    score = combine_scores(0.5, 0.9)
    assert score == 0.7
    assert pass_fail_from(0.7, 0.7).value == "pass"
    assert pass_fail_from(0.69, 0.7).value == "fail"


def test_extract_judge_json_clamps_via_build_result(db: Database, openrouter_payload: dict) -> None:
    run = EvalRun(config={}, judge_model="m")
    rec = EvalRecord(input_text="i", output_text="o", run_id=run.run_id)
    ev = LLMEvaluator(
        db=db,
        config=EvaluatorConfig(
            api_key="t",
            judges=["m"],
            rubric=BUILTIN_RUBRIC_V1,
        ),
    )
    result = ev._build_result(
        record=rec,
        run=run,
        judge="m",
        tried=["m"],
        data={"faithfulness": 2.0, "task_completion": -0.5, "reasoning": "r"},
        tokens=1,
    )
    assert result.faithfulness == 1.0
    assert result.task_completion == 0.0


def test_extract_judge_json_handles_integer_scores() -> None:
    s = '{"faithfulness": 1, "task_completion": 0}'
    out = extract_judge_json(s)
    assert float(out["faithfulness"]) == 1.0


def test_evaluator_requires_at_least_one_judge(db: Database) -> None:
    with pytest.raises(ValueError):
        LLMEvaluator(
            db=db,
            config=EvaluatorConfig(api_key="t", judges=[], rubric=BUILTIN_RUBRIC_V1),
        )


@pytest.mark.asyncio
async def test_rate_limiter_unlimited_is_noop() -> None:
    limiter = _RateLimiter(None)
    await limiter.wait()


@pytest.mark.asyncio
async def test_rate_limiter_within_capacity() -> None:
    limiter = _RateLimiter(60)
    for _ in range(3):
        await limiter.wait()
    assert len(limiter._timestamps) == 3
