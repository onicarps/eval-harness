"""Trend tracking for eval-harness.

Provides score timeline visualization and regression detection
across evaluation runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.db import Database

logger = logging.getLogger(__name__)

# Minimum runs required for display / regression detection
MIN_RUNS_DISPLAY = 3
MIN_RUNS_REGRESSION = 5
# Regression threshold: flag if score drops more than this fraction
REGRESSION_THRESHOLD = 0.10


@dataclass
class TrendPoint:
    """A single point in the trend timeline."""
    run_id: str
    created_at: datetime
    mean_score: float | None
    pass_rate: float | None
    record_count: int
    is_regression: bool = False


@dataclass
class TrendResult:
    """Complete trend analysis result."""
    points: list[TrendPoint] = field(default_factory=list)
    total_runs: int = 0
    has_regression: bool = False
    mean_score_overall: float | None = None
    latest_score: float | None = None
    earliest_score: float | None = None


def compute_trends(
    db: Database,
    rubric_template_id: str | None = None,
    judge_model: str | None = None,
    since: datetime | None = None,
) -> TrendResult:
    """Compute trend data from evaluation runs.

    Args:
        db: Database instance.
        rubric_template_id: Optional filter by rubric template.
        judge_model: Optional filter by judge model.
        since: Optional filter by date (only runs after this date).

    Returns:
        TrendResult with timeline points and regression flags.
    """
    result = TrendResult()

    # Build query dynamically based on filters
    query = """
        SELECT run_id, created_at, mean_score, pass_rate, record_count
        FROM eval_runs
        WHERE status = 'completed' AND mean_score IS NOT NULL
    """
    params: list[Any] = []

    if rubric_template_id:
        query += " AND rubric_template_id = ?"
        params.append(rubric_template_id)
    if judge_model:
        query += " AND judge_model = ?"
        params.append(judge_model)
    if since:
        query += " AND created_at >= ?"
        params.append(since.isoformat())

    query += " ORDER BY created_at ASC;"

    cur = db.connection.execute(query, params)
    rows = cur.fetchall()
    result.total_runs = len(rows)

    if len(rows) < MIN_RUNS_DISPLAY:
        logger.debug("Only %d runs found, need %d for display", len(rows), MIN_RUNS_DISPLAY)
        return result

    # Build trend points
    prev_score: float | None = None
    for row in rows:
        created_at = datetime.fromisoformat(row[1]) if row[1] else datetime.min
        mean_score = row[2]
        pass_rate = row[3]
        record_count = row[4] or 0

        is_regression = False
        if (
            mean_score is not None
            and prev_score is not None
            and len(result.points) >= MIN_RUNS_REGRESSION - 1
        ):
            drop = prev_score - mean_score
            if drop > REGRESSION_THRESHOLD:
                is_regression = True
                result.has_regression = True
                logger.info(
                    "Regression detected: run %s score %.3f -> %.3f (drop %.3f)",
                    row[0][:8], prev_score, mean_score, drop,
                )

        result.points.append(
            TrendPoint(
                run_id=row[0],
                created_at=created_at,
                mean_score=mean_score,
                pass_rate=pass_rate,
                record_count=record_count,
                is_regression=is_regression,
            )
        )
        if mean_score is not None:
            prev_score = mean_score

    # Compute summary stats
    scores = [p.mean_score for p in result.points if p.mean_score is not None]
    if scores:
        result.mean_score_overall = sum(scores) / len(scores)
        result.latest_score = scores[-1]
        result.earliest_score = scores[0]

    return result
