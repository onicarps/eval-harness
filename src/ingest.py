"""Input ingestion for eval-harness.

Supports JSONL, CSV and stdin formats with lenient parsing, sampling,
date filtering, limiting, and configurable column mappings.
"""

from __future__ import annotations

import csv
import io
import json
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.models import EvalRecord


@dataclass
class IngestOptions:
    """Options governing ingestion behavior.

    Attributes:
        input_col: Column/key holding the input/prompt text.
        output_col: Column/key holding the model output.
        reference_col: Column/key holding optional ground truth.
        sample: Optional random sample size.
        seed: Random seed when sampling is requested.
        since: Optional ISO-8601 date string; rows older than this are dropped.
        limit: Hard cap on the number of records returned.
        timestamp_col: Column/key containing the row timestamp.
    """

    input_col: str = "input"
    output_col: str = "output"
    reference_col: str = "reference"
    timestamp_col: str = "ts"
    sample: int | None = None
    seed: int | None = None
    since: str | None = None
    limit: int | None = None


def _parse_since(value: str | None) -> datetime | None:
    """Parse a since-date string into a datetime; return None if missing."""
    if not value:
        return None
    return datetime.fromisoformat(value)


def _row_after_since(row_ts: str | None, since: datetime | None) -> bool:
    """Return True when ``row_ts`` is >= ``since`` (always True if either is None)."""
    if since is None or not row_ts:
        return since is None
    try:
        dt = datetime.fromisoformat(row_ts)
    except ValueError:
        return False
    return dt >= since


def _make_record(
    data: dict[str, object],
    opts: IngestOptions,
    source_file: str | None,
) -> EvalRecord | None:
    """Build an EvalRecord from a dict; return None when required keys are missing."""
    inp = data.get(opts.input_col)
    out = data.get(opts.output_col)
    if not isinstance(inp, str) or not isinstance(out, str) or not inp or not out:
        return None
    ref = data.get(opts.reference_col)
    ref_text = ref if isinstance(ref, str) and ref else None
    meta = {
        k: v
        for k, v in data.items()
        if k not in {opts.input_col, opts.output_col, opts.reference_col}
    }
    return EvalRecord(
        input_text=inp,
        output_text=out,
        reference_text=ref_text,
        source_file=source_file,
        metadata=meta,
    )


def _apply_post_filters(records: Iterable[EvalRecord], opts: IngestOptions) -> Iterator[EvalRecord]:
    """Apply sampling and limiting filters to a record iterable."""
    if opts.sample is not None:
        rng = random.Random(opts.seed)
        materialized = list(records)
        if opts.sample < len(materialized):
            materialized = rng.sample(materialized, opts.sample)
        records = materialized
    count = 0
    for r in records:
        if opts.limit is not None and count >= opts.limit:
            return
        yield r
        count += 1


def ingest_jsonl(path: str | Path, options: IngestOptions | None = None) -> Iterator[EvalRecord]:
    """Yield EvalRecord rows from a JSONL file.

    Bad lines are silently skipped.

    Args:
        path: File path to ingest from.
        options: Ingestion options.
    """
    opts = options or IngestOptions()
    since = _parse_since(opts.since)
    path = Path(path)

    def _gen() -> Iterator[EvalRecord]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                ts = obj.get(opts.timestamp_col)
                ts_str = ts if isinstance(ts, str) else None
                if not _row_after_since(ts_str, since):
                    continue
                rec = _make_record(obj, opts, source_file=str(path))
                if rec is not None:
                    yield rec

    yield from _apply_post_filters(_gen(), opts)


def ingest_csv(path: str | Path, options: IngestOptions | None = None) -> Iterator[EvalRecord]:
    """Yield EvalRecord rows from a CSV file using the supplied column mapping."""
    opts = options or IngestOptions()
    since = _parse_since(opts.since)
    path = Path(path)

    def _gen() -> Iterator[EvalRecord]:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_val = row.get(opts.timestamp_col)
                if not _row_after_since(ts_val, since):
                    continue
                rec = _make_record(row, opts, source_file=str(path))
                if rec is not None:
                    yield rec

    yield from _apply_post_filters(_gen(), opts)


def ingest_stdin(
    stream: io.TextIOBase, fmt: str = "jsonl", options: IngestOptions | None = None
) -> Iterator[EvalRecord]:
    """Yield EvalRecord rows from a text stream (stdin or equivalent).

    Args:
        stream: A text-mode stream.
        fmt: Either 'jsonl' or 'csv'.
        options: Ingestion options.
    """
    opts = options or IngestOptions()
    since = _parse_since(opts.since)

    def _gen_jsonl() -> Iterator[EvalRecord]:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            ts = obj.get(opts.timestamp_col)
            ts_str = ts if isinstance(ts, str) else None
            if not _row_after_since(ts_str, since):
                continue
            rec = _make_record(obj, opts, source_file="<stdin>")
            if rec is not None:
                yield rec

    def _gen_csv() -> Iterator[EvalRecord]:
        reader = csv.DictReader(stream)
        for row in reader:
            ts_val = row.get(opts.timestamp_col)
            if not _row_after_since(ts_val, since):
                continue
            rec = _make_record(row, opts, source_file="<stdin>")
            if rec is not None:
                yield rec

    if fmt == "jsonl":
        yield from _apply_post_filters(_gen_jsonl(), opts)
    elif fmt == "csv":
        yield from _apply_post_filters(_gen_csv(), opts)
    else:
        raise ValueError(f"unsupported stdin format: {fmt}")


def ingest_file(
    path: str | Path, fmt: str = "jsonl", options: IngestOptions | None = None
) -> Iterator[EvalRecord]:
    """Dispatch to the correct ingestion function based on ``fmt``."""
    if fmt == "jsonl":
        return ingest_jsonl(path, options)
    if fmt == "csv":
        return ingest_csv(path, options)
    raise ValueError(f"unsupported file format: {fmt}")
