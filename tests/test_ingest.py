"""Tests for src/ingest.py."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from src.ingest import (
    IngestOptions,
    ingest_csv,
    ingest_file,
    ingest_jsonl,
    ingest_stdin,
)


@pytest.fixture()
def jsonl_path(tmp_path: Path) -> Path:
    p = tmp_path / "input.jsonl"
    lines = [
        {"input": "i1", "output": "o1", "reference": "r1", "ts": "2024-01-01T00:00:00"},
        {"input": "i2", "output": "o2"},
        "not json",
        {"input": "i3", "output": "o3", "ts": "2024-06-01T00:00:00"},
    ]
    with p.open("w") as f:
        for line in lines:
            if isinstance(line, dict):
                f.write(json.dumps(line) + "\n")
            else:
                f.write(line + "\n")
    return p


@pytest.fixture()
def csv_path(tmp_path: Path) -> Path:
    p = tmp_path / "input.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["prompt", "response", "ground_truth", "ts"])
        w.writeheader()
        w.writerow(
            {
                "prompt": "i1",
                "response": "o1",
                "ground_truth": "r1",
                "ts": "2024-01-01",
            }
        )
        w.writerow(
            {
                "prompt": "i2",
                "response": "o2",
                "ground_truth": "",
                "ts": "2024-06-01",
            }
        )
    return p


def test_ingest_jsonl_basic(jsonl_path: Path) -> None:
    records = list(ingest_jsonl(jsonl_path))
    assert len(records) == 3
    assert records[0].input_text == "i1"
    assert records[0].reference_text == "r1"
    assert records[1].reference_text is None


def test_ingest_jsonl_lenient_skips_bad_lines(jsonl_path: Path) -> None:
    records = list(ingest_jsonl(jsonl_path))
    assert all(r.input_text.startswith("i") for r in records)


def test_ingest_jsonl_sample(jsonl_path: Path) -> None:
    opts = IngestOptions(sample=2, seed=42)
    records = list(ingest_jsonl(jsonl_path, opts))
    assert len(records) == 2


def test_ingest_jsonl_since(jsonl_path: Path) -> None:
    opts = IngestOptions(since="2024-05-01")
    records = list(ingest_jsonl(jsonl_path, opts))
    assert len(records) == 1
    assert records[0].input_text == "i3"


def test_ingest_csv_with_column_mapping(csv_path: Path) -> None:
    opts = IngestOptions(input_col="prompt", output_col="response", reference_col="ground_truth")
    records = list(ingest_csv(csv_path, opts))
    assert len(records) == 2
    assert records[0].input_text == "i1"
    assert records[0].reference_text == "r1"
    assert records[1].reference_text is None


def test_ingest_csv_since(csv_path: Path) -> None:
    opts = IngestOptions(
        input_col="prompt",
        output_col="response",
        reference_col="ground_truth",
        since="2024-05-01",
    )
    records = list(ingest_csv(csv_path, opts))
    assert len(records) == 1
    assert records[0].input_text == "i2"


def test_ingest_file_routes_by_format(jsonl_path: Path) -> None:
    records = list(ingest_file(jsonl_path, fmt="jsonl"))
    assert len(records) == 3


def test_ingest_file_routes_csv(csv_path: Path) -> None:
    opts = IngestOptions(input_col="prompt", output_col="response")
    records = list(ingest_file(csv_path, fmt="csv", options=opts))
    assert len(records) == 2


def test_ingest_file_unsupported_format(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("hi")
    with pytest.raises(ValueError):
        list(ingest_file(p, fmt="parquet"))


def test_ingest_stdin_jsonl() -> None:
    src = io.StringIO('{"input":"i","output":"o"}\n{"input":"i2","output":"o2"}\n')
    records = list(ingest_stdin(src, fmt="jsonl"))
    assert len(records) == 2
    assert records[1].input_text == "i2"


def test_ingest_stdin_csv() -> None:
    src = io.StringIO("input,output\ni,o\ni2,o2\n")
    records = list(ingest_stdin(src, fmt="csv"))
    assert len(records) == 2


def test_ingest_jsonl_missing_keys_skipped(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text('{"input":"only"}\n{"output":"only"}\n{"input":"a","output":"b"}\n')
    recs = list(ingest_jsonl(p))
    assert len(recs) == 1


def test_ingest_jsonl_limit(jsonl_path: Path) -> None:
    opts = IngestOptions(limit=1)
    recs = list(ingest_jsonl(jsonl_path, opts))
    assert len(recs) == 1


def test_ingest_options_defaults() -> None:
    opts = IngestOptions()
    assert opts.sample is None
    assert opts.limit is None
    assert opts.since is None
    assert opts.input_col == "input"
    assert opts.output_col == "output"
    assert opts.reference_col == "reference"
