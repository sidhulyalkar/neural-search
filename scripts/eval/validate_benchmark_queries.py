"""Validate benchmark query file against the v1 specification.

Usage:
    python scripts/eval/validate_benchmark_queries.py artifacts/benchmark_queries.jsonl
    python scripts/eval/validate_benchmark_queries.py \
        --strict artifacts/benchmark_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.eval.benchmark_schema import (
    CANONICAL_INTENTS,
    BenchmarkQueryV1,
    ValidationResult,
)


def _load_queries(path: Path) -> tuple[list[dict], list[str]]:
    """Return (records, parse_errors)."""
    records: list[dict] = []
    errors: list[str] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"line {lineno}: {e}")
    return records, errors


def validate_queries(  # noqa: C901
    path: Path, strict: bool = False
) -> tuple[list[BenchmarkQueryV1], ValidationResult]:
    """Validate all queries in the JSONL file.

    Returns parsed queries (empty list on parse failure) and a ValidationResult.
    """
    result = ValidationResult(ok=True)

    if not path.exists():
        result.add_error("file", f"File not found: {path}")
        return [], result

    raw_records, parse_errors = _load_queries(path)
    for err in parse_errors:
        result.add_error("json", err)

    result.record_count = len(raw_records)
    if not raw_records:
        result.add_error("file", "No records found in query file")
        return [], result

    queries: list[BenchmarkQueryV1] = []
    seen_ids: set[str] = set()

    for i, record in enumerate(raw_records):
        record_id = record.get("query_id", f"row_{i}")
        try:
            q = BenchmarkQueryV1.model_validate(record)
        except Exception as e:
            result.add_error("schema", str(e), record_id)
            continue

        # Duplicate IDs
        if q.query_id in seen_ids:
            result.add_error("query_id", f"Duplicate query_id: {q.query_id}", record_id)
        seen_ids.add(q.query_id)

        # Non-empty query_text
        if not q.query_text.strip():
            result.add_error("query_text", "query_text is empty", record_id)

        # Non-empty scientific_goal
        if not q.scientific_goal.strip():
            result.add_error("scientific_goal", "scientific_goal is empty", record_id)

        # must_have non-empty
        if not q.must_have:
            msg = "must_have is empty — every query needs ≥1 constraint"
            if strict:
                result.add_error("must_have", msg, record_id)
            else:
                result.add_warning("must_have", msg, record_id)

        # hard_negatives non-empty
        if not q.hard_negatives:
            msg = "hard_negatives is empty — specify at least one false-positive pattern"  # noqa: E501
            if strict:
                result.add_error("hard_negatives", msg, record_id)
            else:
                result.add_warning("hard_negatives", msg, record_id)

        # Non-canonical intents become warnings
        if q.intent not in CANONICAL_INTENTS:
            result.add_warning(
                "intent",
                f"intent {q.intent!r} is a legacy alias; canonical = {q.canonical_intent()!r}",  # noqa: E501
                record_id,
            )

        queries.append(q)

    # Intent diversity
    if queries:
        canonical_counts = Counter(q.canonical_intent() for q in queries)
        missing = CANONICAL_INTENTS - set(canonical_counts.keys())
        if missing:
            result.add_warning(
                "intent_diversity",
                f"Intents not covered: {sorted(missing)}",
            )

    return queries, result


def _print_result(result: ValidationResult, path: Path) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"[{status}] {path} — {result.record_count} records")

    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for e in result.errors:
            ref = f" [{e.record_id}]" if e.record_id else ""
            print(f"  ERROR  {e.field}{ref}: {e.message}")

    if result.warnings:
        print(f"\nWarnings ({len(result.warnings)}):")
        for w in result.warnings:
            ref = f" [{w.record_id}]" if w.record_id else ""
            print(f"  WARN   {w.field}{ref}: {w.message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate benchmark query JSONL file")
    parser.add_argument("path", type=Path, help="Path to benchmark queries JSONL")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    args = parser.parse_args(argv)

    queries, result = validate_queries(args.path, strict=args.strict)
    _print_result(result, args.path)

    if result.ok:
        print(f"\n{len(queries)} queries parsed successfully.")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
