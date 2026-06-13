"""Validate a qrels JSONL file against the v1 specification.

Checks:
- All records parse to QrelsEntryV1
- No duplicate (query_id, dataset_id) pairs
- All query_ids exist in the benchmark query file (when --queries provided)
- Relevance scores are 0-3
- Rationale required for relevance 0 and 3
- hard_negative_violation=True only on relevance=0 entries
- No duplicate annotation per (query_id, dataset_id, annotator_id)

Usage:
    python scripts/eval/validate_qrels.py artifacts/qrels.jsonl
    python scripts/eval/validate_qrels.py artifacts/qrels.jsonl \\
        --queries artifacts/benchmark_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from scripts.eval.benchmark_schema import (
    QrelsEntryV1,
    ValidationResult,
)


def _load_jsonl(path: Path) -> tuple[list[dict], list[str]]:
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


def _load_query_ids(queries_path: Path) -> set[str] | None:
    """Return set of known query_ids, or None if file is missing."""
    if not queries_path.exists():
        return None
    records, _ = _load_jsonl(queries_path)
    ids = set()
    for r in records:
        # accept both query_id forms
        qid = r.get("query_id")
        if qid:
            ids.add(qid)
    return ids


def validate_qrels(  # noqa: C901
    path: Path,
    queries_path: Path | None = None,
    strict: bool = False,
) -> tuple[list[QrelsEntryV1], ValidationResult]:
    result = ValidationResult(ok=True)

    if not path.exists():
        result.add_error("file", f"File not found: {path}")
        return [], result

    raw_records, parse_errors = _load_jsonl(path)
    for err in parse_errors:
        result.add_error("json", err)

    result.record_count = len(raw_records)

    known_query_ids: set[str] | None = None
    if queries_path is not None:
        known_query_ids = _load_query_ids(queries_path)
        if known_query_ids is None:
            result.add_warning("queries_path", f"Query file not found: {queries_path}")

    entries: list[QrelsEntryV1] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_annotator_triples: set[tuple[str, str, str]] = set()

    for i, record in enumerate(raw_records):
        record_id = f"{record.get('query_id','?')}:{record.get('dataset_id','?')}"
        try:
            entry = QrelsEntryV1.model_validate(record)
        except Exception as e:
            result.add_error("schema", str(e), record_id)
            continue

        pair = (entry.query_id, entry.dataset_id)
        annotator_triple = (entry.query_id, entry.dataset_id, entry.annotator_id)

        # Duplicate pair (multi-annotator is fine, but same annotator twice is not)
        if annotator_triple in seen_annotator_triples:
            result.add_error(
                "duplicate",
                f"Duplicate (query_id, dataset_id, annotator_id): {annotator_triple}",
                record_id,
            )
        seen_annotator_triples.add(annotator_triple)
        seen_pairs.add(pair)

        # Query ID exists
        if known_query_ids is not None and entry.query_id not in known_query_ids:
            result.add_error(
                "query_id",
                f"query_id {entry.query_id!r} not found in benchmark query file",
                record_id,
            )

        # Rationale required for 0 and 3
        if entry.requires_rationale() and not entry.rationale.strip():
            msg = f"relevance={entry.relevance} requires a non-empty rationale"
            if strict:
                result.add_error("rationale", msg, record_id)
            else:
                result.add_warning("rationale", msg, record_id)

        # hard_negative_violation only valid on relevance=0
        if entry.hard_negative_violation and entry.relevance != 0:
            result.add_error(
                "hard_negative_violation",
                f"hard_negative_violation=True but relevance={entry.relevance} (must be 0)",  # noqa: E501
                record_id,
            )

        # annotator_id should not be empty
        if not entry.annotator_id.strip():
            msg = "annotator_id is empty"
            if strict:
                result.add_error("annotator_id", msg, record_id)
            else:
                result.add_warning("annotator_id", msg, record_id)

        entries.append(entry)

    # Summary stats
    if entries:
        rel_counts = Counter(e.relevance for e in entries)
        hn_count = sum(1 for e in entries if e.hard_negative_violation)
        adjudicated = sum(1 for e in entries if e.adjudicated)

        print(f"  Relevance distribution: {dict(sorted(rel_counts.items()))}")
        print(f"  Hard-negative violations: {hn_count}")
        print(f"  Adjudicated: {adjudicated} / {len(entries)}")
        print(f"  Unique (query, dataset) pairs: {len(seen_pairs)}")

    return entries, result


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
    parser = argparse.ArgumentParser(description="Validate qrels JSONL file")
    parser.add_argument("path", type=Path, help="Path to qrels JSONL")
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="Optional: benchmark queries JSONL for cross-referencing query_ids",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors",
    )
    args = parser.parse_args(argv)

    entries, result = validate_qrels(args.path, args.queries, strict=args.strict)
    _print_result(result, args.path)

    if result.ok:
        print(f"\n{len(entries)} qrels entries validated successfully.")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
