#!/usr/bin/env python3
"""Compute corpus extraction-quality tables without inventing audit metrics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CORPUS = Path("data/corpus/normalized/combined_corpus.jsonl")
DEFAULT_REJECTIONS = Path("data/corpus/rejected/tier2_rejected.jsonl")
DEFAULT_OUT_DIR = Path("reports/eval")

CORE_FIELDS = [
    "title",
    "url",
    "description",
    "license",
    "doi",
    "species",
    "modalities",
    "brain_regions",
    "tasks",
    "behaviors",
    "data_standards",
]


def present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list | tuple | set | dict):
        return bool(value)
    return True


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--rejections", type=Path, default=DEFAULT_REJECTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        raise SystemExit(f"corpus not found: {args.corpus}")

    records = load_jsonl(args.corpus)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[str(record.get("source", "unknown"))].append(record)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    with (args.out_dir / "field_completeness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "field", "non_empty", "total", "coverage"],
        )
        writer.writeheader()
        for source, source_records in sorted(by_source.items()):
            total = len(source_records)
            for field in CORE_FIELDS:
                non_empty = sum(1 for record in source_records if present(record.get(field)))
                writer.writerow(
                    {
                        "source": source,
                        "field": field,
                        "non_empty": non_empty,
                        "total": total,
                        "coverage": round(non_empty / total, 4) if total else 0.0,
                    }
                )

    with (args.out_dir / "pid_license_coverage.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "records", "doi_or_pid", "license", "doi_or_pid_coverage", "license_coverage"],
        )
        writer.writeheader()
        for source, source_records in sorted(by_source.items()):
            total = len(source_records)
            pid_count = sum(
                1 for record in source_records if present(record.get("doi")) or present(record.get("url"))
            )
            license_count = sum(1 for record in source_records if present(record.get("license")))
            writer.writerow(
                {
                    "source": source,
                    "records": total,
                    "doi_or_pid": pid_count,
                    "license": license_count,
                    "doi_or_pid_coverage": round(pid_count / total, 4) if total else 0.0,
                    "license_coverage": round(license_count / total, 4) if total else 0.0,
                }
            )

    rejection_counts: Counter[str] = Counter()
    if args.rejections.exists():
        with args.rejections.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rejection_counts[str(row.get("reason", "unknown"))] += 1

    with (args.out_dir / "rejection_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reason", "count"])
        writer.writeheader()
        for reason, count in rejection_counts.most_common():
            writer.writerow({"reason": reason, "count": count})

    with (args.out_dir / "manual_extraction_precision.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source", "field", "sample_size", "precision", "status"],
        )
        writer.writeheader()
        for source in sorted(by_source):
            for field in ["modalities", "species", "tasks"]:
                writer.writerow(
                    {
                        "source": source,
                        "field": field,
                        "sample_size": "",
                        "precision": "",
                        "status": "Pending benchmark artifact",
                    }
                )

    print(json.dumps({"records": len(records), "sources": len(by_source)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
