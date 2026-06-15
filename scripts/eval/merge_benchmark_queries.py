#!/usr/bin/env python3
"""Merge all benchmark query YAML files into a single canonical set.

Reads every benchmark_queries*.yaml under data/eval/, extracts query objects
(any dict with both 'id' and 'query' keys), deduplicates by normalized query
text, and writes a canonical YAML with stable IDs.

Output: data/eval/benchmark_queries_canonical.yaml

Usage:
    python scripts/eval/merge_benchmark_queries.py
    python scripts/eval/merge_benchmark_queries.py --out data/eval/my_queries.yaml
    python scripts/eval/merge_benchmark_queries.py --files data/eval/benchmark_queries.yaml data/eval/benchmark_queries_v2.yaml
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_EVAL_DIR = Path("data/eval")
DEFAULT_OUT = Path("data/eval/benchmark_queries_canonical.yaml")
DEFAULT_GLOB = "benchmark_queries*.yaml"

# Files to skip (contain no standard query objects or are output files)
SKIP_FILES = {"benchmark_queries_canonical.yaml"}


def _extract_queries(data: Any) -> list[dict[str, Any]]:
    """Recursively extract all dicts that look like query objects."""
    found: list[dict[str, Any]] = []
    if isinstance(data, dict):
        if "id" in data and "query" in data:
            found.append(data)
        else:
            for value in data.values():
                found.extend(_extract_queries(value))
    elif isinstance(data, list):
        for item in data:
            found.extend(_extract_queries(item))
    return found


def _normalize_query_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _assign_intent(q: dict[str, Any]) -> str:
    if "intent" in q:
        return str(q["intent"])
    text = _normalize_query_text(str(q.get("query", "")))
    if any(kw in text for kw in ["known item", "find the", "specific dataset"]):
        return "STRICT_LOOKUP"
    if any(kw in text for kw in ["meta-analysis", "across studies", "multiple datasets"]):
        return "META_ANALYSIS"
    if any(kw in text for kw in ["pipeline", "reuse", "re-use", "apply method"]):
        return "PIPELINE_REUSE"
    if any(kw in text for kw in ["compare", "comparison", "cross-dataset"]):
        return "CROSS_DATASET_COMPARISON"
    if any(kw in text for kw in ["replicate", "replication", "reproduce"]):
        return "REPLICATION"
    if any(kw in text for kw in ["decode", "decoding", "classify", "classification"]):
        return "REANALYSIS_FEASIBILITY"
    return "EXPLORATION"


def _canonical_query(
    idx: int,
    q: dict[str, Any],
    source_file: str,
) -> dict[str, Any]:
    canonical_id = f"can_{idx:04d}"
    entry: dict[str, Any] = {
        "id": canonical_id,
        "query": q["query"],
        "intent": _assign_intent(q),
        "source_file": source_file,
        "original_id": str(q["id"]),
    }
    # Carry forward structured constraint fields
    for field in (
        "expected_dataset_ids",
        "expected_sources",
        "expected_modalities_any",
        "expected_species",
        "expected_tasks",
        "expected_regions_any",
        "expected_behaviors",
        "expected_analysis_any",
        "hard_negative_modalities",
        "hard_negative_tasks",
        "hard_negative_regions",
        "minimum_precision_at_5",
        "minimum_label_recall_at_10",
    ):
        if field in q:
            entry[field] = q[field]
    return entry


def merge(
    source_files: list[Path],
    out: Path,
) -> list[dict[str, Any]]:
    seen_texts: dict[str, str] = {}  # normalized_text → canonical_id
    canonical: list[dict[str, Any]] = []

    for src in sorted(source_files):
        if src.name in SKIP_FILES:
            continue
        try:
            data = yaml.safe_load(src.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  Warning: could not parse {src.name}: {exc}")
            continue

        raw_queries = _extract_queries(data)
        added = 0
        for q in raw_queries:
            norm = _normalize_query_text(str(q["query"]))
            if norm in seen_texts:
                continue
            idx = len(canonical) + 1
            entry = _canonical_query(idx, q, src.name)
            seen_texts[norm] = entry["id"]
            canonical.append(entry)
            added += 1
        print(f"  {src.name}: {len(raw_queries)} queries, {added} new after dedup")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.dump(
            {"benchmark_queries": canonical},
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return canonical


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge benchmark query YAML files.")
    parser.add_argument(
        "--eval-dir", type=Path, default=DEFAULT_EVAL_DIR,
        help="Directory to glob for benchmark_queries*.yaml",
    )
    parser.add_argument(
        "--files", type=Path, nargs="+", default=None,
        help="Explicit list of YAML files (overrides --eval-dir glob)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if args.files:
        source_files = list(args.files)
    else:
        source_files = sorted(args.eval_dir.glob(DEFAULT_GLOB))

    print(f"Merging {len(source_files)} files → {args.out}")
    canonical = merge(source_files, args.out)
    print(f"\nDone: {len(canonical)} unique queries → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
