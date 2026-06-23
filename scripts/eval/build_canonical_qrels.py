#!/usr/bin/env python3
"""Emit canonical qrels in both TREC and JSONL form from llm_judgments.jsonl.

Drops judge_error rows and deduplicates (query_id, dataset_id) keeping first.

Usage:
    PYTHONPATH=. python scripts/eval/build_canonical_qrels.py   # run from repo root
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_JUDGMENTS = Path("data/qrels/llm_judgments.jsonl")
DEFAULT_TREC = Path("data/qrels/qrels.canonical.trec")
DEFAULT_JSONL = Path("data/qrels/qrels.canonical.jsonl")


def canonicalize(judgments: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    seen: set[tuple[str, str]] = set()
    trec: list[str] = []
    rows: list[dict[str, Any]] = []
    for rec in judgments:
        # Error rows are written by run_parallel_llm_qrels with rationale_short
        # prefixed "judge_error: ..."; matching that substring is the established
        # convention for detecting them (there is no dedicated error field).
        if "judge_error" in str(rec.get("rationale_short", "")):
            continue
        qid, did = str(rec["query_id"]), str(rec["dataset_id"])
        if (qid, did) in seen:
            continue
        label = int(rec.get("label", -1))
        if label < 0:
            continue
        seen.add((qid, did))
        trec.append(f"{qid} 0 {did} {label}")
        rows.append({"query_id": qid, "dataset_id": did, "label": label})
    return trec, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    ap.add_argument("--trec", type=Path, default=DEFAULT_TREC)
    ap.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    args = ap.parse_args(argv)

    if not args.judgments.exists():
        print(f"Judgments file not found: {args.judgments}", file=sys.stderr)
        return 1
    judgments = [json.loads(l) for l in args.judgments.read_text(encoding="utf-8").splitlines() if l.strip()]
    trec, rows = canonicalize(judgments)
    n_in = len(judgments)
    n_errors = sum(1 for r in judgments if "judge_error" in str(r.get("rationale_short", "")))
    n_dropped = n_in - len(rows)
    print(f"Read {n_in} judgments; dropped {n_dropped} ({n_errors} judge_error, "
          f"{n_dropped - n_errors} dup/invalid-label).")
    args.trec.parent.mkdir(parents=True, exist_ok=True)
    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.trec.write_text("\n".join(trec) + "\n", encoding="utf-8")
    with args.jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(trec)} qrels -> {args.trec} and {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
