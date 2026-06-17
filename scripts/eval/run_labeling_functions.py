#!/usr/bin/env python3
"""Run all labeling functions over pair_evidence.jsonl.

Usage:
    python scripts/eval/run_labeling_functions.py \
        --evidence artifacts/eval/pair_evidence.jsonl \
        --out artifacts/eval/label_function_votes.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.evidence import DatasetEvidence, PairEvidence, QuerySpec
from neural_search.eval.labeling_functions import run_all_lfs


def _load_pair(row: dict) -> PairEvidence:
    q_data = row["query"]
    d_data = row["dataset"]
    return PairEvidence(
        query_id=row["query_id"],
        record_id=row["record_id"],
        query=QuerySpec(**q_data),
        dataset=DatasetEvidence(**d_data),
        pooled_from=row.get("pooled_from", []),
        min_rank=row.get("min_rank", 1000),
        priority=row.get("priority", "normal"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.evidence.open(encoding="utf-8") as in_fh, \
         args.out.open("w", encoding="utf-8") as out_fh:
        for line in in_fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pair = _load_pair(row)
            votes = run_all_lfs(pair)
            record = {
                "query_id": pair.query_id,
                "record_id": pair.record_id,
                "votes": [v.to_dict() for v in votes],
            }
            out_fh.write(json.dumps(record) + "\n")
            written += 1

    print(f"Labeling functions applied to {written} pairs → {args.out}")


if __name__ == "__main__":
    main()
