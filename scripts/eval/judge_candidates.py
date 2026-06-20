#!/usr/bin/env python3
"""Optionally judge query-dataset pairs with an LLM rubric judge.

Exits with code 0 and a warning if ANTHROPIC_API_KEY is not set or the
config file is missing — does not block the pipeline.

Usage:
    python scripts/eval/judge_candidates.py \
        --evidence artifacts/eval/pair_evidence.jsonl \
        --config configs/judges/rubric_judge_v1.yaml \
        --out artifacts/eval/llm_judgments.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.evidence import DatasetEvidence, PairEvidence, QuerySpec
from neural_search.eval.llm_judge import judge_pair, load_config

_MIN_RANK_DEFAULT = 1000


def _load_pairs(evidence_path: Path) -> list[PairEvidence]:
    pairs = []
    with evidence_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append(PairEvidence(
                query_id=row["query_id"],
                record_id=row["record_id"],
                query=QuerySpec(**row["query"]),
                dataset=DatasetEvidence(**row["dataset"]),
                pooled_from=row.get("pooled_from", []),
                min_rank=row.get("min_rank", _MIN_RANK_DEFAULT),
                priority=row.get("priority", "normal"),
            ))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--votes", type=Path, default=None)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(str(args.config))
    if config is None:
        print(f"Warning: judge config not found at {args.config} — skipping LLM judging.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("", encoding="utf-8")
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set — skipping LLM judging.")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("", encoding="utf-8")
        return

    pairs = _load_pairs(args.evidence)
    max_pairs = config.get("max_pairs") or len(pairs)
    pairs = pairs[:max_pairs]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = skipped = 0

    with args.out.open("w", encoding="utf-8") as out_fh:
        for pair in pairs:
            judgment = judge_pair(pair, config)
            if judgment is None:
                skipped += 1
                continue
            record = {
                "query_id": pair.query_id,
                "record_id": pair.record_id,
                **judgment,
            }
            out_fh.write(json.dumps(record) + "\n")
            written += 1

    print(f"LLM judgments written: {written}, skipped: {skipped}")


if __name__ == "__main__":
    main()
