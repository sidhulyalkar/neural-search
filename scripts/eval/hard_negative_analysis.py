#!/usr/bin/env python3
"""Dedicated hard-negative violation analysis across all qrel tiers.

Produces a report showing:
- Overall HN violation rate per tier
- Violations broken down by query
- Violations broken down by source adapter

Usage:
    python scripts/eval/hard_negative_analysis.py \
        --qrels-gold artifacts/qrels_gold.jsonl \
        --qrels-silver artifacts/qrels_silver.jsonl \
        --qrels-bronze artifacts/qrels_bronze.jsonl \
        --run reports/eval/runs/usefulness.jsonl \
        --out reports/eval/hard_negative_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_qrels(path: Path | None) -> dict[str, dict[str, int]]:
    """Return {query_id: {record_id: label}}."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qrels[row["query_id"]][row["record_id"]] = int(row["label"])
    return dict(qrels)


def _load_run(path: Path) -> dict[str, list[str]]:
    """Return {query_id: [record_id ordered by rank]}."""
    if not path.exists():
        return {}
    rows_by_q: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_q[row["query_id"]].append(
                (int(row.get("rank", 10**9)), row["record_id"])
            )
    return {
        qid: [rid for _, rid in sorted(rows)]
        for qid, rows in rows_by_q.items()
    }


def _violations(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    cutoff: int = 10,
) -> dict:
    """Return per-query violation counts and overall rate."""
    total_queries = 0
    violated_queries = 0
    total_hn = 0
    by_query: dict[str, int] = {}
    by_source: dict[str, int] = defaultdict(int)

    for qid, ranked in run.items():
        q_qrels = qrels.get(qid, {})
        hard_neg_ids = {rid for rid, lbl in q_qrels.items() if lbl == 0}
        if not hard_neg_ids:
            continue
        total_queries += 1
        total_hn += len(hard_neg_ids)
        top_k = ranked[:cutoff]
        violations = [rid for rid in top_k if rid in hard_neg_ids]
        if violations:
            violated_queries += 1
            by_query[qid] = len(violations)
            for rid in violations:
                source = rid.split(":")[0]
                by_source[source] += 1

    rate = violated_queries / total_queries if total_queries else 0.0
    return {
        "total_queries_with_hn": total_queries,
        "total_hard_negatives": total_hn,
        "violated_queries": violated_queries,
        "violation_rate": round(rate, 4),
        "by_query": by_query,
        "by_source": dict(by_source),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels-gold", type=Path, default=None)
    parser.add_argument("--qrels-silver", type=Path, default=None)
    parser.add_argument("--qrels-bronze", type=Path, default=None)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cutoff", type=int, default=10)
    args = parser.parse_args()

    run = _load_run(args.run)
    report: dict = {"cutoff": args.cutoff}

    for tier_name, tier_path in [
        ("gold", args.qrels_gold),
        ("silver", args.qrels_silver),
        ("bronze", args.qrels_bronze),
    ]:
        qrels = _load_qrels(tier_path)
        if qrels:
            report[tier_name] = _violations(qrels, run, args.cutoff)
        else:
            report[tier_name] = {"note": f"{tier_name} qrels not available"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Hard-negative report → {args.out}")

    for tier in ("gold", "silver", "bronze"):
        if "violation_rate" in report.get(tier, {}):
            vr = report[tier]["violation_rate"]
            vq = report[tier]["violated_queries"]
            tq = report[tier]["total_queries_with_hn"]
            print(f"  {tier.upper():8s}: violation_rate={vr:.1%}  ({vq}/{tq} queries)")


if __name__ == "__main__":
    main()
