#!/usr/bin/env python3
"""Aggregate LF votes (+ optional LLM judgments) into gold/silver/bronze qrels.

Usage:
    python scripts/eval/build_qrels_from_votes.py \
        --evidence artifacts/eval/pair_evidence.jsonl \
        --votes artifacts/eval/label_function_votes.jsonl \
        --out-gold artifacts/qrels_gold.jsonl \
        --out-silver artifacts/qrels_silver.jsonl \
        --out-bronze artifacts/qrels_bronze.jsonl \
        --audit-queue artifacts/eval/audit_queue.jsonl

    # With optional LLM judgments:
        --llm artifacts/eval/llm_judgments.jsonl

    # With human audits to promote to gold:
        --human-audits artifacts/eval/human_audits.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.eval.evidence import DatasetEvidence, LFVote, PairEvidence, QuerySpec
from neural_search.eval.label_ensemble import (
    EnsembleResult,
    aggregate_votes,
    assign_tier,
    compute_audit_priority,
    make_qrel,
)

_MIN_RANK_DEFAULT = 1000


def _load_jsonl(path: Path | None) -> list[dict]:
    if path is None or not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _parse_votes(rows: list[dict]) -> dict[tuple[str, str], list[LFVote]]:
    out: dict[tuple[str, str], list[LFVote]] = {}
    for row in rows:
        key = (row["query_id"], row["record_id"])
        out[key] = [LFVote(**v) for v in row.get("votes", [])]
    return out


def _parse_llm(rows: list[dict]) -> dict[tuple[str, str], LFVote]:
    out: dict[tuple[str, str], LFVote] = {}
    for row in rows:
        key = (row["query_id"], row["record_id"])
        boosted_conf = min(float(row.get("confidence", 0.5)) * 1.5, 1.0)
        out[key] = LFVote(
            lf_name="llm_judge",
            label=int(row.get("label", 1)),
            confidence=boosted_conf,
            rationale=row.get("rationale", ""),
            abstain=False,
        )
    return out


def _parse_human_audits(rows: list[dict]) -> dict[tuple[str, str], int]:
    return {
        (r["query_id"], r["record_id"]): int(r["label"])
        for r in rows
        if r.get("audit_status") == "done" and "label" in r
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--votes", required=True, type=Path)
    parser.add_argument("--llm", type=Path, default=None)
    parser.add_argument("--human-audits", type=Path, default=None)
    parser.add_argument("--out-gold", required=True, type=Path)
    parser.add_argument("--out-silver", required=True, type=Path)
    parser.add_argument("--out-bronze", required=True, type=Path)
    parser.add_argument("--audit-queue", required=True, type=Path)
    args = parser.parse_args()

    evidence_rows = _load_jsonl(args.evidence)
    votes_by_pair = _parse_votes(_load_jsonl(args.votes))
    llm_by_pair = _parse_llm(_load_jsonl(args.llm))
    human_labels = _parse_human_audits(_load_jsonl(args.human_audits))

    for p in (args.out_gold, args.out_silver, args.out_bronze, args.audit_queue):
        p.parent.mkdir(parents=True, exist_ok=True)

    counts = {"gold": 0, "silver": 0, "bronze": 0, "audit": 0}

    with (args.out_gold.open("w", encoding="utf-8") as gold_fh,
          args.out_silver.open("w", encoding="utf-8") as silver_fh,
          args.out_bronze.open("w", encoding="utf-8") as bronze_fh,
          args.audit_queue.open("w", encoding="utf-8") as aq_fh):

        for row in evidence_rows:
            qid = row["query_id"]
            rid = row["record_id"]
            key = (qid, rid)

            votes: list[LFVote] = list(votes_by_pair.get(key, []))
            if key in llm_by_pair:
                votes.append(llm_by_pair[key])

            if not votes:
                from neural_search.eval.labeling_functions import run_all_lfs
                try:
                    pair = PairEvidence(
                        query_id=qid, record_id=rid,
                        query=QuerySpec(**row["query"]),
                        dataset=DatasetEvidence(**row["dataset"]),
                        pooled_from=row.get("pooled_from", []),
                        min_rank=row.get("min_rank", _MIN_RANK_DEFAULT),
                        priority=row.get("priority", "normal"),
                    )
                    votes = run_all_lfs(pair)
                except Exception:
                    pass

            result = aggregate_votes(votes)
            human_audited = key in human_labels
            if human_audited:
                result = EnsembleResult(
                    label=human_labels[key],
                    confidence=1.0,
                    tier="gold",
                    hard_negative_triggered=result.hard_negative_triggered,
                    disagreement=result.disagreement,
                    active_vote_count=result.active_vote_count,
                    audit_priority=result.audit_priority,
                    provenance=result.provenance + ["human_audit"],
                )

            tier = assign_tier(result, human_audited=human_audited)
            result.audit_priority = compute_audit_priority(
                result, min_rank=row.get("min_rank", _MIN_RANK_DEFAULT)
            )
            qrel = make_qrel(qid, rid, result, tier)

            if tier == "gold":
                gold_fh.write(json.dumps(qrel) + "\n")
                counts["gold"] += 1
            elif tier == "silver":
                silver_fh.write(json.dumps(qrel) + "\n")
                counts["silver"] += 1
            else:
                bronze_fh.write(json.dumps(qrel) + "\n")
                counts["bronze"] += 1

            if not human_audited and (result.disagreement > 0.5 or result.hard_negative_triggered):
                aq_entry = {**qrel, "audit_priority": round(result.audit_priority, 4),
                            "pair_evidence": row}
                aq_fh.write(json.dumps(aq_entry) + "\n")
                counts["audit"] += 1

    print(f"Gold: {counts['gold']}, Silver: {counts['silver']}, "
          f"Bronze: {counts['bronze']}, Audit queue: {counts['audit']}")


if __name__ == "__main__":
    main()
