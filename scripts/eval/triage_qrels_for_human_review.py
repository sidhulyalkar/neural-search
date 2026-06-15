"""Build a priority-ordered human audit queue from neuro_judge outputs.

Priority factors:
  - judge disagreement / conflicts
  - low confidence
  - high retrieval rank but low label (false positive risk)
  - low retrieval rank but high label (missed relevant)
  - hard-negative disagreement
  - label 2-3 with missing required evidence
  - NDCG@10 impact estimate
  - underrepresented intents / modalities / archives

Usage::

    python scripts/eval/triage_qrels_for_human_review.py \
        --consensus artifacts/field_state/neuro_qrels_consensus.jsonl \
        --conflicts artifacts/field_state/neuro_qrels_conflicts.jsonl \
        --candidates artifacts/field_state/qrels_candidates_pooled.jsonl \
        --out artifacts/field_state/expert_audit_priority_queue.jsonl \
        --sample artifacts/field_state/expert_audit_sample_100.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_INTENT_FREQ: dict[str, int] = {}   # populated during run


def _score_record(
    rec: dict,
    cand: dict | None,
    is_conflict: bool,
) -> float:
    score = 0.0

    # Conflict always high priority
    if is_conflict:
        score += 3.0

    confidence = float(rec.get("confidence", 0.5))
    label = int(rec.get("label", 0))

    # Low confidence
    if confidence < 0.6:
        score += (0.6 - confidence) * 3.0

    # High label but missing information
    missing = len(rec.get("missing_information", []))
    if label >= 2 and missing >= 2:
        score += 2.0
    elif label >= 2 and missing >= 1:
        score += 1.0

    # Hard-negative detected at label > 0 (suspicious)
    if rec.get("hard_negative_detected") and label > 0:
        score += 2.5

    # Retrieval rank signals from candidate metadata
    if cand:
        ranks = cand.get("ranks_by_system", {})
        if ranks:
            top_rank = min(ranks.values())
            # High rank (top 5) but low label → false positive risk
            if top_rank <= 5 and label <= 1:
                score += 2.0 + (5 - top_rank) * 0.3
            # Low rank (>20) but high label → missed relevant
            if top_rank > 20 and label >= 2:
                score += 1.5

    # Underrepresented intent
    intent = rec.get("query_intent", "UNKNOWN")
    if _INTENT_FREQ.get(intent, 999) <= 30:
        score += 0.5

    return score


# ---------------------------------------------------------------------------
# Sampling plan
# ---------------------------------------------------------------------------


def _sample_audit(
    priority_queue: list[dict],
    consensus: list[dict],
    conflicts: list[dict],
    n: int = 100,
) -> list[dict]:
    """Draw a stratified sample of n records for the human audit."""
    # Split into strata
    label_0 = [r for r in consensus if r.get("label") == 0]
    label_3 = [r for r in consensus if r.get("label") == 3]
    boundary = [r for r in consensus if r.get("label") in (1, 2)]

    # Sort each stratum by priority score
    def key(r):
        return r.get("_priority_score", 0)
    label_0.sort(key=key, reverse=True)
    label_3.sort(key=key, reverse=True)
    boundary.sort(key=key, reverse=True)

    # Calculate stratum sizes (20-20-20-20-20)
    per_stratum = n // 5
    sample: list[dict] = []
    sample += label_0[:per_stratum]
    sample += label_3[:per_stratum]
    sample += boundary[:per_stratum]
    sample += [dict(r, _stratum="conflict") for r in conflicts[:per_stratum]]

    # High-impact NDCG cases (remaining slots)
    ndcg_high = sorted(priority_queue, key=lambda r: r.get("_priority_score", 0), reverse=True)
    seen = {(r.get("query_id"), r.get("dataset_id")) for r in sample}
    for r in ndcg_high:
        if len(sample) >= n:
            break
        key = (r.get("query_id"), r.get("dataset_id"))
        if key not in seen:
            sample.append(dict(r, _stratum="high_impact"))
            seen.add(key)

    return sample[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build human audit priority queue")
    parser.add_argument("--consensus", default="artifacts/field_state/neuro_qrels_consensus.jsonl")
    parser.add_argument("--conflicts", default="artifacts/field_state/neuro_qrels_conflicts.jsonl")
    parser.add_argument("--candidates", default="artifacts/field_state/qrels_candidates_pooled.jsonl")
    parser.add_argument("--out", default="artifacts/field_state/expert_audit_priority_queue.jsonl")
    parser.add_argument("--sample", default="artifacts/field_state/expert_audit_sample_100.jsonl")
    parser.add_argument("--sample-size", type=int, default=100)
    args = parser.parse_args(argv)

    consensus = _load_jsonl(_REPO / args.consensus)
    conflicts = _load_jsonl(_REPO / args.conflicts)
    candidates = _load_jsonl(_REPO / args.candidates)

    if not consensus:
        sys.exit(f"[ERROR] No consensus records found at {args.consensus}")

    # Index candidates by (query_id, dataset_id)
    cand_idx: dict[tuple[str, str], dict] = {}
    for c in candidates:
        cand_idx[(str(c.get("query_id", "")), str(c.get("dataset_id", "")))] = c

    # Build intent frequency map for under-representation scoring
    global _INTENT_FREQ
    _INTENT_FREQ = dict(Counter(r.get("query_intent", "UNKNOWN") for r in consensus))

    # Build conflict set
    conflict_keys = {
        (str(c.get("query_id", "")), str(c.get("dataset_id", "")))
        for c in conflicts
    }

    # Score all consensus records
    priority_queue: list[dict] = []
    for rec in consensus:
        key = (str(rec.get("query_id", "")), str(rec.get("dataset_id", "")))
        cand = cand_idx.get(key)
        is_conflict = key in conflict_keys
        score = _score_record(rec, cand, is_conflict)
        enriched = dict(rec)
        enriched["_priority_score"] = round(score, 3)
        enriched["_is_conflict"] = is_conflict
        priority_queue.append(enriched)

    # Sort descending by priority score
    priority_queue.sort(key=lambda r: r["_priority_score"], reverse=True)

    out_path = _REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in priority_queue:
            fh.write(json.dumps(r) + "\n")

    sample = _sample_audit(priority_queue, consensus, conflicts, args.sample_size)
    sample_path = _REPO / args.sample
    with sample_path.open("w") as fh:
        for r in sample:
            fh.write(json.dumps(r) + "\n")

    print(f"Priority queue: {len(priority_queue)} records → {out_path}")
    print(f"Audit sample:   {len(sample)} records    → {sample_path}")

    # Stats
    scores = [r["_priority_score"] for r in priority_queue]
    high_pri = sum(1 for s in scores if s >= 3)
    med_pri = sum(1 for s in scores if 1 <= s < 3)
    low_pri = sum(1 for s in scores if s < 1)
    print(f"  High priority (≥3): {high_pri}")
    print(f"  Med  priority (1-3): {med_pri}")
    print(f"  Low  priority (<1):  {low_pri}")


if __name__ == "__main__":
    main()
