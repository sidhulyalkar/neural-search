#!/usr/bin/env python3
"""Grid-search dimension weights to maximize NDCG@3 on labeled usefulness pairs.

For each intent, perturbs each dimension weight by ±STEP and keeps the change
if NDCG@3 improves. Runs for --n-trials passes over all dimensions.

Usage:
    python scripts/optimize_usefulness_weights.py
    python scripts/optimize_usefulness_weights.py --n-trials 5 --out reports/optimized_weights.json
    python scripts/optimize_usefulness_weights.py --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

SEED_FILE = Path("data/eval/usefulness_seed_pairs.jsonl")
LABEL_TO_INT = {"not_useful": 0, "weakly_useful": 1, "useful": 2, "highly_useful": 3}
K = 3
STEP = 0.05
MIN_WEIGHT = 0.01


def _ndcg_at_k(ranked_labels: list[int], k: int) -> float:
    """Compute NDCG@k for a list of relevance labels in ranked order."""
    def _dcg(labels: list[int]) -> float:
        return sum(
            (2 ** rel - 1) / math.log2(rank + 2)
            for rank, rel in enumerate(labels[:k])
        )
    ideal = sorted(ranked_labels, reverse=True)
    idcg = _dcg(ideal)
    return _dcg(ranked_labels) / idcg if idcg > 0.0 else 0.0


def _evaluate_weights(
    profiles: dict[str, dict[str, float]],
    pairs_by_intent: dict[str, list[dict]],
) -> float:
    """Score profiles against labeled pairs; return mean NDCG@3 across queries."""
    from neural_search.retrieval.usefulness_scorer import (
        DatasetContext, UsefulnessIntent, score_usefulness,
    )

    total_ndcg, n_queries = 0.0, 0

    for intent_str, pairs in pairs_by_intent.items():
        query_candidates: dict[str, list[dict]] = {}
        for p in pairs:
            query_candidates.setdefault(p["query_id"], []).append(p)

        try:
            intent = UsefulnessIntent(intent_str)
        except ValueError:
            continue

        for qid, cands in query_candidates.items():
            if len(cands) < 2:
                continue
            query_ctx = DatasetContext(dataset_id=f"__query__{qid}")
            scored = []
            for c in cands:
                cand_ctx = DatasetContext(dataset_id=c["candidate_id"])
                s = score_usefulness(query_ctx, cand_ctx, intent)
                scored.append((s.total_score, LABEL_TO_INT.get(c["usefulness_label"], 0)))
            scored.sort(key=lambda x: x[0], reverse=True)
            ranked_labels = [lab for _, lab in scored]
            total_ndcg += _ndcg_at_k(ranked_labels, K)
            n_queries += 1

    return total_ndcg / n_queries if n_queries > 0 else 0.0


def _normalize_profile(profile: dict[str, float]) -> dict[str, float]:
    """Normalize so weights sum to 1.0."""
    total = sum(max(v, MIN_WEIGHT) for v in profile.values())
    return {k: max(v, MIN_WEIGHT) / total for k, v in profile.items()}


def load_pairs_by_intent() -> dict[str, list[dict]]:
    by_intent: dict[str, list[dict]] = {}
    for line in Path(SEED_FILE).read_text().splitlines():
        if not line.strip():
            continue
        p = json.loads(line)
        intent = p.get("intent", "")
        if intent:
            by_intent.setdefault(intent, []).append(p)
    return by_intent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optimize usefulness dimension weights")
    parser.add_argument("--n-trials", type=int, default=3, help="Optimization passes (default: 3)")
    parser.add_argument("--out", default="reports/optimized_weights_v11.json",
                        help="Output path for optimized weights JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print current weights without optimizing")
    args = parser.parse_args(argv)

    from neural_search.retrieval.usefulness_scorer import INTENT_WEIGHT_PROFILES

    # Convert enum-keyed dict to string-keyed for JSON compatibility
    str_profiles = {k.value: dict(v) for k, v in INTENT_WEIGHT_PROFILES.items()}

    if args.dry_run:
        print("DRY RUN — current INTENT_WEIGHT_PROFILES:")
        print(json.dumps(str_profiles, indent=2))
        return 0

    pairs_by_intent = load_pairs_by_intent()
    print(f"Loaded pairs: { {k: len(v) for k, v in pairs_by_intent.items()} }")

    profiles = copy.deepcopy(str_profiles)
    baseline = _evaluate_weights(profiles, pairs_by_intent)
    print(f"Baseline NDCG@{K}: {baseline:.4f}")

    for trial in range(args.n_trials):
        improved = 0
        for intent, dims in profiles.items():
            if intent not in pairs_by_intent or len(pairs_by_intent[intent]) < 4:
                continue
            for dim in list(dims.keys()):
                for delta in [+STEP, -STEP]:
                    candidate = copy.deepcopy(profiles)
                    candidate[intent][dim] = max(MIN_WEIGHT, candidate[intent][dim] + delta)
                    candidate[intent] = _normalize_profile(candidate[intent])
                    score = _evaluate_weights(candidate, pairs_by_intent)
                    if score > baseline + 1e-6:
                        profiles = candidate
                        baseline = score
                        improved += 1
        print(f"Trial {trial + 1}/{args.n_trials}: NDCG@{K} = {baseline:.4f}  ({improved} improvements)")

    for intent in profiles:
        profiles[intent] = _normalize_profile(profiles[intent])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(profiles, indent=2))
    print(f"\nOptimized weights written to: {out_path}")
    print(f"Final NDCG@{K}: {baseline:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
