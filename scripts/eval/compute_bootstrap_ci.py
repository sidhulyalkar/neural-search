"""Compute bootstrap confidence intervals across multiple retrieval system run files.

Reads JSONL run files and qrels, then reports per-system NDCG@10, MRR, P@5
with 95% bootstrap CIs and pairwise significance tests.

Usage
-----
    python scripts/eval/compute_bootstrap_ci.py \
        --qrels artifacts/qrels_silver.jsonl \
        --runs reports/eval/runs/bm25.jsonl reports/eval/runs/bge.jsonl \
        --out reports/eval/bootstrap_ci_report.json \
        --n-bootstrap 2000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

N_BOOTSTRAP = 2000
CI_LEVEL = 0.95


# ── Metric functions ──────────────────────────────────────────────────────────

def _dcg(rels: list[int], k: int) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels[:k]))


def ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    rels = [qrels.get(did, 0) for did in ranked_ids[:k]]
    ideal = sorted(qrels.values(), reverse=True)
    idcg = _dcg(ideal, k)
    return _dcg(rels, k) / idcg if idcg > 0 else 0.0


def mrr(ranked_ids: list[str], qrels: dict[str, int]) -> float:
    for i, did in enumerate(ranked_ids):
        if qrels.get(did, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int = 5) -> float:
    rel = sum(1 for did in ranked_ids[:k] if qrels.get(did, 0) > 0)
    return rel / k


def recall_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int = 20) -> float:
    n_rel = sum(1 for v in qrels.values() if v > 0)
    if n_rel == 0:
        return 0.0
    return sum(1 for did in ranked_ids[:k] if qrels.get(did, 0) > 0) / n_rel


def compute_per_query(run: dict[str, list[str]], qrels_by_query: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for qid, ranked_ids in run.items():
        q_qrels = qrels_by_query.get(qid, {})
        if not q_qrels:
            continue
        results[qid] = {
            "ndcg@10": ndcg_at_k(ranked_ids, q_qrels, 10),
            "ndcg@5": ndcg_at_k(ranked_ids, q_qrels, 5),
            "mrr": mrr(ranked_ids, q_qrels),
            "p@5": precision_at_k(ranked_ids, q_qrels, 5),
            "recall@20": recall_at_k(ranked_ids, q_qrels, 20),
        }
    return results


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def bootstrap_ci(
    per_query: dict[str, dict[str, float]],
    metric: str,
    n_bootstrap: int = N_BOOTSTRAP,
    ci_level: float = CI_LEVEL,
) -> dict[str, float]:
    values = [v[metric] for v in per_query.values() if metric in v]
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0}
    mean = sum(values) / len(values)
    rng = random.Random(42)
    boot_means = sorted(
        sum(rng.choices(values, k=len(values))) / len(values)
        for _ in range(n_bootstrap)
    )
    alpha = (1 - ci_level) / 2
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[int((1 - alpha) * n_bootstrap)]
    variance = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    return {
        "mean": round(mean, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "std": round(variance ** 0.5, 4),
        "n_queries": len(values),
    }


def wilcoxon_sign_test(
    a_per_query: dict[str, dict[str, float]],
    b_per_query: dict[str, dict[str, float]],
    metric: str,
) -> dict[str, Any]:
    shared = set(a_per_query) & set(b_per_query)
    diffs = [
        a_per_query[q][metric] - b_per_query[q][metric]
        for q in shared
        if metric in a_per_query[q] and metric in b_per_query[q]
    ]
    if not diffs:
        return {"n": 0, "a_wins": 0, "b_wins": 0, "ties": 0, "p_value_approx": 1.0}
    a_wins = sum(1 for d in diffs if d > 0)
    b_wins = sum(1 for d in diffs if d < 0)
    ties = sum(1 for d in diffs if d == 0)
    n_effective = len(diffs) - ties
    if n_effective == 0:
        return {"n": len(diffs), "a_wins": 0, "b_wins": 0, "ties": ties, "p_value_approx": 1.0}
    # Approximate sign test p-value: binomial(n_effective, 0.5) two-tailed
    # Use normal approximation for n > 10
    k = max(a_wins, b_wins)
    p_approx = 2 * (0.5 ** n_effective) if n_effective <= 20 else (
        2 * (1 - _norm_cdf((k - 0.5 * n_effective) / (0.5 * n_effective ** 0.5)))
    )
    return {
        "n": len(diffs),
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "p_value_approx": round(min(p_approx, 1.0), 4),
        "significant_at_05": p_approx < 0.05,
    }


def _norm_cdf(z: float) -> float:
    import math
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Load JSONL qrels into {query_id: {dataset_id: relevance}}."""
    qrels: dict[str, dict[str, int]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("query_id", rec.get("qid", "")))
            did = str(rec.get("dataset_id") or rec.get("record_id") or rec.get("doc_id") or "")
            rel = int(rec.get("relevance", rec.get("label", 0)))
            if qid and did:
                qrels.setdefault(qid, {})[did] = rel
    return qrels


def load_run(path: Path) -> dict[str, list[str]]:
    """Load JSONL run into {query_id: [ranked dataset_ids]}."""
    run: dict[str, list[tuple[int, str]]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("query_id", rec.get("qid", "")))
            did = str(rec.get("record_id") or rec.get("dataset_id") or rec.get("doc_id") or "")
            rank = int(rec.get("rank", 999))
            if qid and did:
                run.setdefault(qid, []).append((rank, did))
    return {qid: [did for _, did in sorted(pairs)] for qid, pairs in run.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, required=True, help="JSONL qrels file")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="JSONL run files")
    parser.add_argument("--out", type=Path, default=Path("reports/eval/bootstrap_ci_report.json"))
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--allow-silver", action="store_true")
    args = parser.parse_args(argv)

    if not args.qrels.exists():
        print(f"Qrels not found: {args.qrels}", file=sys.stderr)
        return 1

    qrels_by_query = load_qrels(args.qrels)
    n_labeled = sum(len(v) for v in qrels_by_query.values())
    print(f"Loaded qrels: {len(qrels_by_query)} queries, {n_labeled} labeled pairs")

    METRICS = ["ndcg@10", "ndcg@5", "mrr", "p@5", "recall@20"]
    systems: dict[str, dict[str, Any]] = {}

    for run_path in args.runs:
        if not run_path.exists():
            print(f"Run file not found, skipping: {run_path}", file=sys.stderr)
            continue
        name = run_path.stem
        run = load_run(run_path)
        per_query = compute_per_query(run, qrels_by_query)
        cis = {m: bootstrap_ci(per_query, m, args.n_bootstrap) for m in METRICS}
        systems[name] = {"per_query": per_query, "ci": cis}
        print(f"\n{name} ({len(per_query)} queries evaluated):")
        for m in METRICS:
            ci = cis[m]
            print(f"  {m:12s}: {ci['mean']:.4f}  95% CI [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")

    # Pairwise comparisons
    system_names = list(systems.keys())
    comparisons = []
    for i in range(len(system_names)):
        for j in range(i + 1, len(system_names)):
            a, b = system_names[i], system_names[j]
            for m in ["ndcg@10", "mrr"]:
                sig = wilcoxon_sign_test(systems[a]["per_query"], systems[b]["per_query"], m)
                sig["metric"] = m
                sig["system_a"] = a
                sig["system_b"] = b
                comparisons.append(sig)

    if comparisons:
        print("\nPairwise significance (sign test):")
        for c in comparisons:
            flag = " ✓" if c.get("significant_at_05") else ""
            print(f"  {c['system_a']} vs {c['system_b']} on {c['metric']}: "
                  f"p={c['p_value_approx']:.4f} (a_wins={c['a_wins']}, b_wins={c['b_wins']}){flag}")

    report = {
        "n_bootstrap": args.n_bootstrap,
        "ci_level": CI_LEVEL,
        "n_queries": len(qrels_by_query),
        "n_labeled_pairs": n_labeled,
        "systems": {
            name: {"ci": systems[name]["ci"]}
            for name in systems
        },
        "pairwise_significance": comparisons,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nReport → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
