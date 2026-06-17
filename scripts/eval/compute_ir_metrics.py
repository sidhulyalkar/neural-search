#!/usr/bin/env python3
"""Compute IR metrics from qrels and run files.

Supports:
  NDCG@10, NDCG@20, MRR, Precision@10, Recall@50,
  hard_negative_violation_rate, source_skew_at_10,
  bootstrap confidence intervals (95% CI, 1000 samples).

Usage (gold qrels — paper-credible):
    python scripts/eval/compute_ir_metrics.py \
        --qrels artifacts/qrels.jsonl \
        --run reports/eval/runs/bm25.jsonl \
        --out reports/eval/eval_report.json

Usage (silver qrels — development diagnostic ONLY):
    python scripts/eval/compute_ir_metrics.py \
        --qrels artifacts/qrels_silver.jsonl \
        --run reports/eval/runs/bm25.jsonl \
        --out reports/eval/eval_report_silver.json \
        --allow-silver

Silver labels are NOT expert-validated. Use --allow-silver to acknowledge this.
Do NOT include silver-label metrics in the whitepaper as final results.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

_SILVER_PATH_MARKERS = ("silver", "qrels_silver")
_NEURO_JUDGE_PATH_MARKERS = ("neuro_qrels", "neuro_judge", "neuro-consensus")
_SILVER_WATERMARK = (
    "SILVER LABEL DIAGNOSTIC — NOT EXPERT VALIDATION. "
    "Do not report these metrics as final scientific results."
)
_NEURO_JUDGE_WATERMARK = (
    "NEURO-JUDGE DIAGNOSTIC — RAG-GROUNDED LLM LABELS, NOT HUMAN GOLD. "
    "Use for development triage and report separately from expert validation."
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    if not path.exists():
        return qrels
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("record_id") or row.get("dataset_id")
            label = row.get("label")
            if label is None or isinstance(label, str):
                label = row.get("relevance")
            if record_id is None or label is None:
                continue
            qrels[str(row["query_id"])][str(record_id)] = int(label)
    return dict(qrels)


def load_run(path: Path) -> dict[str, list[tuple[str, float]]]:
    """Load run file → {query_id: [(record_id, score), ...]} sorted by rank."""
    runs: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if not path.exists():
        return runs
    rows_by_query: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["query_id"])
            rid = str(row["record_id"])
            rank = int(row.get("rank", 10 ** 9))
            score = float(row.get("score", 0.0))
            rows_by_query[qid].append((rank, rid, score))
    for qid, rows in rows_by_query.items():
        rows.sort(key=lambda x: x[0])
        runs[qid] = [(rid, score) for _, rid, score in rows]
    return dict(runs)


def ranked_ids(run_entry: list[tuple[str, float]]) -> list[str]:
    return [rid for rid, _ in run_entry]


def run_scores(run_entry: list[tuple[str, float]]) -> dict[str, float]:
    return dict(run_entry)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def dcg(labels: list[int]) -> float:
    return float(sum((2 ** label - 1) / math.log2(idx + 2) for idx, label in enumerate(labels)))


def ndcg_at_k(qrel: dict[str, int], ranked: list[str], k: int) -> float:
    gains = [qrel.get(rid, 0) for rid in ranked[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg else 0.0


def mrr(qrel: dict[str, int], ranked: list[str], relevant_threshold: int = 2) -> float:
    for idx, rid in enumerate(ranked, start=1):
        if qrel.get(rid, 0) >= relevant_threshold:
            return 1.0 / idx
    return 0.0


def precision_at_k(qrel: dict[str, int], ranked: list[str], k: int, relevant_threshold: int = 2) -> float:
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    return sum(1 for rid in top_k if qrel.get(rid, 0) >= relevant_threshold) / len(top_k)


def recall_at_k(qrel: dict[str, int], ranked: list[str], k: int, relevant_threshold: int = 2) -> float:
    relevant = {rid for rid, label in qrel.items() if label >= relevant_threshold}
    if not relevant:
        return 0.0
    returned = set(ranked[:k])
    return len(relevant & returned) / len(relevant)


def hard_negative_violation_rate(
    qrel: dict[str, int],
    ranked: list[str],
    hard_negative_threshold: int = 0,
    relevant_threshold: int = 2,
) -> float:
    """Fraction of hard negatives ranked above any relevant document.

    A hard negative is a record with label == hard_negative_threshold (default 0).
    A violation occurs when a hard negative appears at a rank strictly above
    the rank of the first relevant result (label >= relevant_threshold).
    """
    hard_negatives = {rid for rid, label in qrel.items() if label <= hard_negative_threshold}
    relevant = {rid for rid, label in qrel.items() if label >= relevant_threshold}
    if not hard_negatives or not relevant:
        return 0.0

    first_relevant_rank = None
    for i, rid in enumerate(ranked):
        if rid in relevant:
            first_relevant_rank = i
            break

    if first_relevant_rank is None:
        return 0.0

    violations = sum(
        1 for i, rid in enumerate(ranked[:first_relevant_rank])
        if rid in hard_negatives
    )
    return violations / len(hard_negatives)


def source_skew_at_k(ranked: list[str], k: int) -> dict[str, float]:
    """Compute source distribution in top-k results.

    Returns dict mapping source → fraction of top-k results.
    Assumes record IDs are formatted as '{source}:{source_id}'.
    """
    top_k = ranked[:k]
    if not top_k:
        return {}
    source_counts: dict[str, int] = defaultdict(int)
    for rid in top_k:
        source = rid.split(":")[0] if ":" in rid else "unknown"
        source_counts[source] += 1
    total = len(top_k)
    return {source: count / total for source, count in sorted(source_counts.items(), key=lambda x: -x[1])}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Return (lower, upper) bootstrap CI for the mean of values."""
    if len(values) < 2:
        v = values[0] if values else 0.0
        return v, v
    rng = random.Random(seed)
    n = len(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [rng.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = int((1 + ci) / 2 * n_bootstrap)
    return boot_means[lo_idx], boot_means[min(hi_idx, n_bootstrap - 1)]


# ---------------------------------------------------------------------------
# Per-query and aggregate computation
# ---------------------------------------------------------------------------

def compute_query_metrics(
    qrel: dict[str, int],
    ranked: list[str],
    scores: dict[str, float],
) -> dict[str, Any]:
    return {
        "ndcg_at_10": ndcg_at_k(qrel, ranked, 10),
        "ndcg_at_20": ndcg_at_k(qrel, ranked, 20),
        "mrr": mrr(qrel, ranked),
        "precision_at_10": precision_at_k(qrel, ranked, 10),
        "recall_at_50": recall_at_k(qrel, ranked, 50),
        "hard_negative_violation_rate": hard_negative_violation_rate(qrel, ranked),
        "source_skew_at_10": source_skew_at_k(ranked, 10),
    }


def aggregate_metrics(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_keys = [
        "ndcg_at_10", "ndcg_at_20", "mrr",
        "precision_at_10", "recall_at_50", "hard_negative_violation_rate",
    ]
    agg: dict[str, Any] = {}
    for key in scalar_keys:
        vals = [row[key] for row in per_query if isinstance(row.get(key), (int, float))]
        m = mean(vals)
        lo, hi = bootstrap_ci(vals) if len(vals) >= 2 else (m, m)
        agg[key] = round(m, 4)
        agg[f"{key}_ci95"] = [round(lo, 4), round(hi, 4)]
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _is_silver_path(path: Path) -> bool:
    return any(marker in path.name.lower() for marker in _SILVER_PATH_MARKERS)


def _is_neuro_judge_path(path: Path) -> bool:
    path_text = str(path).lower()
    return any(marker in path_text for marker in _NEURO_JUDGE_PATH_MARKERS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute IR metrics from qrels and run files.")
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", dest="runs", default=None,
                        help="Run file(s); pass multiple --run flags for comparison")
    parser.add_argument("--out", type=Path, default=Path("reports/eval/eval_report.json"))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--allow-silver",
        action="store_true",
        default=False,
        help="Required when using silver qrels. Acknowledges labels are not expert-validated.",
    )
    parser.add_argument(
        "--qrels-tier",
        choices=["gold", "silver", "bronze"],
        default=None,
        help="Tier label for this qrels file (used for warnings and report tagging).",
    )
    args = parser.parse_args(argv)

    if not args.runs:
        parser.error("At least one --run file is required.")

    if args.qrels_tier and args.qrels_tier != "gold":
        sys.stderr.write(
            f"WARNING: Using {args.qrels_tier.upper()} qrels. "
            f"Results from {args.qrels_tier} labels should NOT be cited as "
            f"scientific validation. Use gold qrels for whitepaper claims.\n"
        )

    # Silver/neuro-judge guard — require explicit acknowledgement
    if (_is_silver_path(args.qrels) or _is_neuro_judge_path(args.qrels)) and not args.allow_silver:
        label_kind = "neuro-judge" if _is_neuro_judge_path(args.qrels) else "silver"
        print(
            f"ERROR: qrels path appears to be {label_kind} labels.\n"
            "These labels are machine-generated and NOT expert-validated.\n"
            "Add --allow-silver to proceed. Do NOT report these metrics in the whitepaper.",
            file=sys.stderr,
        )
        return 2

    qrels = load_qrels(args.qrels)
    if not qrels:
        early_report: dict[str, Any] = {
            "status": "Pending benchmark artifact",
            "note": "No qrels loaded. Annotate candidates with annotate_candidates.py first.",
            "qrels_path": str(args.qrels),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(early_report, indent=2), encoding="utf-8")
        print(json.dumps({"status": early_report["status"]}, indent=2))
        return 0

    all_run_reports: list[dict[str, Any]] = []

    for run_path in args.runs:
        run = load_run(run_path)
        query_ids = sorted(set(qrels) | set(run))

        per_query = []
        for qid in query_ids:
            qrel = qrels.get(qid, {})
            run_entry = run.get(qid, [])
            ranked = ranked_ids(run_entry)
            scores = run_scores(run_entry)
            q_metrics = compute_query_metrics(qrel, ranked, scores)
            per_query.append({"query_id": qid, **q_metrics})

        agg = aggregate_metrics(per_query)

        all_run_reports.append({
            "run_file": str(run_path),
            "variant": run_path.stem,
            "status": "computed" if query_ids else "Pending benchmark artifact",
            "query_count": len(query_ids),
            "qrels_query_count": len(qrels),
            "metrics": agg,
            "per_query": per_query,
        })

    # Single-run mode: compact output
    report: dict[str, Any]
    if len(all_run_reports) == 1:
        report = all_run_reports[0]
    else:
        report = {
            "status": "computed",
            "runs": all_run_reports,
        }

    report["qrels_tier"] = args.qrels_tier or "unknown"
    report["qrels_path"] = str(args.qrels)

    # Watermark non-human-label reports
    if _is_silver_path(args.qrels):
        report["silver_label_warning"] = _SILVER_WATERMARK
    if _is_neuro_judge_path(args.qrels):
        report["neuro_judge_warning"] = _NEURO_JUDGE_WATERMARK

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Summary print
    for r in all_run_reports:
        m = r.get("metrics", {})
        print(json.dumps({
            "variant": r.get("variant"),
            "ndcg_at_10": m.get("ndcg_at_10"),
            "mrr": m.get("mrr"),
            "recall_at_50": m.get("recall_at_50"),
            "hard_neg_violation_rate": m.get("hard_negative_violation_rate"),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
