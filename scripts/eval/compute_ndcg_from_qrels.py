#!/usr/bin/env python3
"""Compute NDCG@10, MRR, and Recall@50 from LLM-generated qrels.

Reads:
  data/qrels/qrels.trec          — TREC qrels (from run_parallel_llm_qrels.py)
  reports/eval/runs/<rung>.jsonl — per-rung retrieval results

Writes:
  reports/eval/ndcg_report.json  — machine-readable metrics per rung
  reports/eval/ndcg_report.md    — human-readable table

Usage
-----
    PYTHONPATH=. python scripts/eval/compute_ndcg_from_qrels.py

    # Custom paths
    PYTHONPATH=. python scripts/eval/compute_ndcg_from_qrels.py \\
        --qrels data/qrels/qrels.trec \\
        --runs-dir reports/eval/runs \\
        --out reports/eval/ndcg_report.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.eval.docid import normalize_docid

DEFAULT_QRELS = Path("data/qrels/qrels.trec")
DEFAULT_RUNS_DIR = Path("reports/eval/runs")
DEFAULT_OUT = Path("reports/eval/ndcg_report.json")
DEFAULT_MD = Path("reports/eval/ndcg_report.md")

RUNG_ORDER = [
    "bm25",
    "bm25_structured",
    "dense_bge",
    "hybrid_rrf",
    "hybrid_graph",
    "typed_kg",
    "typed_kg_qualified",
    "full",
]

K_NDCG = 10
K_RECALL = 50


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Return {query_id: {dataset_id: grade}}."""
    qrels: dict[str, dict[str, int]] = {}
    if not path.exists():
        return qrels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        # Treat the final token as the grade and every middle token as the
        # dataset id, then normalize whitespace so this stays correct for both
        # freshly-exported (single-token) and legacy (space-containing) qrels.
        qid, did, grade = parts[0], normalize_docid(" ".join(parts[2:-1])), parts[-1]
        qrels.setdefault(qid, {})[did] = int(grade)
    return qrels


def _load_run(path: Path) -> dict[str, list[tuple[int, str]]]:
    """Return {query_id: [(rank, record_id), ...]} sorted by rank."""
    run: dict[str, list[tuple[int, str]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = str(rec["query_id"])
        rank = int(rec["rank"])
        rid = normalize_docid(str(rec.get("record_id") or rec.get("dataset_id") or rec.get("doc_id")))
        run.setdefault(qid, []).append((rank, rid))
    for qid in run:
        run[qid].sort()
    return run


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _dcg(gains: list[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))


def _ideal_dcg(qrels: dict[str, int], k: int) -> float:
    sorted_grades = sorted(qrels.values(), reverse=True)
    return _dcg([float(g) for g in sorted_grades], k)


def ndcg_at_k(
    query_qrels: dict[str, int],
    ranked_docs: list[tuple[int, str]],
    k: int,
) -> float:
    idcg = _ideal_dcg(query_qrels, k)
    if idcg == 0:
        return 0.0
    gains = [float(query_qrels.get(did, 0)) for _, did in ranked_docs[:k]]
    return _dcg(gains, k) / idcg


def mrr(
    query_qrels: dict[str, int],
    ranked_docs: list[tuple[int, str]],
) -> float:
    for i, (_, did) in enumerate(ranked_docs):
        if query_qrels.get(did, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(
    query_qrels: dict[str, int],
    ranked_docs: list[tuple[int, str]],
    k: int,
) -> float:
    relevant = {did for did, g in query_qrels.items() if g > 0}
    if not relevant:
        return 0.0
    retrieved = {did for _, did in ranked_docs[:k]}
    return len(relevant & retrieved) / len(relevant)


# ---------------------------------------------------------------------------
# Per-rung evaluation
# ---------------------------------------------------------------------------

def evaluate_rung(
    rung_path: Path,
    qrels: dict[str, dict[str, int]],
) -> dict[str, float]:
    run = _load_run(rung_path)

    # Only evaluate queries that have at least one qrel judgment
    judged_queries = set(qrels.keys()) & set(run.keys())
    if not judged_queries:
        return {"ndcg@10": 0.0, "mrr": 0.0, "recall@50": 0.0, "judged_queries": 0}

    ndcg_scores, mrr_scores, recall_scores = [], [], []
    for qid in judged_queries:
        q_qrels = qrels[qid]
        ranked = run[qid]
        ndcg_scores.append(ndcg_at_k(q_qrels, ranked, K_NDCG))
        mrr_scores.append(mrr(q_qrels, ranked))
        recall_scores.append(recall_at_k(q_qrels, ranked, K_RECALL))

    return {
        "ndcg@10": sum(ndcg_scores) / len(ndcg_scores),
        "mrr": sum(mrr_scores) / len(mrr_scores),
        "recall@50": sum(recall_scores) / len(recall_scores),
        "judged_queries": len(judged_queries),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute NDCG from LLM qrels.")
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    print(f"\nLoading qrels from {args.qrels} ...")
    qrels = _load_qrels(args.qrels)
    total_judgments = sum(len(v) for v in qrels.values())
    print(f"  {len(qrels)} queries with judgments ({total_judgments} total pairs)")

    if not qrels:
        print("\nNo qrels found -- run run_parallel_llm_qrels.py first.")
        return 1

    # Relevance distribution
    grade_counts: dict[int, int] = {}
    for q_qrels in qrels.values():
        for g in q_qrels.values():
            grade_counts[g] = grade_counts.get(g, 0) + 1
    print("  Relevance distribution:", {k: grade_counts.get(k, 0) for k in range(4)})

    results: dict[str, dict[str, float]] = {}
    print(f"\nEvaluating rungs from {args.runs_dir} ...")

    for rung in RUNG_ORDER:
        rung_path = args.runs_dir / f"{rung}.jsonl"
        if not rung_path.exists():
            print(f"  {rung:20} -- file not found, skipping")
            continue
        metrics = evaluate_rung(rung_path, qrels)
        results[rung] = metrics
        print(
            f"  {rung:20} | NDCG@10={metrics['ndcg@10']:.4f} "
            f"| MRR={metrics['mrr']:.4f} "
            f"| R@50={metrics['recall@50']:.4f} "
            f"| {metrics['judged_queries']} queries"
        )

    # Write JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nMetrics written -> {args.out}")

    # Write Markdown table
    md_lines = [
        "# Ablation Ladder — NDCG Report (LLM Qrels)\n",
        f"**Qrels:** {total_judgments} pairs across {len(qrels)} queries\n",
        "",
        "| Rung | Queries | NDCG@10 | MRR | Recall@50 |",
        "|------|---------|---------|-----|-----------|",
    ]
    for rung in RUNG_ORDER:
        if rung not in results:
            continue
        m = results[rung]
        md_lines.append(
            f"| {rung} | {m['judged_queries']} "
            f"| {m['ndcg@10']:.4f} "
            f"| {m['mrr']:.4f} "
            f"| {m['recall@50']:.4f} |"
        )
    md_lines.append("")
    args.md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Markdown written -> {args.md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
