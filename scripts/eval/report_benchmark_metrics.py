#!/usr/bin/env python3
"""Compute and report benchmark metrics from adjudicated qrels.

Reads adjudicated_qrels.jsonl, computes NDCG@10, MRR, Precision@5/10,
Recall@10, and hard-negative violation rate per query and per intent.
Writes a Markdown report to reports/eval/sprint1_benchmark.md.

Usage:
    python scripts/eval/report_benchmark_metrics.py
    python scripts/eval/report_benchmark_metrics.py --qrels artifacts/field_state/adjudicated_qrels.jsonl
    python scripts/eval/report_benchmark_metrics.py --quiet
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

QRELS_PATH = ROOT / "artifacts" / "field_state" / "adjudicated_qrels.jsonl"
CANDIDATES_PATH = ROOT / "artifacts" / "field_state" / "qrels_candidates_full.jsonl"
FALLBACK_CANDIDATES = ROOT / "artifacts" / "field_state" / "qrels_candidates.jsonl"
QUERIES_PATH = ROOT / "artifacts" / "benchmark_queries.jsonl"
OUTPUT_PATH = ROOT / "reports" / "eval" / "sprint1_benchmark.md"

MIN_PAIRS_FOR_STABLE_ESTIMATE = 30
RELEVANCE_THRESHOLD = 2  # score ≥ 2 counts as "relevant" for binary metrics


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def dcg(scores: list[int], k: int = 10) -> float:
    return sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def ndcg(scores: list[int], k: int = 10) -> float:
    ideal = sorted(scores, reverse=True)
    idcg = dcg(ideal, k)
    return dcg(scores, k) / idcg if idcg > 0 else 0.0


def mrr(scores: list[int], threshold: int = RELEVANCE_THRESHOLD) -> float:
    for i, s in enumerate(scores):
        if s >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(scores: list[int], k: int, threshold: int = RELEVANCE_THRESHOLD) -> float:
    top = scores[:k]
    return sum(1 for s in top if s >= threshold) / k if top else 0.0


def recall_at_k(scores: list[int], all_relevant: int, k: int, threshold: int = RELEVANCE_THRESHOLD) -> float:
    top = scores[:k]
    retrieved = sum(1 for s in top if s >= threshold)
    return retrieved / all_relevant if all_relevant > 0 else 0.0


def compute_metrics(scores: list[int]) -> dict[str, float]:
    n_rel = sum(1 for s in scores if s >= RELEVANCE_THRESHOLD)
    return {
        "ndcg@10": ndcg(scores, 10),
        "mrr": mrr(scores),
        "p@5": precision_at_k(scores, 5),
        "p@10": precision_at_k(scores, 10),
        "recall@10": recall_at_k(scores, n_rel, 10),
        "n_labeled": len(scores),
        "n_relevant": n_rel,
    }


def macro_average(per_query_metrics: list[dict[str, float]]) -> dict[str, float]:
    if not per_query_metrics:
        return {}
    keys = [k for k in per_query_metrics[0] if isinstance(per_query_metrics[0][k], float)]
    result: dict[str, float] = {}
    for key in keys:
        vals = [m[key] for m in per_query_metrics if key in m]
        result[key] = sum(vals) / len(vals) if vals else 0.0
    result["n_queries"] = float(len(per_query_metrics))
    return result


def _fmt(v: float, pct: bool = False) -> str:
    if pct:
        return f"{v * 100:.1f}%"
    return f"{v:.4f}"


def build_report(
    qrels: list[dict],
    query_map: dict[str, dict],
    candidate_map: dict[str, list[str]],
) -> str:
    n_pairs = len(qrels)
    preliminary = n_pairs < MIN_PAIRS_FOR_STABLE_ESTIMATE
    generated = datetime.now(timezone.utc).isoformat()

    # Group by query, maintain retrieval order (rank from candidates)
    rank_lookup: dict[str, int] = {}
    for cid, cands in candidate_map.items():
        for i, dataset_id in enumerate(cands):
            rank_lookup[f"{cid}::{dataset_id}"] = i + 1

    by_query: dict[str, list[tuple[int, int, bool]]] = defaultdict(list)  # (rank, score, hn)
    for label in qrels:
        q = label["query_id"]
        ds = label["dataset_id"]
        key = f"{q}::{ds}"
        rank = rank_lookup.get(key, 999)
        hn = label.get("hard_negative_violation", False)
        by_query[q].append((rank, label["relevance"], hn))

    # Sort each query by rank
    for q in by_query:
        by_query[q].sort(key=lambda x: x[0])

    per_query_metrics: list[dict] = []
    per_intent_groups: dict[str, list[dict]] = defaultdict(list)
    hn_violations = 0
    hn_total = 0

    for q_id, items in sorted(by_query.items()):
        scores = [s for _, s, _ in items]
        hn_count = sum(1 for _, _, hn in items if hn)
        hn_violations += hn_count
        hn_total += len(items)

        m = compute_metrics(scores)
        m["query_id"] = q_id
        m["hn_violation_rate"] = hn_count / len(items) if items else 0.0

        query_meta = query_map.get(q_id, {})
        m["intent"] = query_meta.get("intent", "unknown")
        m["query_text"] = (query_meta.get("query_text") or query_meta.get("query", ""))[:80]
        per_query_metrics.append(m)
        per_intent_groups[m["intent"]].append(m)

    macro = macro_average(per_query_metrics)
    hn_rate = hn_violations / hn_total if hn_total > 0 else 0.0

    lines: list[str] = []
    lines.append("# Neural Search Sprint 1 Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {generated}")
    lines.append(f"Qrels: `{QRELS_PATH.relative_to(ROOT)}`")
    lines.append(f"Annotated pairs: **{n_pairs}**")
    if preliminary:
        lines.append("")
        lines.append(
            f"> **PRELIMINARY** — Only {n_pairs} pairs labeled. "
            f"Estimates are unstable below {MIN_PAIRS_FOR_STABLE_ESTIMATE} pairs. "
            "Do not cite these numbers in a paper."
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Overall Metrics (macro-average over queries)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| NDCG@10 | {_fmt(macro.get('ndcg@10', 0))} |")
    lines.append(f"| MRR | {_fmt(macro.get('mrr', 0))} |")
    lines.append(f"| Precision@5 | {_fmt(macro.get('p@5', 0), pct=True)} |")
    lines.append(f"| Precision@10 | {_fmt(macro.get('p@10', 0), pct=True)} |")
    lines.append(f"| Recall@10 | {_fmt(macro.get('recall@10', 0), pct=True)} |")
    lines.append(f"| HN Violation Rate | {_fmt(hn_rate, pct=True)} |")
    lines.append(f"| Queries evaluated | {int(macro.get('n_queries', 0))} |")
    lines.append(f"| Total labeled pairs | {n_pairs} |")
    lines.append("")

    if per_intent_groups:
        lines.append("## By Intent")
        lines.append("")
        lines.append("| Intent | Queries | NDCG@10 | MRR | P@5 | HN Viol |")
        lines.append("|--------|---------|---------|-----|-----|---------|")
        for intent, mlist in sorted(per_intent_groups.items()):
            avg = macro_average(mlist)
            hn_avg = sum(m.get("hn_violation_rate", 0) for m in mlist) / len(mlist)
            lines.append(
                f"| {intent} | {len(mlist)} | {_fmt(avg.get('ndcg@10', 0))} | "
                f"{_fmt(avg.get('mrr', 0))} | {_fmt(avg.get('p@5', 0), pct=True)} | "
                f"{_fmt(hn_avg, pct=True)} |"
            )
        lines.append("")

    if per_query_metrics:
        lines.append("## Per-Query Results")
        lines.append("")
        lines.append("| Query | Intent | N | NDCG@10 | MRR | P@5 | HN Viol | Query Text |")
        lines.append("|-------|--------|---|---------|-----|-----|---------|------------|")
        for m in per_query_metrics:
            lines.append(
                f"| {m['query_id']} | {m.get('intent', '?')} | {m['n_labeled']} | "
                f"{_fmt(m['ndcg@10'])} | {_fmt(m['mrr'])} | {_fmt(m['p@5'], pct=True)} | "
                f"{_fmt(m.get('hn_violation_rate', 0), pct=True)} | {m.get('query_text', '')} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Next Steps")
    lines.append("")
    if preliminary:
        remaining = MIN_PAIRS_FOR_STABLE_ESTIMATE - n_pairs
        lines.append(f"- Label {remaining} more pairs to reach the minimum stable estimate threshold.")
    lines.append("- Run dual annotation on 20% of pairs to compute inter-annotator agreement.")
    lines.append("- Run ablation comparison (BM25 vs concept_boost) using these qrels.")
    lines.append("- Expand to 50+ queries for broader coverage.")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("# Re-run this report")
    lines.append("python scripts/eval/report_benchmark_metrics.py")
    lines.append("")
    lines.append("# Add more labels")
    lines.append("python scripts/annotate_qrels_fast.py --resume")
    lines.append("```")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute benchmark metrics from adjudicated qrels.")
    parser.add_argument("--qrels", type=Path, default=QRELS_PATH)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    qrels = load_jsonl(args.qrels)
    if not qrels:
        print(f"No labels found in {args.qrels}. Run annotate_qrels_fast.py first.", file=sys.stderr)
        sys.exit(1)

    # Load query metadata
    queries = load_jsonl(args.queries)
    query_map: dict[str, dict] = {q["query_id"]: q for q in queries}

    # Load candidate pools to get retrieval rank order
    cands_path = args.candidates
    if cands_path is None:
        cands_path = CANDIDATES_PATH if CANDIDATES_PATH.exists() else FALLBACK_CANDIDATES
    candidates = load_jsonl(cands_path)

    candidate_map: dict[str, list[str]] = defaultdict(list)
    for c in sorted(candidates, key=lambda x: x.get("rank", 999)):
        candidate_map[c["query_id"]].append(c["dataset_id"])

    report = build_report(qrels, query_map, candidate_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)

    if not args.quiet:
        print(report)
        print(f"\nWrote to {args.output}")


if __name__ == "__main__":
    main()
