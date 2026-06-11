#!/usr/bin/env python3
"""Compute per-retrieval-system metrics from adjudicated (gold) qrels.

Unlike report_benchmark_metrics.py (which uses a single pooled rank), this
script exploits the per-system rank information in the pooled candidates
to report NDCG@10 / MRR / P@5 / Recall@10 for each retrieval variant:
  bm25, dense_prf, hybrid_rrf, usefulness

Clearly watermarks results as PRELIMINARY when fewer than 100 labels exist.

Usage:
    python scripts/eval/report_gold_qrels_metrics.py
    python scripts/eval/report_gold_qrels_metrics.py --qrels artifacts/field_state/adjudicated_qrels.jsonl
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
CANDIDATES_PATH = ROOT / "artifacts" / "field_state" / "qrels_candidates_pooled.jsonl"
FALLBACK_CANDIDATES = ROOT / "artifacts" / "field_state" / "qrels_candidates_full.jsonl"
QUERIES_PATH = ROOT / "artifacts" / "benchmark_queries.jsonl"
OUTPUT_PATH = ROOT / "reports" / "eval" / "sprint1_gold_benchmark.md"

MIN_PAIRS_FOR_CREDIBLE_REPORT = 100
RELEVANCE_THRESHOLD = 2  # score ≥ 2 counts as relevant for binary metrics


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# IR metrics
# ---------------------------------------------------------------------------

def _dcg(scores: list[int], k: int = 10) -> float:
    return sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def _ndcg(scores: list[int], k: int = 10) -> float:
    ideal = sorted(scores, reverse=True)
    idcg = _dcg(ideal, k)
    return _dcg(scores, k) / idcg if idcg > 0 else 0.0


def _mrr(scores: list[int], threshold: int = RELEVANCE_THRESHOLD) -> float:
    for i, s in enumerate(scores):
        if s >= threshold:
            return 1.0 / (i + 1)
    return 0.0


def _p_at_k(scores: list[int], k: int, threshold: int = RELEVANCE_THRESHOLD) -> float:
    top = scores[:k]
    return sum(1 for s in top if s >= threshold) / k if top else 0.0


def _recall_at_k(scores: list[int], n_rel: int, k: int, threshold: int = RELEVANCE_THRESHOLD) -> float:
    retrieved = sum(1 for s in scores[:k] if s >= threshold)
    return retrieved / n_rel if n_rel > 0 else 0.0


def _compute_metrics(scores: list[int]) -> dict[str, float]:
    n_rel = sum(1 for s in scores if s >= RELEVANCE_THRESHOLD)
    return {
        "ndcg@10": _ndcg(scores, 10),
        "mrr": _mrr(scores),
        "p@5": _p_at_k(scores, 5),
        "recall@10": _recall_at_k(scores, n_rel, 10),
        "n_labeled": len(scores),
        "n_relevant": n_rel,
    }


def _macro_avg(metric_list: list[dict[str, float]]) -> dict[str, float]:
    if not metric_list:
        return {}
    keys = [k for k in metric_list[0] if isinstance(metric_list[0][k], float)]
    result = {k: sum(m.get(k, 0.0) for m in metric_list) / len(metric_list) for k in keys}
    result["n_queries"] = float(len(metric_list))
    return result


# ---------------------------------------------------------------------------
# Ranking reconstruction per system
# ---------------------------------------------------------------------------

def _build_system_rankings(
    candidates: list[dict],
    systems: list[str],
) -> dict[str, dict[str, dict[str, int]]]:
    """Return {system: {query_id: {dataset_id: rank}}} from pooled candidates."""
    rankings: dict[str, dict[str, dict[str, int]]] = {s: defaultdict(dict) for s in systems}

    for cand in candidates:
        qid = cand["query_id"]
        did = cand["dataset_id"]
        ranks_by_system = cand.get("ranks_by_system") or {}
        for sys_name, rank in ranks_by_system.items():
            if sys_name in rankings:
                rankings[sys_name][qid][did] = rank

    return rankings


def _scores_for_system(
    system_ranking: dict[str, dict[str, int]],  # {query_id: {dataset_id: rank}}
    qrels_by_query: dict[str, dict[str, int]],  # {query_id: {dataset_id: relevance}}
    query_id: str,
    max_rank: int = 10,
) -> list[int]:
    """Return relevance scores in rank order for a given system+query."""
    ranking = system_ranking.get(query_id, {})
    qrels = qrels_by_query.get(query_id, {})

    if not ranking:
        return []

    # Sort by rank
    ordered = sorted(ranking.items(), key=lambda x: x[1])
    scores = [qrels.get(did, 0) for did, _ in ordered[:max_rank]]
    return scores


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def _fmt(v: float, pct: bool = False) -> str:
    return f"{v * 100:.1f}%" if pct else f"{v:.4f}"


def build_report(
    qrels: list[dict],
    candidates: list[dict],
    query_map: dict[str, dict],
) -> str:
    n_pairs = len(qrels)
    preliminary = n_pairs < MIN_PAIRS_FOR_CREDIBLE_REPORT
    generated = datetime.now(timezone.utc).isoformat()

    # qrels by query/dataset
    qrels_by_query: dict[str, dict[str, int]] = defaultdict(dict)
    hn_by_query: dict[str, list[bool]] = defaultdict(list)
    for label in qrels:
        qid = label["query_id"]
        did = label["dataset_id"]
        qrels_by_query[qid][did] = label["relevance"]
        hn_by_query[qid].append(bool(label.get("hard_negative_violation", False)))

    all_systems = ["bm25", "dense_prf", "hybrid_rrf", "usefulness"]
    system_rankings = _build_system_rankings(candidates, all_systems)

    # Only include systems that have ranking data
    active_systems = [s for s in all_systems if any(system_rankings[s].values())]

    # Per-query, per-system metrics
    all_query_ids = sorted(qrels_by_query.keys())
    per_query: dict[str, dict[str, dict]] = {qid: {} for qid in all_query_ids}

    for qid in all_query_ids:
        for sys_name in active_systems:
            scores = _scores_for_system(system_rankings[sys_name], qrels_by_query, qid)
            per_query[qid][sys_name] = _compute_metrics(scores)

    # Overall HN violation rate
    hn_total = sum(len(v) for v in hn_by_query.values())
    hn_violated = sum(sum(1 for x in v if x) for v in hn_by_query.values())
    hn_rate = hn_violated / hn_total if hn_total > 0 else 0.0

    # Per-intent breakdown
    intent_groups: dict[str, list[str]] = defaultdict(list)
    for qid in all_query_ids:
        intent = (query_map.get(qid, {}).get("intent") or "unknown")
        intent_groups[intent].append(qid)

    lines: list[str] = []
    lines.append("# Neural Search Sprint 1 Gold Benchmark Report")
    lines.append("")
    lines.append(f"Generated: {generated}")
    lines.append(f"Qrels: `{QRELS_PATH.relative_to(ROOT)}`")
    lines.append(f"Candidates: `{CANDIDATES_PATH.name}`")
    lines.append(f"Annotated pairs: **{n_pairs}**")
    lines.append(f"Queries covered: **{len(all_query_ids)}**")
    lines.append("")

    if preliminary:
        lines.append("> [!WARNING]")
        lines.append(
            f"> **PRELIMINARY** — Only {n_pairs}/{MIN_PAIRS_FOR_CREDIBLE_REPORT} labels collected. "
            "Estimates are highly unstable. Do NOT cite these numbers."
        )
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Per-System Metrics (macro-avg over labeled queries)")
    lines.append("")
    lines.append("| System | NDCG@10 | MRR | P@5 | Recall@10 |")
    lines.append("|--------|---------|-----|-----|-----------|")

    for sys_name in active_systems:
        query_metrics = []
        for qid in all_query_ids:
            m = per_query[qid].get(sys_name, {})
            if m:
                query_metrics.append(m)
        avg = _macro_avg(query_metrics)
        lines.append(
            f"| {sys_name} | {_fmt(avg.get('ndcg@10', 0))} | {_fmt(avg.get('mrr', 0))} | "
            f"{_fmt(avg.get('p@5', 0), pct=True)} | {_fmt(avg.get('recall@10', 0), pct=True)} |"
        )

    lines.append("")
    lines.append(f"HN violation rate: **{_fmt(hn_rate, pct=True)}**  ({hn_violated}/{hn_total} pairs)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-intent breakdown (for the hybrid_rrf system, as it's the primary)
    primary_system = "hybrid_rrf" if "hybrid_rrf" in active_systems else (active_systems[0] if active_systems else None)
    if primary_system and intent_groups:
        lines.append(f"## By Intent (system: `{primary_system}`)")
        lines.append("")
        lines.append("| Intent | Queries | NDCG@10 | MRR | P@5 |")
        lines.append("|--------|---------|---------|-----|-----|")
        for intent, qids in sorted(intent_groups.items()):
            mlist = [per_query[qid].get(primary_system, {}) for qid in qids if per_query[qid].get(primary_system)]
            if not mlist:
                continue
            avg = _macro_avg(mlist)
            lines.append(
                f"| {intent} | {len(qids)} | {_fmt(avg.get('ndcg@10', 0))} | "
                f"{_fmt(avg.get('mrr', 0))} | {_fmt(avg.get('p@5', 0), pct=True)} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    # Per-query table
    lines.append("## Per-Query Results")
    lines.append("")
    cols = ["Query", "Intent", "N"] + [f"NDCG@10 ({s})" for s in active_systems]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")

    for qid in all_query_ids:
        intent = query_map.get(qid, {}).get("intent", "?")
        n_labeled = sum(len(qrels_by_query[qid]) for _ in [1])
        row = [qid, intent, str(n_labeled)]
        for sys_name in active_systems:
            m = per_query[qid].get(sys_name, {})
            row.append(_fmt(m.get("ndcg@10", 0.0)) if m else "—")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Confidence Notes")
    lines.append("")
    if preliminary:
        remaining = MIN_PAIRS_FOR_CREDIBLE_REPORT - n_pairs
        lines.append(
            f"- Need **{remaining} more** labeled pairs to reach the "
            f"{MIN_PAIRS_FOR_CREDIBLE_REPORT}-pair credibility threshold."
        )
        lines.append("- Current metrics have high variance — do not use for ablation conclusions.")
    else:
        lines.append(
            f"- {n_pairs} pairs labeled across {len(all_query_ids)} queries — "
            "results are approaching minimum credibility."
        )
    lines.append("- Dual annotation (20% of pairs) needed for inter-annotator agreement.")
    lines.append("- Per-query coverage is uneven — queries with N<5 labels are unreliable.")
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append("# Re-generate candidates")
    lines.append("python scripts/eval/build_pooled_qrels_candidates.py")
    lines.append("")
    lines.append("# Annotate (--limit 30 for a quick session)")
    lines.append("python scripts/annotate_qrels_fast.py --resume --limit 30")
    lines.append("")
    lines.append("# Re-run this report")
    lines.append("python scripts/eval/report_gold_qrels_metrics.py")
    lines.append("```")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute gold-qrels metrics by retrieval system.")
    parser.add_argument("--qrels", type=Path, default=QRELS_PATH)
    parser.add_argument("--candidates", type=Path, default=None)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    qrels = _load_jsonl(args.qrels)
    if not qrels:
        print(
            f"No labels in {args.qrels}. Run: python scripts/annotate_qrels_fast.py --resume",
            file=sys.stderr,
        )
        sys.exit(1)

    cands_path = args.candidates
    if cands_path is None:
        cands_path = CANDIDATES_PATH if CANDIDATES_PATH.exists() else FALLBACK_CANDIDATES
    candidates = _load_jsonl(cands_path)

    queries = _load_jsonl(args.queries)
    query_map = {q["query_id"]: q for q in queries}

    report = build_report(qrels, candidates, query_map)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write(report)

    if not args.quiet:
        n = len(qrels)
        status = "PRELIMINARY" if n < MIN_PAIRS_FOR_CREDIBLE_REPORT else "OK"
        print(f"[{status}] {n} labeled pairs — wrote report to {args.output}")


if __name__ == "__main__":
    main()
