#!/usr/bin/env python3
"""Report retrieval metrics stratified by canonical query intent.

Reads canonical benchmark queries with ``intent`` labels, qrels, and run files,
then writes per-system/per-intent NDCG@10, MRR, and Recall@50.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.eval.compute_ndcg_from_qrels import (  # noqa: E402
    K_NDCG,
    K_RECALL,
    RUNG_ORDER,
    _load_qrels,
    _load_run,
    mrr,
    ndcg_at_k,
    recall_at_k,
)

DEFAULT_QUERIES = Path("data/eval/benchmark_queries_canonical.yaml")
DEFAULT_QRELS = Path("data/qrels/qrels.canonical.trec")
DEFAULT_RUNS_DIR = Path("reports/eval/runs")
DEFAULT_OUT = Path("reports/eval/intent_stratification_report.json")
DEFAULT_MD = Path("reports/eval/intent_stratification_report.md")


def load_query_intents(path: Path) -> dict[str, str]:
    """Return {query_id: intent} from canonical query YAML."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    queries = payload.get("benchmark_queries", [])
    intents: dict[str, str] = {}
    for query in queries:
        qid = str(query.get("id", ""))
        if not qid:
            continue
        intents[qid] = str(query.get("intent") or "unknown")
    return intents


def evaluate_by_intent(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[int, str]]],
    query_intents: dict[str, str],
) -> dict[str, dict[str, float]]:
    """Return macro-averaged metrics per intent for one run."""
    grouped_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"ndcg@10": [], "mrr": [], "recall@50": []}
    )
    for qid in sorted(set(qrels) & set(run)):
        intent = query_intents.get(qid, "unknown")
        q_qrels = qrels[qid]
        ranked = run[qid]
        grouped_scores[intent]["ndcg@10"].append(ndcg_at_k(q_qrels, ranked, K_NDCG))
        grouped_scores[intent]["mrr"].append(mrr(q_qrels, ranked))
        grouped_scores[intent]["recall@50"].append(recall_at_k(q_qrels, ranked, K_RECALL))

    results: dict[str, dict[str, float]] = {}
    for intent, metrics in sorted(grouped_scores.items()):
        n_queries = len(metrics["ndcg@10"])
        if n_queries == 0:
            continue
        results[intent] = {
            "n_queries": n_queries,
            "ndcg@10": sum(metrics["ndcg@10"]) / n_queries,
            "mrr": sum(metrics["mrr"]) / n_queries,
            "recall@50": sum(metrics["recall@50"]) / n_queries,
        }
    return results


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Intent-Stratified Retrieval Report",
        "",
        f"**Qrels:** {report['n_labeled_pairs']} pairs across {report['n_queries']} queries",
        f"**Queries:** `{report['queries_path']}`",
        "",
        "| System | Intent | Queries | NDCG@10 | MRR | Recall@50 |",
        "|--------|--------|---------|---------|-----|-----------|",
    ]
    for system in RUNG_ORDER:
        intents = report["systems"].get(system, {})
        for intent, metrics in sorted(intents.items()):
            lines.append(
                f"| {system} | {intent} | {metrics['n_queries']} "
                f"| {metrics['ndcg@10']:.4f} "
                f"| {metrics['mrr']:.4f} "
                f"| {metrics['recall@50']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    if not args.queries.exists():
        print(f"Queries not found: {args.queries}", file=sys.stderr)
        return 1
    if not args.qrels.exists():
        print(f"Qrels not found: {args.qrels}", file=sys.stderr)
        return 1

    query_intents = load_query_intents(args.queries)
    qrels = _load_qrels(args.qrels)
    n_labeled = sum(len(v) for v in qrels.values())

    systems: dict[str, dict[str, dict[str, float]]] = {}
    for system in RUNG_ORDER:
        run_path = args.runs_dir / f"{system}.jsonl"
        if not run_path.exists() or run_path.stat().st_size == 0:
            continue
        systems[system] = evaluate_by_intent(qrels, _load_run(run_path), query_intents)

    report = {
        "queries_path": str(args.queries),
        "qrels_path": str(args.qrels),
        "runs_dir": str(args.runs_dir),
        "n_queries": len(qrels),
        "n_labeled_pairs": n_labeled,
        "systems": systems,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.md.write_text(build_markdown(report), encoding="utf-8")
    print(f"Intent stratification report -> {args.out}")
    print(f"Markdown -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
