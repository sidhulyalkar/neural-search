#!/usr/bin/env python3
"""Build a claim ledger from the current qrels-backed eval reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_NDCG = Path("reports/eval/ndcg_report.json")
DEFAULT_BOOTSTRAP = Path("reports/eval/bootstrap_ci_report.json")
DEFAULT_INTENT = Path("reports/eval/intent_stratification_report.json")
DEFAULT_RELIABILITY = Path("reports/eval/dual_judge_reliability.json")
DEFAULT_OUT = Path("reports/eval/eval_claim_ledger.json")
DEFAULT_MD = Path("reports/eval/eval_claim_ledger.md")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _metric_delta(metrics: dict[str, Any], system: str, baseline: str, metric: str) -> float | None:
    if system not in metrics or baseline not in metrics:
        return None
    return float(metrics[system].get(metric, 0.0)) - float(metrics[baseline].get(metric, 0.0))


def _significant(
    bootstrap: dict[str, Any],
    system_a: str,
    system_b: str,
    metric: str,
    winner: str,
) -> bool:
    for row in bootstrap.get("pairwise_significance", []):
        if row.get("metric") != metric:
            continue
        a, b = row.get("system_a"), row.get("system_b")
        if {a, b} != {system_a, system_b}:
            continue
        if not row.get("significant_at_05"):
            return False
        if winner == a:
            return int(row.get("a_wins", 0)) > int(row.get("b_wins", 0))
        if winner == b:
            return int(row.get("b_wins", 0)) > int(row.get("a_wins", 0))
    return False


def build_ledger(
    ndcg: dict[str, Any],
    bootstrap: dict[str, Any],
    intent: dict[str, Any],
    reliability: dict[str, Any],
) -> list[dict[str, Any]]:
    n_queries = max((int(row.get("judged_queries", 0)) for row in ndcg.values()), default=0)
    n_pairs = int(bootstrap.get("n_labeled_pairs", 0))
    hybrid_ndcg_delta = _metric_delta(ndcg, "hybrid_rrf", "bm25", "ndcg@10")
    hybrid_mrr_delta = _metric_delta(ndcg, "hybrid_rrf", "bm25", "mrr")
    hybrid_recall_delta = _metric_delta(ndcg, "hybrid_rrf", "bm25", "recall@50")

    return [
        {
            "claim_id": "claim_eval_canonical_qrels",
            "claim": "Canonical labeled qrels are available for qrels-backed retrieval metrics.",
            "evidence_level": "supported" if n_queries > 0 and n_pairs > 0 else "missing",
            "evidence": {
                "queries": n_queries,
                "labeled_pairs": n_pairs,
                "artifacts": ["data/qrels/qrels.canonical.trec", "data/qrels/qrels.canonical.jsonl"],
            },
        },
        {
            "claim_id": "claim_hybrid_rrf_beats_bm25",
            "claim": "Hybrid RRF improves over BM25 on the labeled qrels aggregate.",
            "evidence_level": (
                "partially_supported"
                if all(v is not None and v > 0 for v in [hybrid_ndcg_delta, hybrid_mrr_delta, hybrid_recall_delta])
                else "not_supported"
            ),
            "evidence": {
                "ndcg@10_delta": hybrid_ndcg_delta,
                "mrr_delta": hybrid_mrr_delta,
                "recall@50_delta": hybrid_recall_delta,
                "mrr_significant_vs_bm25": _significant(bootstrap, "bm25", "hybrid_rrf", "mrr", "hybrid_rrf"),
                "ndcg_significant_vs_bm25": _significant(bootstrap, "bm25", "hybrid_rrf", "ndcg@10", "hybrid_rrf"),
                "artifacts": ["reports/eval/ndcg_report.json", "reports/eval/bootstrap_ci_report.json"],
            },
            "caveat": "NDCG@10 is directionally higher but not significant versus BM25 by the current sign test.",
        },
        {
            "claim_id": "claim_hybrid_rrf_beats_dense_bge",
            "claim": "Hybrid RRF improves over dense BGE on the labeled qrels aggregate.",
            "evidence_level": (
                "supported"
                if _significant(bootstrap, "dense_bge", "hybrid_rrf", "ndcg@10", "hybrid_rrf")
                and _significant(bootstrap, "dense_bge", "hybrid_rrf", "mrr", "hybrid_rrf")
                else "partially_supported"
            ),
            "evidence": {
                "ndcg@10_delta": _metric_delta(ndcg, "hybrid_rrf", "dense_bge", "ndcg@10"),
                "mrr_delta": _metric_delta(ndcg, "hybrid_rrf", "dense_bge", "mrr"),
                "recall@50_delta": _metric_delta(ndcg, "hybrid_rrf", "dense_bge", "recall@50"),
                "ndcg_significant_vs_dense_bge": _significant(bootstrap, "dense_bge", "hybrid_rrf", "ndcg@10", "hybrid_rrf"),
                "mrr_significant_vs_dense_bge": _significant(bootstrap, "dense_bge", "hybrid_rrf", "mrr", "hybrid_rrf"),
            },
        },
        {
            "claim_id": "claim_intent_stratification_available",
            "claim": "Retrieval metrics are stratified by canonical query intent.",
            "evidence_level": "supported" if intent.get("systems") else "missing",
            "evidence": {
                "systems": sorted((intent.get("systems") or {}).keys()),
                "artifacts": ["reports/eval/intent_stratification_report.json"],
            },
            "caveat": "Several non-exploration intent buckets have very small query counts.",
        },
        {
            "claim_id": "claim_dual_judge_qwk",
            "claim": "Dual-judge reliability can be reported with QWK.",
            "evidence_level": "not_estimable" if not reliability.get("estimable") else "supported",
            "evidence": {
                "pairs_with_two_or_more_judges": reliability.get("pairs_with_two_or_more_judges", 0),
                "pairwise": reliability.get("pairwise", {}),
                "artifacts": ["reports/eval/dual_judge_reliability.json"],
            },
            "caveat": "Current labels have no non-error pair judged by two models.",
        },
    ]


def build_markdown(ledger: list[dict[str, Any]]) -> str:
    lines = [
        "# Eval Claim Ledger",
        "",
        "| Claim | Evidence Level | Key Evidence |",
        "|-------|----------------|--------------|",
    ]
    for row in ledger:
        evidence = row.get("evidence", {})
        key_bits = []
        for key in ["queries", "labeled_pairs", "ndcg@10_delta", "mrr_delta", "recall@50_delta", "pairs_with_two_or_more_judges"]:
            if key in evidence and evidence[key] is not None:
                value = evidence[key]
                if isinstance(value, float):
                    value = round(value, 4)
                key_bits.append(f"{key}={value}")
        if row.get("caveat"):
            key_bits.append(f"caveat={row['caveat']}")
        lines.append(f"| `{row['claim_id']}` | `{row['evidence_level']}` | {'; '.join(key_bits)} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndcg", type=Path, default=DEFAULT_NDCG)
    parser.add_argument("--bootstrap", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--intent", type=Path, default=DEFAULT_INTENT)
    parser.add_argument("--reliability", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    ledger = build_ledger(
        _load_json(args.ndcg),
        _load_json(args.bootstrap),
        _load_json(args.intent),
        _load_json(args.reliability),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    args.md.write_text(build_markdown(ledger), encoding="utf-8")
    print(f"Eval claim ledger -> {args.out}")
    print(f"Markdown -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
