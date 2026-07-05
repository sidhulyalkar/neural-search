#!/usr/bin/env python3
"""Regression gate for qrels-backed eval reports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_NDCG = Path("reports/eval/ndcg_report.json")
DEFAULT_BOOTSTRAP = Path("reports/eval/bootstrap_ci_report.json")
DEFAULT_RELIABILITY = Path("reports/eval/dual_judge_reliability.json")
DEFAULT_OUT = Path("reports/eval/regression_gate_report.json")
DEFAULT_MD = Path("reports/eval/regression_gate_report.md")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def check_gate(
    ndcg: dict[str, Any],
    bootstrap: dict[str, Any],
    reliability: dict[str, Any],
    min_queries: int,
    min_labeled_pairs: int,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    warnings: list[str] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    n_pairs = int(bootstrap.get("n_labeled_pairs", 0))
    add(
        "qrels_labeled_pairs",
        n_pairs >= min_labeled_pairs,
        f"{n_pairs} labeled pairs >= {min_labeled_pairs}",
    )

    for system in ["bm25", "bm25_structured", "dense_bge", "hybrid_rrf", "hybrid_graph", "full"]:
        if system not in ndcg:
            continue
        n_queries = int((ndcg.get(system) or {}).get("judged_queries", 0))
        add(
            f"{system}_query_coverage",
            n_queries >= min_queries,
            f"{n_queries} judged queries >= {min_queries}",
        )

    bm25 = ndcg.get("bm25") or {}
    hybrid = ndcg.get("hybrid_rrf") or {}
    for metric in ["ndcg@10", "mrr", "recall@50"]:
        h_val = float(hybrid.get(metric, 0.0))
        b_val = float(bm25.get(metric, 0.0))
        add(
            f"hybrid_rrf_ge_bm25_{metric}",
            h_val >= b_val,
            f"hybrid_rrf {metric}={h_val:.4f}; bm25 {metric}={b_val:.4f}",
        )

    graph = ndcg.get("hybrid_graph") or {}
    if graph:
        for metric in ["ndcg@10", "mrr", "recall@50"]:
            g_val = float(graph.get(metric, 0.0))
            h_val = float(hybrid.get(metric, 0.0))
            add(
                f"hybrid_graph_ge_hybrid_rrf_{metric}",
                g_val >= h_val,
                f"hybrid_graph {metric}={g_val:.4f}; hybrid_rrf {metric}={h_val:.4f}",
            )

    if not reliability.get("estimable"):
        warnings.append("Dual-judge QWK is not estimable because no pair has two non-error judge labels.")

    return {
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "warnings": warnings,
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Eval Regression Gate",
        "",
        f"Status: `{'PASS' if report['passed'] else 'FAIL'}`",
        "",
        "| Check | Status | Detail |",
        "|-------|--------|--------|",
    ]
    for check in report["checks"]:
        lines.append(
            f"| {check['name']} | {'PASS' if check['passed'] else 'FAIL'} | {check['detail']} |"
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndcg", type=Path, default=DEFAULT_NDCG)
    parser.add_argument("--bootstrap", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--reliability", type=Path, default=DEFAULT_RELIABILITY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--min-queries", type=int, default=300)
    parser.add_argument("--min-labeled-pairs", type=int, default=10_000)
    args = parser.parse_args(argv)

    report = check_gate(
        _load_json(args.ndcg),
        _load_json(args.bootstrap),
        _load_json(args.reliability),
        args.min_queries,
        args.min_labeled_pairs,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.md.write_text(build_markdown(report), encoding="utf-8")
    print(f"Regression gate -> {args.out}")
    print(f"Markdown -> {args.md}")
    if not report["passed"]:
        print("Eval regression gate failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
