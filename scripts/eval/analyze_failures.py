"""Failure analysis for Neural Search benchmark evaluation.

Produces structured reports on:
- False positives: high-ranked results that are not relevant (relevance=0 or 1)
- False negatives: relevant results (relevance≥2) not retrieved in top-K
- Hard-negative violations: results matching a stated hard-negative pattern
- Concept-specific failures: poor results linked to specific concept types
- Source-specific failures: sources that over- or under-perform

Usage:
    python scripts/eval/analyze_failures.py \\
        --qrels artifacts/qrels.jsonl \\
        --queries artifacts/benchmark_queries.jsonl \\
        --runs-dir reports/eval/runs \\
        --out reports/eval/benchmark_v1_failures.md \\
        [--top-k 10]

If qrels do not exist or are empty, writes a placeholder report and exits 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FalsePositive:
    query_id: str
    record_id: str
    rank: int
    score: float
    relevance: int
    rationale: str = ""
    hard_negative_violation: bool = False
    variant: str = ""


@dataclass
class FalseNegative:
    query_id: str
    record_id: str
    relevance: int
    best_rank: int | None  # None = not retrieved at all in top-K
    variant: str = ""


@dataclass
class FailureReport:
    variant: str
    false_positives: list[FalsePositive] = field(default_factory=list)
    false_negatives: list[FalseNegative] = field(default_factory=list)
    hn_violations: list[FalsePositive] = field(default_factory=list)
    source_fp_counts: Counter = field(default_factory=Counter)
    source_fn_counts: Counter = field(default_factory=Counter)
    intent_fp_counts: Counter = field(default_factory=Counter)
    intent_fn_counts: Counter = field(default_factory=Counter)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _load_qrels(
    path: Path,
) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, Any]]]:
    """Returns (qrel_scores, qrel_meta).

    qrel_scores[query_id][dataset_id] = relevance
    qrel_meta[query_id][dataset_id] = {rationale, hard_negative_violation}
    """
    scores: dict[str, dict[str, int]] = defaultdict(dict)
    meta: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in _load_jsonl(path):
        qid = str(row.get("query_id", ""))
        did = str(row.get("dataset_id") or row.get("record_id", ""))
        rel = int(row.get("relevance", row.get("label", 0)))
        scores[qid][did] = rel
        meta[qid][did] = {
            "rationale": row.get("rationale", ""),
            "hard_negative_violation": bool(row.get("hard_negative_violation", False)),
        }
    return dict(scores), dict(meta)


def _load_runs(
    runs_dir: Path, depth: int
) -> dict[str, dict[str, list[tuple[int, str, float]]]]:
    """runs[variant][query_id] = [(rank, record_id, score)]"""
    runs: dict[str, dict[str, list[tuple[int, str, float]]]] = {}
    for run_file in sorted(runs_dir.glob("*.jsonl")) if runs_dir.exists() else []:
        variant = run_file.stem
        variant_runs: dict[str, list[tuple[int, str, float]]] = defaultdict(list)
        for row in _load_jsonl(run_file):
            rank = int(row.get("rank", 10**9))
            if rank <= depth:
                qid = str(row["query_id"])
                rid = str(row.get("record_id", row.get("dataset_id", "")))
                score = float(row.get("score", 0.0))
                variant_runs[qid].append((rank, rid, score))
        for qid in variant_runs:
            variant_runs[qid].sort(key=lambda x: x[0])
        runs[variant] = dict(variant_runs)
    return runs


def _source_prefix(record_id: str) -> str:
    return record_id.split(":")[0] if ":" in record_id else "unknown"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _analyze_variant(
    variant: str,
    ranked: dict[str, list[tuple[int, str, float]]],
    qrel_scores: dict[str, dict[str, int]],
    qrel_meta: dict[str, dict[str, Any]],
    query_intent: dict[str, str],
    top_k: int,
) -> FailureReport:
    report = FailureReport(variant=variant)

    for query_id, rows in ranked.items():
        qrels = qrel_scores.get(query_id, {})
        meta = qrel_meta.get(query_id, {})
        intent = query_intent.get(query_id, "UNKNOWN")

        top_k_ids = [rid for _, rid, _ in rows[:top_k]]
        top_k_scores = {rid: score for _, rid, score in rows[:top_k]}
        top_k_ranks = {rid: rank for rank, rid, _ in rows[:top_k]}

        # False positives: in top-K but irrelevant
        for rid in top_k_ids:
            if rid not in qrels:
                continue
            rel = qrels[rid]
            if rel <= 1:
                m = meta.get(rid, {})
                fp = FalsePositive(
                    query_id=query_id,
                    record_id=rid,
                    rank=top_k_ranks[rid],
                    score=top_k_scores[rid],
                    relevance=rel,
                    rationale=m.get("rationale", ""),
                    hard_negative_violation=m.get("hard_negative_violation", False),
                    variant=variant,
                )
                report.false_positives.append(fp)
                if fp.hard_negative_violation:
                    report.hn_violations.append(fp)
                report.source_fp_counts[_source_prefix(rid)] += 1
                report.intent_fp_counts[intent] += 1

        # False negatives: relevant but not in top-K
        for did, rel in qrels.items():
            if rel < 2:
                continue
            if did in top_k_ids:
                continue
            # Check if retrieved beyond top-K
            all_ranked = {rid: rank for rank, rid, _ in rows}
            best_rank = all_ranked.get(did)
            fn = FalseNegative(
                query_id=query_id,
                record_id=did,
                relevance=rel,
                best_rank=best_rank,
                variant=variant,
            )
            report.false_negatives.append(fn)
            report.source_fn_counts[_source_prefix(did)] += 1
            report.intent_fn_counts[intent] += 1

    return report


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_placeholder(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# Failure Analysis Report\n\n"
        "**Status:** No adjudicated qrels found.\n\n"
        "This report requires `artifacts/qrels.jsonl` with at least one adjudicated "
        "entry. See `docs/benchmark_v1_spec.md` for the annotation workflow.\n\n"
        "Re-run after completing annotation:\n"
        "```\n"
        "python scripts/eval/analyze_failures.py \\\n"
        "    --qrels artifacts/qrels.jsonl \\\n"
        "    --queries artifacts/benchmark_queries.jsonl \\\n"
        "    --runs-dir reports/eval/runs\n"
        "```\n",
        encoding="utf-8",
    )


def _render_markdown(reports: list[FailureReport], queries_count: int) -> str:
    lines: list[str] = [
        "# Failure Analysis Report\n",
        f"Queries evaluated: {queries_count}  ",
        f"Variants: {', '.join(r.variant for r in reports)}\n",
        "---\n",
    ]

    for report in reports:
        lines.append(f"## Variant: `{report.variant}`\n")

        # Summary table
        fp_n = len(report.false_positives)
        fn_n = len(report.false_negatives)
        hn_n = len(report.hn_violations)
        lines.append(
            f"| Metric | Count |\n"
            f"|--------|-------|\n"
            f"| False positives (top-K, relevance ≤ 1) | {fp_n} |\n"
            f"| False negatives (relevant, not in top-K) | {fn_n} |\n"
            f"| Hard-negative violations | {hn_n} |\n"
        )
        lines.append("")

        # Top false positives
        if report.false_positives:
            lines.append("### Top False Positives (rank ≤ 10)\n")
            lines.append("| Rank | Record | Relevance | HN | Query | Rationale |")
            lines.append("|------|--------|-----------|----|----|-----------|")
            top_fps = sorted(report.false_positives, key=lambda x: x.rank)[:20]
            for fp in top_fps:
                hn = "**YES**" if fp.hard_negative_violation else "no"
                rat = fp.rationale[:80].replace("|", "/") if fp.rationale else "—"
                lines.append(
                    f"| {fp.rank} | `{fp.record_id}` | {fp.relevance} | {hn} "
                    f"| {fp.query_id} | {rat} |"
                )
            lines.append("")

        # Top false negatives (missed relevant)
        if report.false_negatives:
            lines.append("### False Negatives (relevant, not in top-K)\n")
            lines.append("| Record | Relevance | Best Rank Outside K | Query |")
            lines.append("|--------|-----------|--------------------|----|")
            top_fns = sorted(
                report.false_negatives,
                key=lambda x: (-(x.relevance), x.best_rank or 10**9),
            )[:20]
            for fn in top_fns:
                br = str(fn.best_rank) if fn.best_rank else "not retrieved"
                lines.append(
                    f"| `{fn.record_id}` | {fn.relevance} | {br} | {fn.query_id} |"
                )
            lines.append("")

        # Source breakdown
        if report.source_fp_counts or report.source_fn_counts:
            lines.append("### Source Breakdown\n")
            lines.append("| Source | FP Count | FN Count |")
            lines.append("|--------|----------|----------|")
            sources = set(report.source_fp_counts) | set(report.source_fn_counts)
            for src in sorted(sources):
                fp_n = report.source_fp_counts.get(src, 0)
                fn_n = report.source_fn_counts.get(src, 0)
                lines.append(f"| {src} | {fp_n} | {fn_n} |")
            lines.append("")

        # Intent breakdown
        if report.intent_fp_counts or report.intent_fn_counts:
            lines.append("### By Query Intent\n")
            lines.append("| Intent | FP Count | FN Count |")
            lines.append("|--------|----------|----------|")
            intents = set(report.intent_fp_counts) | set(report.intent_fn_counts)
            for intent in sorted(intents):
                fp_n = report.intent_fp_counts.get(intent, 0)
                fn_n = report.intent_fn_counts.get(intent, 0)
                lines.append(f"| {intent} | {fp_n} | {fn_n} |")
            lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyze_failures(
    qrels_path: Path,
    queries_path: Path,
    runs_dir: Path,
    out_path: Path,
    top_k: int = 10,
) -> int:
    qrel_scores, qrel_meta = _load_qrels(qrels_path)
    if not qrel_scores:
        print(
            f"[INFO] No qrels found at {qrels_path} — writing placeholder report.",
            file=sys.stderr,
        )
        _render_placeholder(out_path)
        print(f"Placeholder written to {out_path}")
        return 0

    # Load query intents
    query_intent: dict[str, str] = {}
    for row in _load_jsonl(queries_path):
        qid = str(row.get("query_id", ""))
        intent = str(row.get("intent", "UNKNOWN"))
        if qid:
            query_intent[qid] = intent

    runs = _load_runs(runs_dir, depth=top_k * 5)
    if not runs:
        print(f"[WARN] No run files found in {runs_dir}", file=sys.stderr)

    reports: list[FailureReport] = []
    for variant, ranked in runs.items():
        report = _analyze_variant(
            variant=variant,
            ranked=ranked,
            qrel_scores=qrel_scores,
            qrel_meta=qrel_meta,
            query_intent=query_intent,
            top_k=top_k,
        )
        reports.append(report)

    markdown = _render_markdown(reports, queries_count=len(query_intent))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(markdown, encoding="utf-8")
    tmp.replace(out_path)
    print(f"Failure analysis written to {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Failure analysis for benchmark eval")
    parser.add_argument(
        "--qrels", type=Path, default=Path("artifacts/qrels.jsonl")
    )
    parser.add_argument(
        "--queries", type=Path, default=Path("artifacts/benchmark_queries.jsonl")
    )
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("reports/eval/runs")
    )
    parser.add_argument(
        "--out", type=Path, default=Path("reports/eval/benchmark_v1_failures.md")
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    return analyze_failures(
        qrels_path=args.qrels,
        queries_path=args.queries,
        runs_dir=args.runs_dir,
        out_path=args.out,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    sys.exit(main())
