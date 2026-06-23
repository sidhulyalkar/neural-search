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

import yaml

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
    failure_modes: list[str] = field(default_factory=list)
    required_dimensions_missing: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    variant: str = ""


@dataclass
class FalseNegative:
    query_id: str
    record_id: str
    relevance: int
    best_rank: int | None  # None = not retrieved at all in top-K
    required_dimensions_missing: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
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
    fp_failure_mode_counts: Counter = field(default_factory=Counter)
    fp_mismatch_counts: Counter = field(default_factory=Counter)
    fp_metadata_missing_counts: Counter = field(default_factory=Counter)
    fp_hard_negative_failure_counts: Counter = field(default_factory=Counter)
    fn_metadata_missing_counts: Counter = field(default_factory=Counter)


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


def _load_queries(path: Path) -> dict[str, dict[str, Any]]:
    """Load query metadata from JSONL, YAML list, or YAML wrapper."""
    if not path.exists():
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            rows = []
            for value in data.values():
                if isinstance(value, list):
                    rows = value
                    break
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
    else:
        rows = _load_jsonl(path)

    queries: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("query_id") or row.get("id") or "")
        if qid:
            queries[qid] = row
    return queries


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
            "rationale": row.get("rationale") or row.get("rationale_short", ""),
            "hard_negative_violation": bool(
                row.get("hard_negative_violation", row.get("hard_negative_detected", False))
            ),
        }
    return dict(scores), dict(meta)


def _load_judgment_meta(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load explanatory neuro-judge metadata keyed by (query_id, dataset_id)."""
    meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _load_jsonl(path):
        if "judge_error" in str(row.get("rationale_short", "")):
            continue
        qid = str(row.get("query_id", ""))
        did = str(row.get("dataset_id") or row.get("record_id", ""))
        if not qid or not did:
            continue
        meta[(qid, did)] = {
            "rationale": row.get("rationale_short", ""),
            "hard_negative_violation": bool(row.get("hard_negative_detected", False)),
            "failure_modes": list(row.get("failure_modes") or []),
            "required_dimensions_missing": list(row.get("required_dimensions_missing") or []),
            "missing_information": list(row.get("missing_information") or []),
            "evidence_against": list(row.get("evidence_against") or []),
        }
    return meta


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
        if variant_runs:
            runs[variant] = dict(variant_runs)
    return runs


def _source_prefix(record_id: str) -> str:
    return record_id.split(":")[0] if ":" in record_id else "unknown"


def _merge_meta(
    qrel_meta: dict[str, dict[str, Any]],
    judgment_meta: dict[tuple[str, str], dict[str, Any]],
    query_id: str,
    record_id: str,
) -> dict[str, Any]:
    merged = dict(qrel_meta.get(record_id, {}))
    merged.update(judgment_meta.get((query_id, record_id), {}))
    return merged


def _mismatch_buckets(failure_modes: list[str], missing_dims: list[str]) -> set[str]:
    text = " ".join([*failure_modes, *missing_dims]).lower()
    buckets: set[str] = set()
    if "modality" in text or "neural_record" in text:
        buckets.add("modality_mismatch")
    if "species" in text:
        buckets.add("species_mismatch")
    if "task" in text:
        buckets.add("task_mismatch")
    if "region" in text or "brain_region" in text:
        buckets.add("brain_region_mismatch")
    if "behavior" in text or "event" in text or "lick" in text:
        buckets.add("behavioral_event_mismatch")
    if "raw" in text:
        buckets.add("raw_data_missing")
    return buckets


def _hard_negative_buckets(failure_modes: list[str], hard_negative: bool) -> set[str]:
    buckets: set[str] = set()
    for mode in failure_modes:
        lower = mode.lower()
        if "hard_negative" in lower or "hard-negative" in lower:
            buckets.add(mode)
    if hard_negative and not buckets:
        buckets.add("hard_negative_detected")
    return buckets


def _normalize_missing_dimension(value: str) -> str:
    lower = value.lower()
    if "species" in lower:
        return "species"
    if "modality" in lower or "modalities" in lower or "eeg" in lower or "ieeg" in lower or "ecog" in lower or "neuropixels" in lower or "calcium" in lower or "lfp" in lower:
        return "modality"
    if "brain_region" in lower or "brain region" in lower or "region" in lower or "cortex" in lower or "hippocampus" in lower or "striatum" in lower:
        return "brain_region"
    if "task" in lower or "tasks" in lower:
        return "task"
    if "affordance" in lower or "analysis" in lower or "decoding" in lower or "connectivity" in lower:
        return "affordance"
    if "raw" in lower:
        return "raw_data"
    if "behavior" in lower or "event" in lower or "lick" in lower or "choice" in lower or "reward" in lower:
        return "behavioral_event"
    if "standard" in lower or "format" in lower or "bids" in lower or "nwb" in lower:
        return "data_standard"
    return "other"


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _analyze_variant(
    variant: str,
    ranked: dict[str, list[tuple[int, str, float]]],
    qrel_scores: dict[str, dict[str, int]],
    qrel_meta: dict[str, dict[str, Any]],
    judgment_meta: dict[tuple[str, str], dict[str, Any]],
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
                m = _merge_meta(meta, judgment_meta, query_id, rid)
                failure_modes = list(m.get("failure_modes") or [])
                missing_dims = list(m.get("required_dimensions_missing") or [])
                missing_info = list(m.get("missing_information") or [])
                fp = FalsePositive(
                    query_id=query_id,
                    record_id=rid,
                    rank=top_k_ranks[rid],
                    score=top_k_scores[rid],
                    relevance=rel,
                    rationale=m.get("rationale", ""),
                    hard_negative_violation=m.get("hard_negative_violation", False),
                    failure_modes=failure_modes,
                    required_dimensions_missing=missing_dims,
                    missing_information=missing_info,
                    variant=variant,
                )
                report.false_positives.append(fp)
                if fp.hard_negative_violation:
                    report.hn_violations.append(fp)
                report.source_fp_counts[_source_prefix(rid)] += 1
                report.intent_fp_counts[intent] += 1
                for mode in failure_modes:
                    report.fp_failure_mode_counts[mode] += 1
                for bucket in _mismatch_buckets(failure_modes, missing_dims):
                    report.fp_mismatch_counts[bucket] += 1
                for dim in missing_dims:
                    report.fp_metadata_missing_counts[_normalize_missing_dimension(dim)] += 1
                for bucket in _hard_negative_buckets(failure_modes, fp.hard_negative_violation):
                    report.fp_hard_negative_failure_counts[bucket] += 1

        # False negatives: relevant but not in top-K
        for did, rel in qrels.items():
            if rel < 2:
                continue
            if did in top_k_ids:
                continue
            # Check if retrieved beyond top-K
            all_ranked = {rid: rank for rank, rid, _ in rows}
            best_rank = all_ranked.get(did)
            m = _merge_meta(meta, judgment_meta, query_id, did)
            missing_dims = list(m.get("required_dimensions_missing") or [])
            missing_info = list(m.get("missing_information") or [])
            fn = FalseNegative(
                query_id=query_id,
                record_id=did,
                relevance=rel,
                best_rank=best_rank,
                required_dimensions_missing=missing_dims,
                missing_information=missing_info,
                variant=variant,
            )
            report.false_negatives.append(fn)
            report.source_fn_counts[_source_prefix(did)] += 1
            report.intent_fn_counts[intent] += 1
            for dim in missing_dims:
                report.fn_metadata_missing_counts[_normalize_missing_dimension(dim)] += 1

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

        if report.fp_mismatch_counts:
            lines.append("### False-Positive Mismatch Breakdown\n")
            lines.append("| Mismatch bucket | Count |")
            lines.append("|-----------------|-------|")
            for bucket, count in report.fp_mismatch_counts.most_common():
                lines.append(f"| {bucket} | {count} |")
            lines.append("")

        if report.fp_metadata_missing_counts or report.fn_metadata_missing_counts:
            lines.append("### Metadata Missingness Breakdown\n")
            lines.append("| Missing dimension | FP Count | FN Count |")
            lines.append("|-------------------|----------|----------|")
            dims = set(report.fp_metadata_missing_counts) | set(report.fn_metadata_missing_counts)
            for dim in sorted(dims):
                lines.append(
                    f"| {dim} | {report.fp_metadata_missing_counts.get(dim, 0)} "
                    f"| {report.fn_metadata_missing_counts.get(dim, 0)} |"
                )
            lines.append("")

        if report.fp_failure_mode_counts:
            lines.append("### Top False-Positive Failure Modes\n")
            lines.append("| Failure mode | Count |")
            lines.append("|--------------|-------|")
            for mode, count in report.fp_failure_mode_counts.most_common(20):
                lines.append(f"| {mode} | {count} |")
            lines.append("")

        if report.fp_hard_negative_failure_counts:
            lines.append("### Hard-Negative Failure Modes\n")
            lines.append("| Hard-negative failure mode | Count |")
            lines.append("|----------------------------|-------|")
            for mode, count in report.fp_hard_negative_failure_counts.most_common():
                lines.append(f"| {mode} | {count} |")
            lines.append("")

    return "\n".join(lines)


def _counter_dict(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def _reports_to_json(reports: list[FailureReport], queries_count: int, top_k: int) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for report in reports:
        variants[report.variant] = {
            "top_k": top_k,
            "false_positive_count": len(report.false_positives),
            "false_negative_count": len(report.false_negatives),
            "hard_negative_violation_count": len(report.hn_violations),
            "source_fp_counts": _counter_dict(report.source_fp_counts),
            "source_fn_counts": _counter_dict(report.source_fn_counts),
            "intent_fp_counts": _counter_dict(report.intent_fp_counts),
            "intent_fn_counts": _counter_dict(report.intent_fn_counts),
            "fp_mismatch_counts": _counter_dict(report.fp_mismatch_counts),
            "fp_metadata_missing_counts": _counter_dict(report.fp_metadata_missing_counts),
            "fn_metadata_missing_counts": _counter_dict(report.fn_metadata_missing_counts),
            "fp_failure_mode_counts": _counter_dict(report.fp_failure_mode_counts),
            "fp_hard_negative_failure_counts": _counter_dict(report.fp_hard_negative_failure_counts),
            "top_false_positives": [
                {
                    "query_id": fp.query_id,
                    "record_id": fp.record_id,
                    "rank": fp.rank,
                    "relevance": fp.relevance,
                    "source": _source_prefix(fp.record_id),
                    "hard_negative_violation": fp.hard_negative_violation,
                    "failure_modes": fp.failure_modes,
                    "required_dimensions_missing": fp.required_dimensions_missing,
                    "missing_information": fp.missing_information,
                    "rationale": fp.rationale,
                }
                for fp in sorted(report.false_positives, key=lambda x: x.rank)[:100]
            ],
            "top_false_negatives": [
                {
                    "query_id": fn.query_id,
                    "record_id": fn.record_id,
                    "relevance": fn.relevance,
                    "source": _source_prefix(fn.record_id),
                    "best_rank": fn.best_rank,
                    "required_dimensions_missing": fn.required_dimensions_missing,
                    "missing_information": fn.missing_information,
                }
                for fn in sorted(
                    report.false_negatives,
                    key=lambda x: (-(x.relevance), x.best_rank or 10**9),
                )[:100]
            ],
        }
    return {
        "queries_count": queries_count,
        "variants": variants,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyze_failures(
    qrels_path: Path,
    queries_path: Path,
    runs_dir: Path,
    out_path: Path,
    json_out_path: Path | None = None,
    judgments_path: Path | None = None,
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

    queries = _load_queries(queries_path)
    query_intent = {
        qid: str(row.get("intent", "UNKNOWN"))
        for qid, row in queries.items()
    }
    judgment_meta = _load_judgment_meta(judgments_path) if judgments_path else {}

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
            judgment_meta=judgment_meta,
            query_intent=query_intent,
            top_k=top_k,
        )
        reports.append(report)

    markdown = _render_markdown(reports, queries_count=len(query_intent) or len(qrel_scores))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(markdown, encoding="utf-8")
    tmp.replace(out_path)
    if json_out_path:
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(
            json.dumps(
                _reports_to_json(reports, queries_count=len(query_intent) or len(qrel_scores), top_k=top_k),
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Failure analysis JSON written to {json_out_path}")
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
    parser.add_argument(
        "--json-out", type=Path, default=Path("reports/eval/benchmark_v1_failures.json")
    )
    parser.add_argument(
        "--judgments", type=Path, default=Path("data/qrels/llm_judgments.jsonl")
    )
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args(argv)

    return analyze_failures(
        qrels_path=args.qrels,
        queries_path=args.queries,
        runs_dir=args.runs_dir,
        out_path=args.out,
        json_out_path=args.json_out,
        judgments_path=args.judgments,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    sys.exit(main())
