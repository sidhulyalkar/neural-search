#!/usr/bin/env python3
"""Summarize frontend downstream retrieval feedback.

The report treats frontend feedback as downstream usage signal, not gold
relevance labels. It combines search sessions, feedback events, saved/exported
dataset events, and optional neuro-judge/candidate artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_FEEDBACK = ROOT / "artifacts/frontend/retrieval_feedback.jsonl"
DEFAULT_SESSIONS = ROOT / "artifacts/frontend/search_sessions.jsonl"
DEFAULT_SAVED = ROOT / "artifacts/frontend/saved_datasets.jsonl"
DEFAULT_JSON_OUT = ROOT / "reports/eval/downstream_retrieval_success.json"
DEFAULT_MD_OUT = ROOT / "reports/eval/downstream_retrieval_success.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bucket_rank(rank: int | None) -> str:
    if rank is None:
        return "unknown"
    if rank <= 3:
        return "1-3"
    if rank <= 10:
        return "4-10"
    if rank <= 50:
        return "11-50"
    return "51+"


def bucket_completeness(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 0.25:
        return "0.00-0.24"
    if value < 0.50:
        return "0.25-0.49"
    if value < 0.75:
        return "0.50-0.74"
    return "0.75-1.00"


def pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def distribution(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get(field) or "unknown")] += 1
    return dict(sorted(counts.items()))


def usefulness_by(rows: list[dict[str, Any]], key_fn) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[str(key_fn(row))][str(row.get("usefulness") or "unknown")] += 1
    return {key: dict(counter) for key, counter in sorted(grouped.items())}


def judge_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    snapshot = row.get("judge_snapshot")
    return snapshot if isinstance(snapshot, dict) else {}


def compute_report(
    feedback: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    saved: list[dict[str, Any]],
) -> dict[str, Any]:
    query_keys = {
        row.get("query_id") or row.get("query_text") or row.get("session_id")
        for row in feedback + sessions
    }
    query_keys.discard(None)

    false_high = []
    false_low = []
    for row in feedback:
        snapshot = judge_snapshot(row)
        label = snapshot.get("label")
        usefulness = row.get("usefulness")
        if isinstance(label, int) and label >= 2 and usefulness == "not_useful":
            false_high.append(row)
        if isinstance(label, int) and label <= 1 and usefulness == "useful":
            false_low.append(row)

    reason_tags: Counter[str] = Counter()
    for row in feedback:
        for tag in row.get("reason_tags") or []:
            reason_tags[str(tag)] += 1

    save_events = sum(1 for row in feedback if row.get("saved")) + len(saved)
    export_events = sum(1 for row in feedback if row.get("exported")) + sum(
        1 for row in saved if row.get("exported")
    )
    would_use_yes = sum(1 for row in feedback if row.get("would_use_for_analysis") == "yes")
    would_use_yes_or_maybe = sum(
        1 for row in feedback if row.get("would_use_for_analysis") in {"yes", "maybe"}
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "provenance": "user_feedback_downstream_signal",
        "number_of_sessions": len(sessions),
        "number_of_queries": len(query_keys),
        "number_of_feedback_events": len(feedback),
        "usefulness_distribution": distribution(feedback, "usefulness"),
        "usefulness_by_rank_bucket": usefulness_by(
            feedback, lambda row: bucket_rank(row.get("rank"))
        ),
        "usefulness_by_retrieval_method": usefulness_by(
            feedback, lambda row: row.get("retrieval_method") or "unknown"
        ),
        "usefulness_by_neuro_judge_label": usefulness_by(
            feedback, lambda row: judge_snapshot(row).get("label", "missing")
        ),
        "usefulness_by_evidence_completeness_bucket": usefulness_by(
            feedback,
            lambda row: bucket_completeness(
                judge_snapshot(row).get("evidence_completeness")
            ),
        ),
        "save_export_rate": pct(save_events + export_events, len(feedback)),
        "save_rate": pct(save_events, len(feedback)),
        "export_rate": pct(export_events, len(feedback)),
        "would_use_for_analysis_rate": pct(would_use_yes_or_maybe, len(feedback)),
        "would_use_yes_rate": pct(would_use_yes, len(feedback)),
        "false_high_feedback_count": len(false_high),
        "false_low_feedback_count": len(false_low),
        "false_high_feedback": [
            _compact_example(row) for row in false_high[:20]
        ],
        "false_low_feedback": [
            _compact_example(row) for row in false_low[:20]
        ],
        "common_reason_tags": dict(reason_tags.most_common(20)),
    }


def enrich_feedback(
    feedback: list[dict[str, Any]],
    neuro_judge: list[dict[str, Any]],
    rankings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    neuro_by_pair = {
        (row.get("query_id"), row.get("dataset_id")): row
        for row in neuro_judge
        if row.get("query_id") and row.get("dataset_id")
    }
    rank_by_pair: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in rankings:
        dataset_id = row.get("dataset_id")
        if not dataset_id:
            continue
        for query_key in (row.get("query_id"), row.get("query_text")):
            if query_key:
                rank_by_pair[(query_key, dataset_id)] = row

    enriched = []
    for original in feedback:
        row = dict(original)
        dataset_id = row.get("dataset_id")
        query_id = row.get("query_id")
        query_text = row.get("query_text")
        if not row.get("judge_snapshot") and query_id and dataset_id:
            snapshot = neuro_by_pair.get((query_id, dataset_id))
            if snapshot:
                row["judge_snapshot"] = snapshot
        ranking = None
        if query_id and dataset_id:
            ranking = rank_by_pair.get((query_id, dataset_id))
        if ranking is None and query_text and dataset_id:
            ranking = rank_by_pair.get((query_text, dataset_id))
        if ranking:
            row.setdefault("rank", ranking.get("rank"))
            row.setdefault("retrieval_method", ranking.get("retrieval_method"))
        enriched.append(row)
    return enriched


def _compact_example(row: dict[str, Any]) -> dict[str, Any]:
    snapshot = judge_snapshot(row)
    return {
        "query_text": row.get("query_text"),
        "dataset_id": row.get("dataset_id"),
        "dataset_title": row.get("dataset_title"),
        "rank": row.get("rank"),
        "usefulness": row.get("usefulness"),
        "would_use_for_analysis": row.get("would_use_for_analysis"),
        "judge_label": snapshot.get("label"),
        "judge_confidence": snapshot.get("confidence"),
        "evidence_completeness": snapshot.get("evidence_completeness"),
        "reason_tags": row.get("reason_tags") or [],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Downstream Retrieval Success",
        "",
        f"Generated: {report['generated_at']}",
        "",
        (
            "> Feedback provenance: `user_feedback_downstream_signal`. "
            "These are downstream usage signals, not human gold labels."
        ),
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Sessions | {report['number_of_sessions']} |",
        f"| Queries | {report['number_of_queries']} |",
        f"| Feedback events | {report['number_of_feedback_events']} |",
        f"| Save/export rate | {report['save_export_rate']:.1%} |",
        f"| Would-use yes/maybe rate | {report['would_use_for_analysis_rate']:.1%} |",
        f"| False-high feedback | {report['false_high_feedback_count']} |",
        f"| False-low feedback | {report['false_low_feedback_count']} |",
        "",
    ]
    lines.extend(_table("Usefulness Distribution", report["usefulness_distribution"]))
    lines.extend(_nested_table("Usefulness by Rank", report["usefulness_by_rank_bucket"]))
    lines.extend(
        _nested_table(
            "Usefulness by Retrieval Method",
            report["usefulness_by_retrieval_method"],
        )
    )
    lines.extend(
        _nested_table(
            "Usefulness by Neuro-Judge Label",
            report["usefulness_by_neuro_judge_label"],
        )
    )
    lines.extend(
        _nested_table(
            "Usefulness by Evidence Completeness",
            report["usefulness_by_evidence_completeness_bucket"],
        )
    )
    lines.extend(_table("Common Reason Tags", report["common_reason_tags"]))
    return "\n".join(lines) + "\n"


def _table(title: str, values: dict[str, int]) -> list[str]:
    lines = [f"## {title}", "", "| Value | Count |", "|---|---:|"]
    if values:
        lines.extend(f"| {key} | {value} |" for key, value in values.items())
    else:
        lines.append("| none | 0 |")
    lines.append("")
    return lines


def _nested_table(title: str, values: dict[str, dict[str, int]]) -> list[str]:
    lines = [f"## {title}", "", "| Bucket | Useful | Partial | Unsure | Not useful |", "|---|---:|---:|---:|---:|"]
    if values:
        for key, counts in values.items():
            lines.append(
                f"| {key} | {counts.get('useful', 0)} | "
                f"{counts.get('partially_useful', 0)} | "
                f"{counts.get('unsure', 0)} | {counts.get('not_useful', 0)} |"
            )
    else:
        lines.append("| none | 0 | 0 | 0 | 0 |")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feedback", default=str(DEFAULT_FEEDBACK))
    parser.add_argument("--sessions", default=str(DEFAULT_SESSIONS))
    parser.add_argument("--saved", default=str(DEFAULT_SAVED))
    parser.add_argument("--neuro-judge", default=None, help="Optional neuro-judge judgments JSONL.")
    parser.add_argument("--rankings", default=None, help="Optional candidate ranking JSONL.")
    parser.add_argument("--out", "--out-md", dest="out", default=str(DEFAULT_MD_OUT))
    parser.add_argument("--json-out", "--out-json", dest="json_out", default=str(DEFAULT_JSON_OUT))
    args = parser.parse_args(argv)

    feedback = enrich_feedback(
        load_jsonl(Path(args.feedback)),
        load_jsonl(Path(args.neuro_judge)) if args.neuro_judge else [],
        load_jsonl(Path(args.rankings)) if args.rankings else [],
    )
    report = compute_report(feedback, load_jsonl(Path(args.sessions)), load_jsonl(Path(args.saved)))
    json_out = Path(args.json_out)
    md_out = Path(args.out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_out.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
