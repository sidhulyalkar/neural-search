#!/usr/bin/env python3
"""Build a prioritised feedback audit queue for human review.

Combines retrieval feedback, neuro-judge judgments, and search sessions to
identify pairs most in need of expert annotation — judge/user disagreements,
systematically poor results, high-interest low-evidence cases, etc.

All output is labelled downstream_signal, NOT human gold.

Usage::

    python scripts/eval/build_feedback_audit_queue.py \
        --feedback artifacts/frontend/retrieval_feedback.jsonl \
        --sessions artifacts/frontend/search_sessions.jsonl \
        --judgments artifacts/field_state/neuro_qrels_judgments_mock.jsonl \
        --out-jsonl artifacts/field_state/feedback_audit_queue.jsonl \
        --out-md reports/eval/feedback_audit_queue.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_FEEDBACK = ROOT / "artifacts/frontend/retrieval_feedback.jsonl"
DEFAULT_SESSIONS = ROOT / "artifacts/frontend/search_sessions.jsonl"
DEFAULT_JUDGMENTS = ROOT / "artifacts/field_state/neuro_qrels_judgments_mock.jsonl"
DEFAULT_EVIDENCE = ROOT / "artifacts/field_state/neuro_judge_evidence_packets.jsonl"
DEFAULT_JSONL_OUT = ROOT / "artifacts/field_state/feedback_audit_queue.jsonl"
DEFAULT_MD_OUT = ROOT / "reports/eval/feedback_audit_queue.md"

DISCLAIMER = (
    "AUDIT QUEUE — derived from downstream user feedback signals and neuro-judge silver labels. "
    "This is NOT human-annotated ground truth. Use to prioritise expert annotation effort."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _judge_index(judgments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index: (query_id, dataset_id) or (query_text_norm, dataset_id) → judgment."""
    idx: dict[str, dict[str, Any]] = {}
    for j in judgments:
        qid = str(j.get("query_id") or "")
        did = str(j.get("dataset_id") or "")
        if qid and did:
            idx[f"{qid}||{did}"] = j
    return idx


def _norm_query(text: str) -> str:
    return text.lower().strip()


def _judge_for_feedback(
    row: dict[str, Any], idx: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    snap = row.get("judge_snapshot") or {}
    if snap.get("label") is not None:
        return snap
    did = str(row.get("dataset_id") or "")
    qid = str(row.get("query_id") or "")
    if qid and did:
        return idx.get(f"{qid}||{did}")
    return None


def _usefulness_score(usefulness: str) -> float:
    return {"useful": 1.0, "partially_useful": 0.5, "unsure": 0.5, "not_useful": 0.0}.get(
        usefulness, 0.5
    )


def score_entry(
    row: dict[str, Any],
    judge: dict[str, Any] | None,
    show_count: int,
    save_count: int,
) -> tuple[float, list[str]]:
    """Return (priority_score, reason_list). Higher score = higher audit priority."""
    score = 0.0
    reasons: list[str] = []

    usefulness = str(row.get("usefulness") or "unsure")
    rank = int(row.get("rank") or 99)
    ec = float((judge or {}).get("evidence_completeness") or 0.5)
    _j_label_raw = (judge or {}).get("label")
    j_label = int(_j_label_raw) if _j_label_raw is not None else -1
    abstain = bool((judge or {}).get("abstain_recommended"))
    saved = bool(row.get("saved") or row.get("exported"))
    reason_tags = list(row.get("reason_tags") or [])

    # Judge/user agreement disagreements
    if j_label >= 2 and usefulness == "not_useful":
        score += 3.0
        reasons.append("false_high: judge>=2 but user says not_useful")
    if j_label <= 1 and j_label >= 0 and usefulness == "useful":
        score += 3.0
        reasons.append("false_low: judge<=1 but user says useful")

    # High rank but not useful
    if rank <= 3 and usefulness == "not_useful":
        score += 2.5
        reasons.append("high_rank_but_not_useful")

    # Abstain + user thinks it's useful
    if abstain and usefulness in ("useful", "partially_useful"):
        score += 2.0
        reasons.append("abstain_flagged_but_user_finds_useful")

    # Low evidence completeness but saved/exported
    if ec < 0.4 and saved:
        score += 2.0
        reasons.append("low_evidence_but_saved")

    # Repeated exposure without usefulness
    if show_count >= 3 and usefulness == "not_useful":
        score += 1.5
        reasons.append("repeatedly_shown_not_useful")

    # Saved/exported — worth validating the judge label
    if save_count >= 2:
        score += 1.5
        reasons.append("repeatedly_saved_exported")

    # Failure tags that indicate labeling gaps
    high_priority_tags = {"wrong_modality", "wrong_species", "wrong_region", "missing_raw_data"}
    matched_tags = [t for t in reason_tags if t in high_priority_tags]
    if matched_tags:
        score += 1.0 * len(matched_tags)
        reasons.extend(f"user_tag:{t}" for t in matched_tags)

    # Very low EC overall
    if ec < 0.25:
        score += 0.5
        reasons.append("very_low_evidence_completeness")

    return round(score, 2), reasons


def build_audit_queue(
    feedback: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    judge_idx = _judge_index(judgments)

    # Count show and save events per dataset_id
    show_counts: Counter[str] = Counter()
    save_counts: Counter[str] = Counter()
    for row in feedback:
        did = str(row.get("dataset_id") or "")
        show_counts[did] += 1
        if row.get("saved") or row.get("exported"):
            save_counts[did] += 1

    # Build queue entries
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()  # deduplicate by (query_text, dataset_id)

    for row in feedback:
        did = str(row.get("dataset_id") or "")
        qt = str(row.get("query_text") or "")
        key = f"{_norm_query(qt)}||{did}"
        if key in seen:
            continue
        seen.add(key)

        judge = _judge_for_feedback(row, judge_idx)
        priority, reasons = score_entry(
            row, judge, show_counts[did], save_counts[did]
        )

        entry: dict[str, Any] = {
            "priority_score": priority,
            "audit_reasons": reasons,
            "dataset_id": did,
            "dataset_title": row.get("dataset_title") or "",
            "query_text": qt,
            "query_id": row.get("query_id"),
            "session_id": row.get("session_id"),
            "rank": row.get("rank"),
            "retrieval_method": row.get("retrieval_method"),
            "usefulness": row.get("usefulness"),
            "would_use_for_analysis": row.get("would_use_for_analysis"),
            "saved": bool(row.get("saved")),
            "exported": bool(row.get("exported")),
            "reason_tags": list(row.get("reason_tags") or []),
            "free_text_note": row.get("free_text_note"),
            "judge_label": int(judge.get("label") or -1) if judge else None,
            "judge_confidence": float(judge.get("confidence") or 0.0) if judge else None,
            "judge_evidence_completeness": (
                float(judge.get("evidence_completeness") or 0.0) if judge else None
            ),
            "judge_abstain_recommended": bool(judge.get("abstain_recommended")) if judge else None,
            "judge_abstain_reason": judge.get("abstain_reason") if judge else None,
            "judge_model": judge.get("judge_model") if judge else None,
            "judge_label_provenance": judge.get("label_provenance") if judge else None,
            "show_count": show_counts[did],
            "save_count": save_counts[did],
            "provenance": "feedback_audit_queue_downstream_signal",
            "generated_at": datetime.now(tz=UTC).isoformat(),
        }
        entries.append(entry)

    return sorted(entries, key=lambda x: -x["priority_score"])


def render_markdown(
    queue: list[dict[str, Any]],
    stats: dict[str, Any],
) -> str:
    now = datetime.now(tz=UTC).isoformat()
    lines: list[str] = [
        "# Feedback Audit Queue",
        "",
        f"Generated: {now}",
        "",
        f"> {DISCLAIMER}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total entries | {stats['total_entries']} |",
        f"| High priority (score >= 3) | {stats['high_priority']} |",
        f"| Judge/user disagreements | {stats['disagreements']} |",
        f"| High-rank not-useful | {stats['high_rank_not_useful']} |",
        f"| Abstain + user says useful | {stats['abstain_but_useful']} |",
        "",
        "## Top Priority Entries",
        "",
        "| Priority | Dataset | Query | Usefulness | Judge Label | Reasons |",
        "|---:|---|---|---|---:|---|",
    ]

    for entry in queue[:25]:
        reasons_short = "; ".join(entry.get("audit_reasons") or [])[:80]
        j_label = entry.get("judge_label")
        j_label_str = str(j_label) if j_label is not None else "n/a"
        title = str(entry.get("dataset_title") or entry.get("dataset_id") or "")[:40]
        qt = str(entry.get("query_text") or "")[:40]
        lines.append(
            f"| {entry['priority_score']} "
            f"| {title} "
            f"| {qt} "
            f"| {entry.get('usefulness') or 'n/a'} "
            f"| {j_label_str} "
            f"| {reasons_short} |"
        )

    if len(queue) > 25:
        lines.append(f"\n_...and {len(queue) - 25} more entries in the JSONL output_")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build feedback audit queue")
    parser.add_argument(
        "--feedback", type=Path, default=DEFAULT_FEEDBACK, help="Retrieval feedback JSONL"
    )
    parser.add_argument(
        "--sessions", type=Path, default=DEFAULT_SESSIONS, help="Search sessions JSONL"
    )
    parser.add_argument(
        "--judgments", type=Path, default=DEFAULT_JUDGMENTS, help="Neuro-judge judgments JSONL"
    )
    parser.add_argument(
        "--evidence-packets",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="Evidence packets JSONL",
    )
    parser.add_argument(
        "--out-jsonl", type=Path, default=DEFAULT_JSONL_OUT, help="Output JSONL path"
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD_OUT, help="Output MD path")
    args = parser.parse_args(argv)

    feedback = load_jsonl(args.feedback)
    sessions = load_jsonl(args.sessions)
    judgments = load_jsonl(args.judgments)
    print(f"Loaded {len(feedback)} feedback events, {len(sessions)} sessions, {len(judgments)} judgments")

    queue = build_audit_queue(feedback, sessions, judgments)
    print(f"Built audit queue with {len(queue)} entries")

    disagreements = sum(
        1 for e in queue if any("false_high" in r or "false_low" in r for r in e["audit_reasons"])
    )
    high_priority = sum(1 for e in queue if e["priority_score"] >= 3.0)
    high_rank_not_useful = sum(
        1 for e in queue if any("high_rank_but_not_useful" in r for r in e["audit_reasons"])
    )
    abstain_but_useful = sum(
        1
        for e in queue
        if any("abstain_flagged_but_user_finds_useful" in r for r in e["audit_reasons"])
    )
    stats = {
        "total_entries": len(queue),
        "high_priority": high_priority,
        "disagreements": disagreements,
        "high_rank_not_useful": high_rank_not_useful,
        "abstain_but_useful": abstain_but_useful,
    }

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w", encoding="utf-8") as fh:
        for entry in queue:
            fh.write(json.dumps(entry) + "\n")
    print(f"Wrote {args.out_jsonl}")

    md = render_markdown(queue, stats)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
