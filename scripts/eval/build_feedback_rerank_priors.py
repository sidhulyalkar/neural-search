#!/usr/bin/env python3
"""Build transparent feedback-prior reranking weights.

Derives simple, interpretable boost/penalty factors from aggregated user feedback.
These priors are applied on top of retrieval scores — they do NOT replace them.
All weights are transparent: each entry includes the evidence that drove the adjustment.

This is NOT a learned model. It is a rule-based prior layer that can be toggled
on/off in the frontend and audited at any time.

Provenance: all weights labelled `feedback_rerank_prior_downstream_signal`.
These are NOT human gold labels.

Usage::

    python scripts/eval/build_feedback_rerank_priors.py \
        --feedback artifacts/frontend/retrieval_feedback.jsonl \
        --judgments artifacts/field_state/neuro_qrels_judgments_mock.jsonl \
        --out-json artifacts/field_state/feedback_rerank_priors.json \
        --out-md reports/eval/feedback_rerank_priors.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_FEEDBACK = ROOT / "artifacts/frontend/retrieval_feedback.jsonl"
DEFAULT_SESSIONS = ROOT / "artifacts/frontend/search_sessions.jsonl"
DEFAULT_JUDGMENTS = ROOT / "artifacts/field_state/neuro_qrels_judgments_mock.jsonl"
DEFAULT_JSON_OUT = ROOT / "artifacts/field_state/feedback_rerank_priors.json"
DEFAULT_MD_OUT = ROOT / "reports/eval/feedback_rerank_priors.md"

DISCLAIMER = (
    "RERANKING PRIORS — derived from downstream user feedback signals only. "
    "These weights are transparent, rule-based, and labelled downstream_signal. "
    "They are NOT human-annotated relevance labels. Audit-priority flags are "
    "separate from the score and must not be used as quality gates without expert review."
)

# Minimum feedback events before we assign any prior
MIN_EVENTS_FOR_PRIOR = 2
# Max magnitude of log-odds adjustment
MAX_LOG_ODDS_ADJ = 1.5


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


def _usefulness_score(usefulness: str) -> float:
    return {"useful": 1.0, "partially_useful": 0.5, "unsure": 0.5, "not_useful": 0.0}.get(
        usefulness, 0.5
    )


def _norm_query_intent(query_text: str, tags: list[str]) -> str:
    """Coarse intent label from query text and user tags.

    Groups similar queries so priors generalise across surface variations.
    """
    text = query_text.lower()
    # Crude heuristic: tag by dominant neuroscience concept
    for keyword, intent in [
        ("place cell", "place_cells"),
        ("theta", "theta_oscillation"),
        ("calcium imaging", "calcium_imaging"),
        ("spike sort", "spike_sorting"),
        ("replication", "replication"),
        ("reversal learning", "reversal_learning"),
        ("decision", "decision_making"),
        ("visual", "visual_processing"),
        ("motor", "motor_control"),
    ]:
        if keyword in text:
            return intent
    # Fall back to dominant reason tag
    for tag in tags:
        if tag in {"wrong_species", "wrong_modality", "wrong_region"}:
            return f"negative_{tag}"
    return "general"


def _judge_label_for(row: dict[str, Any]) -> int | None:
    snap = row.get("judge_snapshot") or {}
    label = snap.get("label")
    if label is not None:
        try:
            return int(label)
        except (TypeError, ValueError):
            pass
    return None


def build_priors(
    feedback: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
) -> dict[str, Any]:
    # ---------- aggregate per (dataset_id) ----------
    dataset_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "useful_count": 0,
            "partial_count": 0,
            "not_useful_count": 0,
            "unsure_count": 0,
            "would_use_yes": 0,
            "would_use_no": 0,
            "save_count": 0,
            "reason_tags": Counter(),
            "query_intents": Counter(),
            "judge_disagreement_count": 0,
            "low_ec_high_interest": 0,
            "titles": set(),
        }
    )

    for row in feedback:
        did = str(row.get("dataset_id") or "")
        if not did:
            continue
        s = dataset_stats[did]
        usefulness = str(row.get("usefulness") or "unsure")
        if usefulness == "useful":
            s["useful_count"] += 1
        elif usefulness == "partially_useful":
            s["partial_count"] += 1
        elif usefulness == "not_useful":
            s["not_useful_count"] += 1
        else:
            s["unsure_count"] += 1

        wua = str(row.get("would_use_for_analysis") or "")
        if wua == "yes":
            s["would_use_yes"] += 1
        elif wua == "no":
            s["would_use_no"] += 1

        if row.get("saved") or row.get("exported"):
            s["save_count"] += 1

        for tag in (row.get("reason_tags") or []):
            s["reason_tags"][str(tag)] += 1

        qt = str(row.get("query_text") or "")
        tags = list(row.get("reason_tags") or [])
        intent = _norm_query_intent(qt, tags)
        s["query_intents"][intent] += 1

        title = str(row.get("dataset_title") or "")
        if title:
            s["titles"].add(title)

        # Judge/user disagreement
        j_label = _judge_label_for(row)
        if j_label is not None:
            if j_label >= 2 and usefulness == "not_useful":
                s["judge_disagreement_count"] += 1
            if j_label <= 1 and usefulness == "useful":
                s["judge_disagreement_count"] += 1

        # Low EC but high interest
        snap = row.get("judge_snapshot") or {}
        ec = float(snap.get("evidence_completeness") or 0.5)
        if ec < 0.4 and usefulness in ("useful", "partially_useful"):
            s["low_ec_high_interest"] += 1

    # ---------- compute per-dataset priors ----------
    dataset_priors: list[dict[str, Any]] = []

    for did, s in dataset_stats.items():
        total = s["useful_count"] + s["partial_count"] + s["not_useful_count"] + s["unsure_count"]
        if total < MIN_EVENTS_FOR_PRIOR:
            continue

        # Weighted usefulness rate
        weighted = (s["useful_count"] + 0.5 * s["partial_count"] + 0.5 * s["unsure_count"])
        usefulness_rate = weighted / total

        # Log-odds adjustment relative to neutral (0.5)
        log_odds_adj = math.log((usefulness_rate + 1e-6) / (1 - usefulness_rate + 1e-6))
        log_odds_adj = max(-MAX_LOG_ODDS_ADJ, min(MAX_LOG_ODDS_ADJ, log_odds_adj))

        # Penalty for repeated "wrong_" tags
        penalty_tags = {t: c for t, c in s["reason_tags"].items() if t.startswith("wrong_")}
        penalty = sum(penalty_tags.values()) * -0.1

        # Boost for repeated saves
        save_boost = min(0.3, s["save_count"] * 0.1)

        net_adjustment = round(log_odds_adj + penalty + save_boost, 3)

        # Audit priority
        audit_priority = s["judge_disagreement_count"] >= 1 or s["low_ec_high_interest"] >= 1

        # Dominant wrong-reason tags
        wrong_tags = sorted(
            [(t, c) for t, c in s["reason_tags"].items() if t.startswith("wrong_")],
            key=lambda x: -x[1],
        )

        # Top query intents this dataset appeared in
        top_intents = [intent for intent, _ in s["query_intents"].most_common(3)]

        prior: dict[str, Any] = {
            "dataset_id": did,
            "titles": sorted(s["titles"])[:3],
            "total_feedback_events": total,
            "usefulness_rate": round(usefulness_rate, 3),
            "net_score_adjustment": net_adjustment,
            "save_boost": round(save_boost, 3),
            "wrong_tag_penalty": round(penalty, 3),
            "save_count": s["save_count"],
            "useful_count": s["useful_count"],
            "not_useful_count": s["not_useful_count"],
            "judge_disagreement_count": s["judge_disagreement_count"],
            "low_ec_high_interest": s["low_ec_high_interest"],
            "audit_priority": audit_priority,
            "dominant_wrong_tags": wrong_tags[:5],
            "top_query_intents": top_intents,
            "provenance": "feedback_rerank_prior_downstream_signal",
        }
        dataset_priors.append(prior)

    dataset_priors.sort(key=lambda x: -abs(x["net_score_adjustment"]))

    # ---------- summary stats ----------
    boost_count = sum(1 for p in dataset_priors if p["net_score_adjustment"] > 0.1)
    penalty_count = sum(1 for p in dataset_priors if p["net_score_adjustment"] < -0.1)
    audit_count = sum(1 for p in dataset_priors if p["audit_priority"])

    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "disclaimer": DISCLAIMER,
        "provenance": "feedback_rerank_prior_downstream_signal",
        "min_events_for_prior": MIN_EVENTS_FOR_PRIOR,
        "max_log_odds_adjustment": MAX_LOG_ODDS_ADJ,
        "total_datasets_with_priors": len(dataset_priors),
        "datasets_with_boost": boost_count,
        "datasets_with_penalty": penalty_count,
        "datasets_flagged_for_audit": audit_count,
        "dataset_priors": dataset_priors,
    }


def render_markdown(priors: dict[str, Any]) -> str:
    now = priors["generated_at"]
    total = priors["total_datasets_with_priors"]
    lines: list[str] = [
        "# Feedback Rerank Priors",
        "",
        f"Generated: {now}",
        "",
        f"> {priors['disclaimer']}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total datasets with priors | {total} |",
        f"| Datasets receiving boost | {priors['datasets_with_boost']} |",
        f"| Datasets receiving penalty | {priors['datasets_with_penalty']} |",
        f"| Datasets flagged for audit | {priors['datasets_flagged_for_audit']} |",
        "",
        "## Prior Weights (top 30 by |adjustment|)",
        "",
        "| Dataset ID | Titles | Events | Usefulness | Adjustment | Save Boost | Tag Penalty | Audit |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for p in priors["dataset_priors"][:30]:
        titles = "; ".join(t[:30] for t in p.get("titles") or [])
        lines.append(
            f"| {p['dataset_id']} "
            f"| {titles} "
            f"| {p['total_feedback_events']} "
            f"| {p['usefulness_rate']:.2f} "
            f"| {p['net_score_adjustment']:+.3f} "
            f"| {p['save_boost']:+.3f} "
            f"| {p['wrong_tag_penalty']:+.3f} "
            f"| {'YES' if p['audit_priority'] else 'no'} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- `net_score_adjustment`: positive = boost this dataset in results; negative = penalise.",
        "- Adjustments are capped at ±1.5 log-odds units.",
        "- `audit_priority=YES` means judge/user disagreement or low-evidence high-interest.",
        "- These weights are downstream signals only. Do NOT use as quality gates without expert review.",
        "- Toggle this layer on/off using the frontend reranking switch.",
    ]

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build feedback rerank priors")
    parser.add_argument(
        "--feedback", type=Path, default=DEFAULT_FEEDBACK, help="Retrieval feedback JSONL"
    )
    parser.add_argument(
        "--sessions", type=Path, default=DEFAULT_SESSIONS, help="Search sessions JSONL (unused)"
    )
    parser.add_argument(
        "--judgments", type=Path, default=DEFAULT_JUDGMENTS, help="Neuro-judge judgments JSONL"
    )
    parser.add_argument(
        "--out-json", type=Path, default=DEFAULT_JSON_OUT, help="Output JSON path"
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD_OUT, help="Output MD path")
    args = parser.parse_args(argv)

    feedback = load_jsonl(args.feedback)
    judgments = load_jsonl(args.judgments)
    print(f"Loaded {len(feedback)} feedback events, {len(judgments)} judgments")

    output = build_priors(feedback, judgments)
    print(
        f"Built priors for {output['total_datasets_with_priors']} datasets "
        f"({output['datasets_with_boost']} boosted, {output['datasets_with_penalty']} penalised)"
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"Wrote {args.out_json}")

    md = render_markdown(output)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
