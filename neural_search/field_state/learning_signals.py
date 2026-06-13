"""Learning signal generation from feedback, neuro-judge disagreements, and retrieval failures.

Learning signals update priors and curation queues — they do NOT overwrite canonical labels.
All signals carry provenance and are marked as provisional.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SIGNAL_VERSION = "v0.9.0"

VALID_SIGNAL_TYPES = frozenset({
    "false_high_candidate",
    "false_low_candidate",
    "missing_metadata_candidate",
    "hard_negative_candidate",
    "concept_link_candidate",
    "affordance_link_candidate",
    "source_quality_warning",
    "query_failure_mode",
    "dataset_reuse_success_signal",
})


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class LearningSignal:
    """A single learning signal record."""

    def __init__(
        self,
        signal_type: str,
        dataset_id: str,
        query_id: str | None = None,
        score: float = 0.5,
        evidence: list[str] | None = None,
        provenance: str = "learning_signal_pipeline",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if signal_type not in VALID_SIGNAL_TYPES:
            raise ValueError(f"Invalid signal type: {signal_type}")
        self.signal_type = signal_type
        self.dataset_id = dataset_id
        self.query_id = query_id
        self.score = score
        self.evidence = evidence or []
        self.provenance = provenance
        self.metadata = metadata or {}
        self.created_at = _utc_now()
        self.signal_version = SIGNAL_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "dataset_id": self.dataset_id,
            "query_id": self.query_id,
            "score": self.score,
            "evidence": self.evidence,
            "provenance": self.provenance,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "signal_version": self.signal_version,
        }

    def to_jsonl(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


class LearningSignalGenerator:
    """Generates learning signals from feedback, judgments, and retrieval data."""

    def __init__(self) -> None:
        self.signals: list[LearningSignal] = []

    # ------------------------------------------------------------------
    # From feedback
    # ------------------------------------------------------------------

    def process_feedback(self, feedback_records: list[dict[str, Any]]) -> None:
        """Convert feedback events into learning signals."""
        dataset_feedback: dict[str, list[dict]] = defaultdict(list)
        for fb in feedback_records:
            did = fb.get("dataset_id", "")
            if did:
                dataset_feedback[did].append(fb)

        for dataset_id, events in dataset_feedback.items():
            not_useful = sum(1 for e in events if e.get("usefulness") == "not_useful")
            useful = sum(1 for e in events if e.get("usefulness") == "useful")
            total = len(events)
            if total < 2:
                continue  # Not enough signal

            # Aggregate wrong-tag evidence
            wrong_tags: list[str] = []
            for ev in events:
                wrong_tags.extend(
                    t for t in ev.get("reason_tags", []) if t.startswith("wrong_")
                )
            missing_tags = [
                t for ev in events for t in ev.get("reason_tags", [])
                if t.startswith("missing_")
            ]

            if not_useful / total >= 0.6 and wrong_tags:
                self.signals.append(LearningSignal(
                    signal_type="false_high_candidate",
                    dataset_id=dataset_id,
                    score=not_useful / total,
                    evidence=list(set(wrong_tags))[:5],
                    provenance="user_feedback_aggregate",
                    metadata={"total_feedback": total, "not_useful_count": not_useful},
                ))

            if useful / total >= 0.7:
                self.signals.append(LearningSignal(
                    signal_type="dataset_reuse_success_signal",
                    dataset_id=dataset_id,
                    score=useful / total,
                    evidence=["high_useful_feedback_rate"],
                    provenance="user_feedback_aggregate",
                    metadata={"total_feedback": total, "useful_count": useful},
                ))

            if missing_tags:
                self.signals.append(LearningSignal(
                    signal_type="missing_metadata_candidate",
                    dataset_id=dataset_id,
                    score=len(missing_tags) / total,
                    evidence=list(set(missing_tags))[:5],
                    provenance="user_feedback_aggregate",
                ))

    # ------------------------------------------------------------------
    # From neuro-judge judgments
    # ------------------------------------------------------------------

    def process_judgments(
        self,
        judgments: list[dict[str, Any]],
        feedback_records: list[dict[str, Any]] | None = None,
    ) -> None:
        """Generate signals from judge outputs and optional judge/user disagreements."""
        fb_by_dataset: dict[str, list[dict]] = defaultdict(list)
        if feedback_records:
            for fb in feedback_records:
                did = fb.get("dataset_id", "")
                if did:
                    fb_by_dataset[did].append(fb)

        for jmt in judgments:
            # Guard: skip any human_gold
            if jmt.get("label_provenance") == "human_gold":
                continue

            dataset_id = jmt.get("dataset_id", "")
            query_id = jmt.get("query_id", "")
            label = jmt.get("label")
            if not dataset_id:
                continue

            abstain = jmt.get("abstain_recommended", False)
            hard_neg = jmt.get("hard_negative_detected", False)
            evidence_completeness = jmt.get("evidence_completeness", 1.0)

            if hard_neg and label is not None and label >= 2:
                self.signals.append(LearningSignal(
                    signal_type="hard_negative_candidate",
                    dataset_id=dataset_id,
                    query_id=query_id,
                    score=0.8,
                    evidence=["judge_hard_negative_detected"],
                    provenance="neuro_judge_silver",
                    metadata={"label": label, "hard_negative_detected": True},
                ))

            if evidence_completeness < 0.5 and not abstain:
                self.signals.append(LearningSignal(
                    signal_type="missing_metadata_candidate",
                    dataset_id=dataset_id,
                    query_id=query_id,
                    score=1.0 - evidence_completeness,
                    evidence=["low_evidence_completeness"],
                    provenance="neuro_judge_silver",
                    metadata={"evidence_completeness": evidence_completeness},
                ))

            # Disagreement: user says useful, judge says label=0 or 1
            user_events = fb_by_dataset.get(dataset_id, [])
            useful_user = sum(1 for ev in user_events if ev.get("usefulness") == "useful")
            if useful_user >= 2 and label is not None and label <= 1:
                self.signals.append(LearningSignal(
                    signal_type="false_low_candidate",
                    dataset_id=dataset_id,
                    query_id=query_id,
                    score=0.7,
                    evidence=["judge_user_disagreement"],
                    provenance="judge_feedback_disagreement",
                    metadata={"judge_label": label, "useful_feedback_count": useful_user},
                ))

    # ------------------------------------------------------------------
    # From retrieval failures (missing top results)
    # ------------------------------------------------------------------

    def process_retrieval_gaps(self, gaps: list[dict[str, Any]]) -> None:
        """Generate signals from known benchmark gaps."""
        for gap in gaps:
            dataset_id = gap.get("dataset_id", "")
            query_id = gap.get("query_id", "")
            if not dataset_id:
                continue
            self.signals.append(LearningSignal(
                signal_type="query_failure_mode",
                dataset_id=dataset_id,
                query_id=query_id,
                score=gap.get("gap_score", 0.5),
                evidence=gap.get("gap_evidence", []),
                provenance="benchmark_gap_analysis",
            ))

    # ------------------------------------------------------------------
    # Audit priority queue
    # ------------------------------------------------------------------

    def compute_audit_priority_queue(self) -> list[dict[str, Any]]:
        """Return datasets ranked by audit priority."""
        priority: dict[str, dict[str, Any]] = {}
        for sig in self.signals:
            did = sig.dataset_id
            if did not in priority:
                priority[did] = {
                    "dataset_id": did,
                    "priority_score": 0.0,
                    "signal_types": [],
                    "signals": [],
                }
            priority[did]["priority_score"] += sig.score
            if sig.signal_type not in priority[did]["signal_types"]:
                priority[did]["signal_types"].append(sig.signal_type)
            priority[did]["signals"].append(sig.to_dict())

        return sorted(priority.values(), key=lambda x: -x["priority_score"])

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_signals(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [sig.to_jsonl() for sig in self.signals]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return len(lines)

    def render_report(self) -> str:
        type_counts: dict[str, int] = defaultdict(int)
        for sig in self.signals:
            type_counts[sig.signal_type] += 1
        audit_queue = self.compute_audit_priority_queue()

        lines = ["# Learning Signals Report", ""]
        lines += [f"**Total signals:** {len(self.signals)}", ""]
        lines += ["## Signal Counts by Type"]
        for stype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            lines.append(f"- `{stype}`: {count}")
        lines += [""]
        lines += ["## Top Audit Targets (highest priority score)"]
        for entry in audit_queue[:10]:
            lines.append(
                f"- `{entry['dataset_id']}` — score {entry['priority_score']:.2f}"
                f", signals: {', '.join(entry['signal_types'])}"
            )
        lines += [""]
        lines += ["---"]
        lines += ["*All signals are provisional downstream signals, not human gold labels.*"]
        return "\n".join(lines)
