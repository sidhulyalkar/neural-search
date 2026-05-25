"""Promotion gates for enabling search intelligence by default."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from neural_search.intelligence.review import (
    load_relevance_judgments,
    summarize_relevance_judgments,
)


@dataclass(frozen=True)
class IntentPromotionDecision:
    """Promotion decision for one planner intent."""

    intent: str
    ready: bool
    enabled_in_manifest: bool
    blockers: tuple[str, ...]
    metrics: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "ready": self.ready,
            "enabled_in_manifest": self.enabled_in_manifest,
            "blockers": list(self.blockers),
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class PromotionGateReport:
    """Promotion gate report for search intelligence."""

    default_enabled: bool
    promotion_ready: bool
    blockers: tuple[str, ...]
    human_label_summary: dict[str, Any]
    intent_decisions: tuple[IntentPromotionDecision, ...]

    def model_dump(self) -> dict[str, Any]:
        return {
            "default_enabled": self.default_enabled,
            "promotion_ready": self.promotion_ready,
            "blockers": list(self.blockers),
            "human_label_summary": dict(self.human_label_summary),
            "intent_decisions": [
                decision.model_dump() for decision in self.intent_decisions
            ],
        }


def load_promotion_manifest(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_evaluation_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _intent_decision(
    intent: str,
    gate: dict[str, Any],
    summary: dict[str, Any] | None,
) -> IntentPromotionDecision:
    blockers: list[str] = []
    enabled = bool(gate.get("enabled", False))
    if summary is None:
        blockers.append("missing evaluation summary")
        summary = {}

    query_count = int(summary.get("query_count", 0) or 0)
    min_query_count = int(gate.get("min_query_count", 1) or 1)
    if query_count < min_query_count:
        blockers.append(f"query_count {query_count} < required {min_query_count}")

    mean_mrr_delta = float(summary.get("mean_mrr_delta", 0.0) or 0.0)
    min_mean_mrr_delta = float(gate.get("min_mean_mrr_delta", 0.0) or 0.0)
    if mean_mrr_delta < min_mean_mrr_delta:
        blockers.append(
            f"mean_mrr_delta {mean_mrr_delta} < required {min_mean_mrr_delta}"
        )

    hard_negative_delta = int(summary.get("hard_negative_violation_delta", 0) or 0)
    max_hard_negative_delta = int(
        gate.get("max_hard_negative_violation_delta", 0) or 0
    )
    if hard_negative_delta > max_hard_negative_delta:
        blockers.append(
            "hard_negative_violation_delta "
            f"{hard_negative_delta} > allowed {max_hard_negative_delta}"
        )

    if not enabled:
        blockers.append("intent disabled in manifest")

    return IntentPromotionDecision(
        intent=intent,
        ready=not blockers,
        enabled_in_manifest=enabled,
        blockers=tuple(blockers),
        metrics={
            "query_count": query_count,
            "mean_mrr_delta": mean_mrr_delta,
            "hard_negative_violation_delta": hard_negative_delta,
        },
    )


def evaluate_promotion_gates(
    evaluation_report: dict[str, Any],
    manifest: dict[str, Any],
    *,
    human_label_summary: dict[str, Any] | None = None,
) -> PromotionGateReport:
    """Evaluate whether planner intents are ready for default promotion."""

    global_gates = manifest.get("global_gates", {})
    human_gates = manifest.get("human_label_gates", {})
    human_summary = dict(human_label_summary or {})
    blockers: list[str] = []
    total_queries = int(evaluation_report.get("query_count", 0) or 0)
    min_total_queries = int(global_gates.get("min_total_queries", 1) or 1)
    if total_queries < min_total_queries:
        blockers.append(f"total query_count {total_queries} < required {min_total_queries}")

    hard_negative_delta = int(
        evaluation_report.get("mean_delta", {}).get("hard_negative_violations", 0) or 0
    )
    max_hard_negative_delta = int(
        global_gates.get("max_hard_negative_violation_delta", 0) or 0
    )
    if hard_negative_delta > max_hard_negative_delta:
        blockers.append(
            "global hard_negative_violation_delta "
            f"{hard_negative_delta} > allowed {max_hard_negative_delta}"
        )

    min_human_judgments = int(human_gates.get("min_judgments", 0) or 0)
    judgment_count = int(human_summary.get("judgment_count", 0) or 0)
    if judgment_count < min_human_judgments:
        blockers.append(
            f"human judgment_count {judgment_count} < required {min_human_judgments}"
        )

    max_human_hard_negatives = int(
        human_gates.get("max_hard_negative_count", 0) or 0
    )
    human_hard_negatives = int(human_summary.get("hard_negative_count", 0) or 0)
    if human_hard_negatives > max_human_hard_negatives:
        blockers.append(
            f"human hard_negative_count {human_hard_negatives} > allowed "
            f"{max_human_hard_negatives}"
        )

    grouped = evaluation_report.get("grouped_by_intent", {})
    decisions = tuple(
        _intent_decision(intent, dict(gate), grouped.get(intent))
        for intent, gate in sorted((manifest.get("intents", {}) or {}).items())
    )
    default_enabled = bool(manifest.get("default_enabled", False))
    if not default_enabled:
        blockers.append("default promotion disabled in manifest")

    return PromotionGateReport(
        default_enabled=default_enabled,
        promotion_ready=not blockers and all(decision.ready for decision in decisions),
        blockers=tuple(blockers),
        human_label_summary=human_summary,
        intent_decisions=decisions,
    )


def _markdown(report: PromotionGateReport) -> str:
    lines = [
        "# Search Intelligence Promotion Gates",
        "",
        f"- Default enabled: {str(report.default_enabled).lower()}",
        f"- Promotion ready: {str(report.promotion_ready).lower()}",
        f"- Human judgments: {report.human_label_summary.get('judgment_count', 0)}",
        "",
        "## Global Blockers",
        "",
    ]
    if report.blockers:
        lines.extend(f"- {blocker}" for blocker in report.blockers)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Intent Decisions",
            "",
            "| Intent | Ready | Enabled | Query Count | MRR Delta | Hard Neg Delta | Blockers |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    for decision in report.intent_decisions:
        lines.append(
            "| "
            + " | ".join(
                [
                    decision.intent,
                    str(decision.ready).lower(),
                    str(decision.enabled_in_manifest).lower(),
                    str(decision.metrics.get("query_count", 0)),
                    str(decision.metrics.get("mean_mrr_delta", 0.0)),
                    str(decision.metrics.get("hard_negative_violation_delta", 0)),
                    "; ".join(decision.blockers) or "None",
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def write_promotion_gate_report(
    report: PromotionGateReport,
    output_dir: str | Path,
) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "promotion_gate_report.json"
    md_path = out / "promotion_gate_report.md"
    json_path.write_text(
        json.dumps(report.model_dump(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    md_path.write_text(_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate search intelligence promotion gates."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--judgments", help="Optional human relevance judgments JSONL.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--fail-on-blockers", action="store_true")
    args = parser.parse_args(argv)

    human_summary = (
        summarize_relevance_judgments(load_relevance_judgments(args.judgments))
        if args.judgments
        else None
    )
    report = evaluate_promotion_gates(
        load_evaluation_report(args.evaluation),
        load_promotion_manifest(args.manifest),
        human_label_summary=human_summary,
    )
    print(
        json.dumps(
            write_promotion_gate_report(report, args.out),
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_blockers and not report.promotion_ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
