"""Diagnose neuro-judge label collapse and evidence gaps.

This script is diagnostic only. It never writes gold qrels and should be used
before scaling neuro-judge labels beyond a validation sample.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from neural_search.eval.neuro_judge.evidence_packet import EvidencePacket  # noqa: E402
from neural_search.eval.neuro_judge.judge import _dimension_evidence  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _resolve_existing(path: Path) -> Path:
    if path.exists():
        return path
    if path.name == "neuro_qrels_judgments.jsonl":
        fallback = path.with_name("neuro_qrels_judgments_mock.jsonl")
        if fallback.exists():
            print(f"[WARN] {path} missing; using {fallback}", file=sys.stderr)
            return fallback
    if path.name == "neuro_qrels_consensus.jsonl":
        fallback = path.with_name("neuro_qrels_consensus_mock.jsonl")
        if fallback.exists():
            print(f"[WARN] {path} missing; using {fallback}", file=sys.stderr)
            return fallback
    return path


def _bucket_confidence(records: list[dict[str, Any]]) -> dict[str, int]:
    buckets = {"<0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, ">=0.9": 0}
    for row in records:
        value = float(row.get("confidence", 0.0))
        if value < 0.5:
            buckets["<0.5"] += 1
        elif value < 0.7:
            buckets["0.5-0.7"] += 1
        elif value < 0.9:
            buckets["0.7-0.9"] += 1
        else:
            buckets[">=0.9"] += 1
    return buckets


def _label_distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {str(i): 0 for i in range(4)}
    for row in records:
        label = row.get("label", row.get("relevance"))
        if label is not None:
            counts[str(int(label))] = counts.get(str(int(label)), 0) + 1
    return counts


def _pct(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _evidence_stats(packets: list[EvidencePacket]) -> dict[str, Any]:
    total = len(packets)
    dimension_counts: dict[str, Counter] = {
        "species": Counter(),
        "modality": Counter(),
        "brain_region": Counter(),
        "task": Counter(),
        "affordance": Counter(),
        "raw_data": Counter(),
    }
    missing_expected: Counter = Counter()
    absent_fields: Counter = Counter()
    label2_reasons: Counter = Counter()
    hard_negative_warnings = 0
    processed_only = 0
    raw_present = 0
    raw_absent = 0

    for packet in packets:
        dims = _dimension_evidence(packet)
        for dim, (matched, status) in dims.items():
            if status == "required":
                if matched is True:
                    dimension_counts[dim]["matched"] += 1
                else:
                    dimension_counts[dim]["missing"] += 1
                    missing_expected[dim] += 1
            else:
                dimension_counts[dim]["not_required"] += 1
        if packet.has_raw_data is True:
            raw_present += 1
        else:
            raw_absent += 1
        if packet.has_processed_data and packet.has_raw_data is not True:
            processed_only += 1
        if packet.known_failure_warnings:
            hard_negative_warnings += 1
        if not packet.description:
            absent_fields["description"] += 1
        if not packet.dataset_species:
            absent_fields["dataset_species"] += 1
        if not packet.dataset_modalities:
            absent_fields["dataset_modalities"] += 1
        if not packet.dataset_brain_regions:
            absent_fields["dataset_brain_regions"] += 1
        if not packet.dataset_tasks:
            absent_fields["dataset_tasks"] += 1
        if not packet.affordance_matches:
            absent_fields["affordance_matches"] += 1
        if not packet.file_format_evidence:
            absent_fields["file_format_evidence"] += 1

    match_rates = {}
    for dim, counts in dimension_counts.items():
        required = counts["matched"] + counts["missing"]
        match_rates[dim] = {
            "required": required,
            "matched": counts["matched"],
            "missing": counts["missing"],
            "match_rate": _pct(counts["matched"], required),
        }

    return {
        "packet_count": total,
        "match_rates": match_rates,
        "missing_expected_dimension_counts": dict(missing_expected.most_common()),
        "absent_field_counts": dict(absent_fields.most_common()),
        "raw_data_present": raw_present,
        "raw_data_absent_or_uncertain": raw_absent,
        "processed_only_evidence_count": processed_only,
        "hard_negative_warning_count": hard_negative_warnings,
        "label2_reason_counts": dict(label2_reasons),
    }


def _judgment_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    label_dist = _label_distribution(records)
    total = sum(label_dist.values())
    failure_modes: Counter = Counter()
    missing_dimensions: Counter = Counter()
    label2_reasons: Counter = Counter()
    examples_by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    completeness_values: list[float] = []
    abstain_count = 0

    for row in records:
        for mode in row.get("failure_modes", []):
            failure_modes[str(mode)] += 1
        for dim in row.get("required_dimensions_missing", []):
            missing_dimensions[str(dim)] += 1
        completeness_values.append(float(row.get("evidence_completeness", 0.0)))
        if row.get("abstain_recommended"):
            abstain_count += 1
        if int(row.get("label", row.get("relevance", 0))) == 2:
            reasons = row.get("failure_modes") or row.get("required_dimensions_missing") or ["unknown"]
            reason = str(reasons[0])
            label2_reasons[reason] += 1
            if len(examples_by_reason[reason]) < 3:
                examples_by_reason[reason].append(
                    {
                        "query_id": row.get("query_id"),
                        "dataset_id": row.get("dataset_id"),
                        "confidence": row.get("confidence"),
                        "rationale_short": row.get("rationale_short", ""),
                        "missing": row.get("required_dimensions_missing", []),
                    }
                )

    dominant_label = max(label_dist, key=lambda key: label_dist[key]) if total else "none"
    dominant_fraction = _pct(label_dist.get(dominant_label, 0), total)
    return {
        "label_distribution": label_dist,
        "dominant_label": dominant_label,
        "dominant_label_fraction": dominant_fraction,
        "confidence_distribution": _bucket_confidence(records),
        "failure_mode_counts": dict(failure_modes.most_common()),
        "missing_dimension_counts_from_judgments": dict(missing_dimensions.most_common()),
        "label2_reason_counts": dict(label2_reasons.most_common()),
        "label2_examples_by_reason": examples_by_reason,
        "mean_evidence_completeness": round(sum(completeness_values) / len(completeness_values), 4)
        if completeness_values
        else 0.0,
        "abstain_recommended_count": abstain_count,
    }


def diagnose(
    evidence_rows: list[dict[str, Any]],
    judgment_rows: list[dict[str, Any]],
    consensus_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    packets = [EvidencePacket.model_validate(row) for row in evidence_rows]
    evidence = _evidence_stats(packets)
    judgments = _judgment_stats(judgment_rows)
    consensus = _judgment_stats(consensus_rows) if consensus_rows else {}

    label2_fraction = float(judgments.get("label_distribution", {}).get("2", 0)) / max(
        sum(judgments.get("label_distribution", {}).values()), 1
    )
    missing_evidence_total = sum(evidence["absent_field_counts"].values())
    required_missing_total = sum(evidence["missing_expected_dimension_counts"].values())
    if label2_fraction >= 0.7 and (missing_evidence_total or required_missing_total):
        likely_cause = "evidence_incompleteness_and_mock_thresholds"
    elif label2_fraction >= 0.7:
        likely_cause = "mock_thresholds_or_candidate_pool_homogeneity"
    else:
        likely_cause = "no_severe_label_collapse_detected"

    return {
        "watermark": "DIAGNOSTIC ONLY - NOT HUMAN GOLD",
        "evidence": evidence,
        "judgments": judgments,
        "consensus": consensus,
        "collapse_assessment": {
            "severe_label_collapse": label2_fraction >= 0.7,
            "label2_fraction": round(label2_fraction, 4),
            "likely_cause": likely_cause,
        },
    }


def _fmt_table(mapping: dict[str, Any], columns: tuple[str, str] = ("Item", "Count")) -> str:
    lines = [f"| {columns[0]} | {columns[1]} |", "|---|---:|"]
    for key, value in mapping.items():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines)


def render_markdown(summary: dict[str, Any]) -> str:
    evidence = summary["evidence"]
    judgments = summary["judgments"]
    assessment = summary["collapse_assessment"]
    lines = [
        "# Neuro-Judge Mock Collapse Diagnostics",
        "",
        "> DIAGNOSTIC ONLY - neuro-judge/mock labels are not human gold and must not be reported as scientific validation.",
        "",
        "## Collapse Assessment",
        "",
        f"- Severe label collapse: `{assessment['severe_label_collapse']}`",
        f"- Label-2 fraction: `{assessment['label2_fraction']}`",
        f"- Likely cause: `{assessment['likely_cause']}`",
        "",
        "## Label Distribution",
        "",
        _fmt_table(judgments["label_distribution"], ("Label", "Count")),
        "",
        "## Confidence Distribution",
        "",
        _fmt_table(judgments["confidence_distribution"], ("Confidence", "Count")),
        "",
        "## Evidence Match Rates",
        "",
        "| Dimension | Required | Matched | Missing | Match Rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for dim, stats in evidence["match_rates"].items():
        lines.append(
            f"| {dim} | {stats['required']} | {stats['matched']} | {stats['missing']} | {stats['match_rate']} |"
        )
    lines += [
        "",
        "## Raw And Processed Evidence",
        "",
        f"- Raw data present: `{evidence['raw_data_present']}`",
        f"- Raw data absent/uncertain: `{evidence['raw_data_absent_or_uncertain']}`",
        f"- Processed-only evidence count: `{evidence['processed_only_evidence_count']}`",
        f"- Hard-negative warning count: `{evidence['hard_negative_warning_count']}`",
        "",
        "## Missing Expected Dimensions",
        "",
        _fmt_table(evidence["missing_expected_dimension_counts"], ("Dimension", "Count")),
        "",
        "## Commonly Absent Evidence Fields",
        "",
        _fmt_table(evidence["absent_field_counts"], ("Field", "Count")),
        "",
        "## Label-2 Rule Reasons",
        "",
        _fmt_table(judgments["label2_reason_counts"], ("Reason", "Count")),
        "",
        "## Label-2 Examples By Reason",
        "",
    ]
    for reason, examples in judgments["label2_examples_by_reason"].items():
        lines.append(f"### {reason}")
        if not examples:
            lines.append("None.")
            continue
        for example in examples:
            lines.append(
                f"- `{example['query_id']}` / `{example['dataset_id']}` "
                f"confidence={example['confidence']} missing={example['missing']} "
                f"rationale={example['rationale_short']}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose neuro-judge label collapse.")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    evidence_path = _REPO / args.evidence if not args.evidence.is_absolute() else args.evidence
    judgments_path = _resolve_existing(
        _REPO / args.judgments if not args.judgments.is_absolute() else args.judgments
    )
    consensus_path = (
        _resolve_existing(_REPO / args.consensus if not args.consensus.is_absolute() else args.consensus)
        if args.consensus
        else None
    )

    if not evidence_path.exists():
        sys.exit(f"[ERROR] Evidence file not found: {evidence_path}")
    if not judgments_path.exists():
        sys.exit(f"[ERROR] Judgments file not found: {judgments_path}")

    summary = diagnose(
        _load_jsonl(evidence_path),
        _load_jsonl(judgments_path),
        _load_jsonl(consensus_path) if consensus_path and consensus_path.exists() else [],
    )

    out_path = _REPO / args.out if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_markdown(summary), encoding="utf-8")

    if args.json_out:
        json_path = _REPO / args.json_out if not args.json_out.is_absolute() else args.json_out
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["collapse_assessment"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
