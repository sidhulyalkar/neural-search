"""Select a stratified neuro-judge validation sample.

The output is evidence-packet JSONL for real LLM judging and human audit. This
does not create gold labels or overwrite qrels.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from neural_search.eval.neuro_judge.evidence_packet import EvidencePacket  # noqa: E402


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _resolve_existing(path: Path) -> Path:
    if path.exists():
        return path
    if path.name == "neuro_qrels_judgments.jsonl":
        fallback = path.with_name("neuro_qrels_judgments_mock.jsonl")
        if fallback.exists():
            print(f"[WARN] {path} missing; using {fallback}", file=sys.stderr)
            return fallback
    return path


def _key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("query_id", "")), str(row.get("dataset_id") or row.get("record_id") or "")


def _candidate_rank_index(candidates: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    ranks: dict[tuple[str, str], int] = {}
    for index, row in enumerate(candidates, start=1):
        key = (str(row.get("query_id", "")), str(row.get("dataset_id") or row.get("record_id") or ""))
        rank = int(row.get("rank") or row.get("min_rank") or index)
        if key[0] and key[1]:
            ranks[key] = min(rank, ranks.get(key, rank))
    return ranks


def _packet_categories(
    packet: EvidencePacket,
    judgment: dict[str, Any] | None,
    rank: int | None,
    include_high_impact: bool,
    include_missing_evidence: bool,
) -> set[str]:
    categories = {
        f"intent:{packet.query_intent or 'unknown'}",
        f"source:{packet.source_archive or 'unknown'}",
    }
    for modality in packet.dataset_modalities or ["unknown_modality"]:
        categories.add(f"modality:{modality}")
    for species in packet.dataset_species or ["unknown_species"]:
        categories.add(f"species:{species}")
    if any(region.lower() not in {"", "any"} for region in packet.expected_brain_regions):
        categories.add("region_specific_query")
    if any(term in packet.query_text.lower() for term in ("raw", "ap-band", "ap band", "spike sorting", "kilosort")):
        categories.add("raw_data_required")
    if packet.known_failure_warnings or packet.concept_hard_negative_conflicts:
        categories.add("warning_heavy")
    if rank is not None and rank <= 10:
        categories.add("high_ranked_candidate")
    if rank is not None and rank >= 50:
        categories.add("low_ranked_candidate")
    if judgment:
        label = int(judgment.get("label", 0))
        confidence = float(judgment.get("confidence", 0.0))
        if label >= 2 and judgment.get("required_dimensions_missing"):
            categories.add("label_ge_2_missing_evidence")
        if label >= 2 and judgment.get("missing_information"):
            categories.add("label_ge_2_missing_information")
        if 0.55 <= confidence <= 0.75:
            categories.add("near_threshold_confidence")
        if judgment.get("abstain_recommended"):
            categories.add("abstain_recommended")
        categories.add(f"label:{label}")
    if include_missing_evidence:
        if not packet.description:
            categories.add("missing_description")
        if not packet.dataset_modalities:
            categories.add("missing_modality")
        if not packet.dataset_species:
            categories.add("missing_species")
        if packet.has_raw_data is not True:
            categories.add("raw_data_absent_or_uncertain")
    if include_high_impact and rank is not None and rank <= 20:
        categories.add("high_impact_proxy_top20")
    return categories


def select_validation_sample(
    evidence_rows: list[dict[str, Any]],
    judgment_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    n: int,
    seed: int,
    require_diversity: bool,
    include_high_impact: bool,
    include_missing_evidence: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    judgments = {_key(row): row for row in judgment_rows}
    ranks = _candidate_rank_index(candidate_rows)

    items: list[dict[str, Any]] = []
    category_index: dict[str, list[int]] = defaultdict(list)
    for idx, raw in enumerate(evidence_rows):
        packet = EvidencePacket.model_validate(raw)
        key = (packet.query_id, packet.dataset_id)
        judgment = judgments.get(key)
        categories = _packet_categories(
            packet,
            judgment,
            ranks.get(key),
            include_high_impact=include_high_impact,
            include_missing_evidence=include_missing_evidence,
        )
        item = {
            "index": idx,
            "key": key,
            "packet": raw,
            "categories": sorted(categories),
            "rank": ranks.get(key),
            "judgment": judgment,
        }
        items.append(item)
        for category in categories:
            category_index[category].append(idx)

    selected: dict[int, str] = {}

    def add_index(index: int, reason: str) -> None:
        if len(selected) < n:
            selected.setdefault(index, reason)

    priority_categories = [
        "warning_heavy",
        "label_ge_2_missing_evidence",
        "label_ge_2_missing_information",
        "abstain_recommended",
        "near_threshold_confidence",
        "raw_data_required",
        "region_specific_query",
        "high_ranked_candidate",
        "low_ranked_candidate",
        "missing_description",
        "missing_modality",
        "missing_species",
        "raw_data_absent_or_uncertain",
    ]
    if include_high_impact:
        priority_categories.insert(0, "high_impact_proxy_top20")

    for category in priority_categories:
        indices = category_index.get(category, [])
        rng.shuffle(indices)
        for index in indices[: max(1, n // 20)]:
            add_index(index, category)

    if require_diversity:
        for category in sorted(category_index):
            indices = category_index[category]
            rng.shuffle(indices)
            if indices:
                add_index(indices[0], f"diversity:{category}")

    remaining = [item["index"] for item in items if item["index"] not in selected]
    rng.shuffle(remaining)
    for index in remaining:
        add_index(index, "random_fill")
        if len(selected) >= n:
            break

    selected_items = [items[index] for index in selected]
    selected_items.sort(key=lambda item: (str(item["key"][0]), str(item["key"][1])))
    output_rows: list[dict[str, Any]] = []
    category_counts: Counter = Counter()
    label_counts: Counter = Counter()
    for item in selected_items:
        row = dict(item["packet"])
        row["_selection_reason"] = selected[item["index"]]
        row["_selection_categories"] = item["categories"]
        if item["rank"] is not None:
            row["_candidate_rank"] = item["rank"]
        output_rows.append(row)
        category_counts.update(item["categories"])
        if item["judgment"]:
            label_counts[str(int(item["judgment"].get("label", 0)))] += 1

    summary = {
        "requested_n": n,
        "selected_n": len(output_rows),
        "seed": seed,
        "require_diversity": require_diversity,
        "include_high_impact": include_high_impact,
        "include_missing_evidence": include_missing_evidence,
        "category_counts": dict(category_counts.most_common()),
        "label_counts": dict(label_counts),
        "selection_reason_counts": dict(Counter(selected.values()).most_common()),
    }
    return output_rows, summary


def _render_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Neuro-Judge Validation Sample Summary",
        "",
        "> Validation sample only. These packets are for real LLM judging and human audit; they are not gold qrels.",
        "",
        f"- Requested n: `{summary['requested_n']}`",
        f"- Selected n: `{summary['selected_n']}`",
        f"- Seed: `{summary['seed']}`",
        f"- Require diversity: `{summary['require_diversity']}`",
        f"- Include high-impact proxy: `{summary['include_high_impact']}`",
        f"- Include missing evidence: `{summary['include_missing_evidence']}`",
        "",
        "## Label Counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in sorted(summary["label_counts"].items()):
        lines.append(f"| {label} | {count} |")
    lines += ["", "## Selection Reasons", "", "| Reason | Count |", "|---|---:|"]
    for reason, count in summary["selection_reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    lines += ["", "## Category Coverage", "", "| Category | Count |", "|---|---:|"]
    for category, count in summary["category_counts"].items():
        lines.append(f"| {category} | {count} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Select a neuro-judge validation sample.")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--n", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-diversity", action="store_true")
    parser.add_argument("--include-high-impact", action="store_true")
    parser.add_argument("--include-missing-evidence", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    evidence_path = _REPO / args.evidence if not args.evidence.is_absolute() else args.evidence
    judgments_path = _resolve_existing(
        _REPO / args.judgments if not args.judgments.is_absolute() else args.judgments
    )
    candidates_path = _REPO / args.candidates if not args.candidates.is_absolute() else args.candidates
    if not evidence_path.exists():
        sys.exit(f"[ERROR] Evidence file not found: {evidence_path}")
    if not judgments_path.exists():
        sys.exit(f"[ERROR] Judgments file not found: {judgments_path}")
    if not candidates_path.exists():
        sys.exit(f"[ERROR] Candidates file not found: {candidates_path}")

    rows, summary = select_validation_sample(
        _load_jsonl(evidence_path),
        _load_jsonl(judgments_path),
        _load_jsonl(candidates_path),
        n=args.n,
        seed=args.seed,
        require_diversity=args.require_diversity,
        include_high_impact=args.include_high_impact,
        include_missing_evidence=args.include_missing_evidence,
    )

    out_path = _REPO / args.out if not args.out.is_absolute() else args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    summary_path = _REPO / args.summary if not args.summary.is_absolute() else args.summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(_render_summary(summary), encoding="utf-8")

    print(json.dumps({"selected_n": summary["selected_n"], "seed": summary["seed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
