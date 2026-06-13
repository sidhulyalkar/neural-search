#!/usr/bin/env python3
"""Build dataset organization views from neuro-judge judgments and corpus records.

Groups datasets into interpretive tiers based on judge labels, evidence completeness,
metadata quality, and raw-data suitability. Output is intended for frontend display
and human review triage — NOT a substitute for expert annotation.

Usage::

    python scripts/eval/build_dataset_organization_views.py \
        --judgments artifacts/field_state/neuro_qrels_judgments_mock.jsonl \
        --corpus-manifest artifacts/corpus_manifest.json \
        --out-json artifacts/field_state/dataset_organization_views.json \
        --out-md reports/eval/dataset_organization_views.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_JUDGMENTS = ROOT / "artifacts/field_state/neuro_qrels_judgments_mock.jsonl"
DEFAULT_MANIFEST = ROOT / "artifacts/corpus_manifest.json"
DEFAULT_EVIDENCE = ROOT / "artifacts/field_state/neuro_judge_evidence_packets.jsonl"
DEFAULT_JSON_OUT = ROOT / "artifacts/field_state/dataset_organization_views.json"
DEFAULT_MD_OUT = ROOT / "reports/eval/dataset_organization_views.md"

DISCLAIMER = (
    "IMPORTANT: These views are derived from preliminary neuro-judge labels "
    "(RAG-grounded LLM outputs). They are NOT human-annotated ground truth. "
    "Use for exploration and triage only. Expert audit required for any "
    "scientific reporting."
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


def load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def _per_dataset_best(judgments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """For each dataset, keep the highest-label, highest-confidence judgment."""
    best: dict[str, dict[str, Any]] = {}
    for j in judgments:
        did = str(j.get("dataset_id") or "")
        if not did:
            continue
        label = int(j.get("label") or 0)
        confidence = float(j.get("confidence") or 0.0)
        prev = best.get(did)
        if prev is None:
            best[did] = j
        else:
            prev_label = int(prev.get("label") or 0)
            prev_conf = float(prev.get("confidence") or 0.0)
            if (label, confidence) > (prev_label, prev_conf):
                best[did] = j
    return best


def _modality_bucket(modalities: list[str]) -> str:
    mods = {m.lower().replace(" ", "_") for m in (modalities or [])}
    if "extracellular_ephys" in mods or "ephys" in mods or "electrophysiology" in mods:
        return "electrophysiology"
    if "calcium_imaging" in mods or "two_photon" in mods or "fluorescence" in mods:
        return "calcium_imaging"
    if "fmri" in mods or "bold" in mods or "mri" in mods:
        return "fmri_mri"
    if "eeg" in mods or "meg" in mods:
        return "eeg_meg"
    if "behavior" in mods or "behaviour" in mods:
        return "behavior_only"
    return "other"


def build_views(
    judgments: list[dict[str, Any]],
    manifest: dict[str, Any],
    evidence_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    best = _per_dataset_best(judgments)

    # Build dataset metadata index from corpus manifest
    meta_index: dict[str, dict[str, Any]] = {}
    records = manifest.get("datasets") or manifest.get("records") or []
    if isinstance(records, list):
        for rec in records:
            did = str(rec.get("id") or rec.get("dataset_id") or "")
            if did:
                meta_index[did] = rec

    # Build evidence packet index
    ep_index: dict[str, dict[str, Any]] = {}
    for ep in evidence_packets:
        did = str(ep.get("dataset_id") or "")
        if did:
            ep_index[did] = ep

    # Containers for views
    highly_relevant: list[dict[str, Any]] = []
    useful_with_caveats: list[dict[str, Any]] = []
    weakly_related: list[dict[str, Any]] = []
    not_relevant: list[dict[str, Any]] = []
    missing_raw_data: list[dict[str, Any]] = []
    missing_species_modality: list[dict[str, Any]] = []
    likely_hard_negatives: list[dict[str, Any]] = []
    needs_human_audit: list[dict[str, Any]] = []
    raw_data_suitable: list[dict[str, Any]] = []
    processed_only: list[dict[str, Any]] = []
    high_evidence_completeness: list[dict[str, Any]] = []
    low_evidence_completeness: list[dict[str, Any]] = []
    abstain_flagged: list[dict[str, Any]] = []
    modality_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for did, judgment in best.items():
        label = int(judgment.get("label") or 0)
        confidence = float(judgment.get("confidence") or 0.0)
        ec = float(judgment.get("evidence_completeness") or 0.0)
        hn = bool(judgment.get("hard_negative_detected"))
        abstain = bool(judgment.get("abstain_recommended"))
        present = list(judgment.get("required_dimensions_present") or [])
        missing_dims = list(judgment.get("required_dimensions_missing") or [])
        failure_modes = list(judgment.get("failure_modes") or [])

        ep = ep_index.get(did, {})
        has_raw = ep.get("has_raw_data")
        has_processed = ep.get("has_processed_data")
        species = ep.get("dataset_species") or []
        modalities = ep.get("dataset_modalities") or []
        meta = meta_index.get(did, {})
        title = meta.get("title") or ep.get("title") or did

        entry: dict[str, Any] = {
            "dataset_id": did,
            "title": title,
            "label": label,
            "confidence": confidence,
            "evidence_completeness": ec,
            "hard_negative_detected": hn,
            "abstain_recommended": abstain,
            "required_dimensions_present": present,
            "required_dimensions_missing": missing_dims,
            "failure_modes": failure_modes,
            "has_raw_data": has_raw,
            "has_processed_data": has_processed,
            "species": species,
            "modalities": modalities,
            "source": meta.get("source") or ep.get("source_archive") or "unknown",
        }

        # Relevance tier buckets
        if label == 3:
            highly_relevant.append(entry)
        elif label == 2:
            useful_with_caveats.append(entry)
        elif label == 1:
            weakly_related.append(entry)
        else:
            not_relevant.append(entry)

        # Special property buckets
        if hn:
            likely_hard_negatives.append(entry)
        if abstain and label >= 2:
            needs_human_audit.append(entry)
        if has_raw is True:
            raw_data_suitable.append(entry)
        elif has_processed is True and has_raw is not True:
            processed_only.append(entry)
        if "species" in missing_dims or "modality" in missing_dims:
            missing_species_modality.append(entry)
        if label >= 2 and any("raw" in fm for fm in failure_modes):
            missing_raw_data.append(entry)
        if ec >= 0.75:
            high_evidence_completeness.append(entry)
        elif ec < 0.4:
            low_evidence_completeness.append(entry)
        if abstain:
            abstain_flagged.append(entry)

        # Modality buckets
        bucket = _modality_bucket(modalities)
        modality_buckets[bucket].append(entry)

    def sort_by_label_confidence(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(items, key=lambda x: (-x["label"], -x["confidence"]))

    views = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "disclaimer": DISCLAIMER,
        "label_provenance": "neuro_judge_silver",
        "total_datasets_judged": len(best),
        "relevance_tiers": {
            "highly_relevant": sort_by_label_confidence(highly_relevant),
            "useful_with_caveats": sort_by_label_confidence(useful_with_caveats),
            "weakly_related": sort_by_label_confidence(weakly_related),
            "not_relevant": sort_by_label_confidence(not_relevant),
        },
        "special_views": {
            "missing_raw_data": sort_by_label_confidence(missing_raw_data),
            "missing_species_or_modality_metadata": missing_species_modality,
            "likely_hard_negatives": likely_hard_negatives,
            "needs_human_audit": sort_by_label_confidence(needs_human_audit),
            "raw_data_suitable": sort_by_label_confidence(raw_data_suitable),
            "processed_only": sort_by_label_confidence(processed_only),
            "high_evidence_completeness": sort_by_label_confidence(high_evidence_completeness),
            "low_evidence_completeness": low_evidence_completeness,
            "abstain_flagged": sort_by_label_confidence(abstain_flagged),
        },
        "modality_buckets": {
            k: sort_by_label_confidence(v) for k, v in sorted(modality_buckets.items())
        },
    }
    return views


def _summary_table(
    entries: list[dict[str, Any]], max_rows: int = 10
) -> str:
    if not entries:
        return "_none_\n"
    rows = ["| Dataset ID | Title | Label | Confidence | EC |"]
    rows.append("|---|---|---:|---:|---:|")
    for e in entries[:max_rows]:
        rows.append(
            f"| {e['dataset_id']} "
            f"| {str(e.get('title', ''))[:60]} "
            f"| {e['label']} "
            f"| {e['confidence']:.2f} "
            f"| {e['evidence_completeness']:.2f} |"
        )
    if len(entries) > max_rows:
        rows.append(f"\n_...and {len(entries) - max_rows} more_")
    return "\n".join(rows) + "\n"


def render_markdown(views: dict[str, Any]) -> str:
    tiers = views["relevance_tiers"]
    special = views["special_views"]
    modality = views["modality_buckets"]
    now = views["generated_at"]
    total = views["total_datasets_judged"]

    lines: list[str] = [
        "# Dataset Organization Views",
        "",
        f"Generated: {now}",
        "",
        f"> {views['disclaimer']}",
        "",
        f"**Total datasets with judge labels:** {total}",
        "",
        "## Relevance Tiers",
        "",
    ]

    tier_labels = {
        "highly_relevant": "Highly Relevant (label=3)",
        "useful_with_caveats": "Useful with Caveats (label=2)",
        "weakly_related": "Weakly Related (label=1)",
        "not_relevant": "Not Relevant (label=0)",
    }
    for key, heading in tier_labels.items():
        entries = tiers.get(key) or []
        lines += [f"### {heading} — {len(entries)} datasets", "", _summary_table(entries)]

    lines += ["## Special Views", ""]
    special_labels = {
        "missing_raw_data": "Missing Raw Data",
        "missing_species_or_modality_metadata": "Missing Species/Modality Metadata",
        "likely_hard_negatives": "Likely Hard Negatives",
        "needs_human_audit": "Needs Human Audit (abstain + label≥2)",
        "raw_data_suitable": "Raw-Data Suitable",
        "processed_only": "Processed Only",
        "high_evidence_completeness": "High Evidence Completeness (≥0.75)",
        "low_evidence_completeness": "Low Evidence Completeness (<0.40)",
        "abstain_flagged": "Abstain Flagged",
    }
    for key, heading in special_labels.items():
        entries = special.get(key) or []
        lines += [f"### {heading} — {len(entries)} datasets", "", _summary_table(entries)]

    lines += ["## Modality Buckets", ""]
    for bucket, entries in sorted(modality.items()):
        lines += [f"### {bucket} — {len(entries)} datasets", "", _summary_table(entries)]

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build dataset organization views")
    parser.add_argument(
        "--judgments",
        type=Path,
        default=DEFAULT_JUDGMENTS,
        help="JSONL of neuro-judge judgments",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="JSON corpus manifest",
    )
    parser.add_argument(
        "--evidence-packets",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help="JSONL evidence packets",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_JSON_OUT,
        help="Output JSON path",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_MD_OUT,
        help="Output Markdown path",
    )
    args = parser.parse_args(argv)

    print(f"Loading judgments from {args.judgments}")
    judgments = load_jsonl(args.judgments)
    print(f"  {len(judgments)} judgment records loaded")

    print(f"Loading corpus manifest from {args.corpus_manifest}")
    manifest = load_json(args.corpus_manifest)

    print(f"Loading evidence packets from {args.evidence_packets}")
    evidence_packets = load_jsonl(args.evidence_packets)
    print(f"  {len(evidence_packets)} evidence packets loaded")

    views = build_views(judgments, manifest, evidence_packets)
    total = views["total_datasets_judged"]
    print(f"Built views for {total} datasets")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as fh:
        json.dump(views, fh, indent=2)
    print(f"Wrote {args.out_json}")

    md = render_markdown(views)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {args.out_md}")


if __name__ == "__main__":
    main()
