"""Cross-dataset pairing scorer for scientific compatibility.

Ranks dataset pairs by scientific compatibility for various use cases:
- Same task, different region
- Same task, different species
- Complementary modalities
- Shared analysis affordances

Usage:
    python -m neural_search.evaluation.cross_dataset_pairing
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.ontology import normalize_text

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


@dataclass
class PairingFeatures:
    """Features for a dataset pair."""

    task_similarity: float = 0.0
    modality_compatibility: float = 0.0
    species_relationship: float = 0.0
    region_overlap: float = 0.0
    shared_events: float = 0.0
    shared_affordances: float = 0.0
    provenance_confidence: float = 0.0
    novelty_bonus: float = 0.0
    incompatibility_penalty: float = 0.0


@dataclass
class DatasetPair:
    """A scored dataset pair."""

    dataset_a_id: str
    dataset_b_id: str
    pair_type: str
    compatibility_score: float
    features: PairingFeatures
    rationale: list[str]
    use_cases: list[str]


@dataclass
class CrossDatasetPairingReport:
    """Complete cross-dataset pairing report."""

    generated_at: str
    total_datasets: int
    total_pairs_evaluated: int
    top_pairs: list[DatasetPair]
    pairs_by_type: dict[str, list[DatasetPair]]
    summary: dict[str, Any]


PAIR_TYPES = (
    "same_task_different_region",
    "same_task_different_species",
    "same_modality_different_task",
    "shared_affordance",
    "complementary_modalities",
    "dataset_to_paper",
)


def _extract_labels(dataset: dict[str, Any], field_name: str) -> set[str]:
    """Extract normalized labels from dataset field."""
    values = dataset.get(field_name, [])
    if isinstance(values, str):
        values = [values]
    return {normalize_text(str(v)) for v in values}


def _jaccard(set_a: set[str], set_b: set[str]) -> float:
    """Compute Jaccard similarity."""
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _overlap_ratio(set_a: set[str], set_b: set[str]) -> float:
    """Compute overlap ratio (intersection / min size)."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    min_size = min(len(set_a), len(set_b))
    return intersection / min_size if min_size > 0 else 0.0


def compute_pairing_features(
    dataset_a: dict[str, Any],
    dataset_b: dict[str, Any],
) -> PairingFeatures:
    """Compute compatibility features for a dataset pair."""
    tasks_a = _extract_labels(dataset_a, "tasks")
    tasks_b = _extract_labels(dataset_b, "tasks")
    task_similarity = _jaccard(tasks_a, tasks_b)

    modalities_a = _extract_labels(dataset_a, "modalities")
    modalities_b = _extract_labels(dataset_b, "modalities")
    modality_compat = _overlap_ratio(modalities_a, modalities_b)

    species_a = _extract_labels(dataset_a, "species")
    species_b = _extract_labels(dataset_b, "species")
    species_rel = _jaccard(species_a, species_b)

    regions_a = _extract_labels(dataset_a, "brain_regions")
    regions_b = _extract_labels(dataset_b, "brain_regions")
    region_overlap = _overlap_ratio(regions_a, regions_b)

    events_a = _extract_labels(dataset_a, "behaviors") | _extract_labels(dataset_a, "behavioral_events")
    events_b = _extract_labels(dataset_b, "behaviors") | _extract_labels(dataset_b, "behavioral_events")
    shared_events = _jaccard(events_a, events_b)

    affordances_a = set(dataset_a.get("analysis_affordances", []))
    affordances_b = set(dataset_b.get("analysis_affordances", []))
    shared_affordances = _jaccard(affordances_a, affordances_b)

    # Novelty: bonus if datasets are from different sources
    source_a = dataset_a.get("source", "")
    source_b = dataset_b.get("source", "")
    novelty_bonus = 0.1 if source_a != source_b and source_a and source_b else 0.0

    # Incompatibility: penalty for obvious mismatches
    incompatibility = 0.0
    # No shared modalities at all
    if modalities_a and modalities_b and not (modalities_a & modalities_b):
        # Different modality families might still be compatible
        pass
    # Different species without taxonomic overlap
    if species_a and species_b and not (species_a & species_b):
        incompatibility += 0.1

    return PairingFeatures(
        task_similarity=task_similarity,
        modality_compatibility=modality_compat,
        species_relationship=species_rel,
        region_overlap=region_overlap,
        shared_events=shared_events,
        shared_affordances=shared_affordances,
        provenance_confidence=1.0,  # TODO: incorporate actual provenance
        novelty_bonus=novelty_bonus,
        incompatibility_penalty=incompatibility,
    )


def classify_pair_type(features: PairingFeatures, dataset_a: dict[str, Any], dataset_b: dict[str, Any]) -> str:
    """Classify the type of dataset pair."""
    # Same task, different region
    if features.task_similarity > 0.5 and features.region_overlap < 0.3:
        return "same_task_different_region"

    # Same task, different species
    if features.task_similarity > 0.5 and features.species_relationship < 0.3:
        return "same_task_different_species"

    # Same modality, different task
    if features.modality_compatibility > 0.5 and features.task_similarity < 0.3:
        return "same_modality_different_task"

    # Shared affordance
    if features.shared_affordances > 0.3:
        return "shared_affordance"

    # Complementary modalities (some overlap in tasks but different modalities)
    if features.task_similarity > 0.3 and features.modality_compatibility < 0.3:
        return "complementary_modalities"

    return "general_similarity"


def generate_rationale(features: PairingFeatures, pair_type: str) -> list[str]:
    """Generate human-readable rationale for pairing."""
    rationale = []

    if features.task_similarity > 0.5:
        rationale.append(f"Strong task overlap ({features.task_similarity:.0%})")
    elif features.task_similarity > 0.2:
        rationale.append(f"Partial task overlap ({features.task_similarity:.0%})")

    if features.modality_compatibility > 0.5:
        rationale.append(f"Compatible modalities ({features.modality_compatibility:.0%})")

    if features.region_overlap > 0.3:
        rationale.append(f"Shared brain regions ({features.region_overlap:.0%})")

    if features.shared_affordances > 0.3:
        rationale.append(f"Shared analysis affordances ({features.shared_affordances:.0%})")

    if features.novelty_bonus > 0:
        rationale.append("Different data sources (adds novelty)")

    if features.incompatibility_penalty > 0:
        rationale.append("Some compatibility concerns")

    return rationale


def generate_use_cases(pair_type: str, features: PairingFeatures) -> list[str]:
    """Generate suggested use cases for the pairing."""
    use_cases = {
        "same_task_different_region": [
            "Compare neural representations across regions",
            "Study region-specific task encoding",
            "Cross-region generalization analysis",
        ],
        "same_task_different_species": [
            "Cross-species comparison",
            "Evolutionary conservation analysis",
            "Species-specific neural coding",
        ],
        "same_modality_different_task": [
            "Modality-specific analysis methods",
            "Task-general neural dynamics",
            "Method validation across tasks",
        ],
        "shared_affordance": [
            "Method validation across datasets",
            "Analysis generalization testing",
            "Benchmark algorithm development",
        ],
        "complementary_modalities": [
            "Multi-modal integration",
            "Cross-validation of findings",
            "Complementary data fusion",
        ],
        "general_similarity": [
            "General comparison study",
            "Meta-analysis inclusion",
        ],
    }
    return use_cases.get(pair_type, ["Further investigation needed"])


def compute_compatibility_score(features: PairingFeatures) -> float:
    """Compute overall compatibility score."""
    score = (
        0.25 * features.task_similarity +
        0.15 * features.modality_compatibility +
        0.10 * features.species_relationship +
        0.15 * features.region_overlap +
        0.10 * features.shared_events +
        0.15 * features.shared_affordances +
        0.05 * features.provenance_confidence +
        features.novelty_bonus -
        features.incompatibility_penalty
    )
    return max(0.0, min(1.0, score))


def score_dataset_pair(
    dataset_a: dict[str, Any],
    dataset_b: dict[str, Any],
) -> DatasetPair:
    """Score a single dataset pair."""
    features = compute_pairing_features(dataset_a, dataset_b)
    pair_type = classify_pair_type(features, dataset_a, dataset_b)
    score = compute_compatibility_score(features)
    rationale = generate_rationale(features, pair_type)
    use_cases = generate_use_cases(pair_type, features)

    id_a = dataset_a.get("id", dataset_a.get("source_id", "unknown_a"))
    id_b = dataset_b.get("id", dataset_b.get("source_id", "unknown_b"))

    return DatasetPair(
        dataset_a_id=str(id_a),
        dataset_b_id=str(id_b),
        pair_type=pair_type,
        compatibility_score=score,
        features=features,
        rationale=rationale,
        use_cases=use_cases,
    )


def run_cross_dataset_pairing(
    datasets: list[dict[str, Any]] | None = None,
    top_k: int = 20,
) -> CrossDatasetPairingReport:
    """Run cross-dataset pairing analysis."""
    if datasets is None:
        datasets = build_demo_seed()

    # Extract dataset records
    records = []
    for item in datasets:
        ds = item.get("dataset", item)
        records.append(ds)

    # Score all pairs
    pairs: list[DatasetPair] = []
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            pair = score_dataset_pair(records[i], records[j])
            pairs.append(pair)

    # Sort by compatibility score
    pairs.sort(key=lambda p: p.compatibility_score, reverse=True)

    # Group by type
    pairs_by_type: dict[str, list[DatasetPair]] = {}
    for pair in pairs:
        if pair.pair_type not in pairs_by_type:
            pairs_by_type[pair.pair_type] = []
        pairs_by_type[pair.pair_type].append(pair)

    # Summary statistics
    summary = {
        "total_pairs": len(pairs),
        "mean_compatibility": sum(p.compatibility_score for p in pairs) / len(pairs) if pairs else 0.0,
        "pairs_per_type": {t: len(ps) for t, ps in pairs_by_type.items()},
        "top_pair": pairs[0].dataset_a_id + " <-> " + pairs[0].dataset_b_id if pairs else None,
        "recommendation": "Review top pairs for study design" if pairs else "No pairs to evaluate",
    }

    return CrossDatasetPairingReport(
        generated_at=datetime.now(UTC).isoformat(),
        total_datasets=len(records),
        total_pairs_evaluated=len(pairs),
        top_pairs=pairs[:top_k],
        pairs_by_type=pairs_by_type,
        summary=summary,
    )


def generate_pairing_markdown(report: CrossDatasetPairingReport) -> str:
    """Generate Markdown pairing report."""
    lines = [
        "# Cross-Dataset Pairing Report",
        "",
        f"Generated: {report.generated_at}",
        "",
        "## Summary",
        "",
        f"- **Total Datasets**: {report.total_datasets}",
        f"- **Total Pairs Evaluated**: {report.total_pairs_evaluated}",
        f"- **Mean Compatibility**: {report.summary.get('mean_compatibility', 0):.1%}",
        "",
    ]

    # Pairs by type
    lines.extend([
        "## Pairs by Type",
        "",
        "| Type | Count |",
        "| --- | --- |",
    ])
    for pair_type, pairs in report.pairs_by_type.items():
        lines.append(f"| {pair_type} | {len(pairs)} |")
    lines.append("")

    # Top pairs
    lines.extend([
        "## Top Dataset Pairs",
        "",
        "| Dataset A | Dataset B | Type | Score | Rationale |",
        "| --- | --- | --- | --- | --- |",
    ])

    for pair in report.top_pairs[:15]:
        rationale_short = "; ".join(pair.rationale[:2])
        lines.append(
            f"| {pair.dataset_a_id} | {pair.dataset_b_id} | "
            f"{pair.pair_type} | {pair.compatibility_score:.1%} | {rationale_short} |"
        )
    lines.append("")

    # Use case examples
    if report.top_pairs:
        lines.extend([
            "## Suggested Use Cases (Top Pair)",
            "",
        ])
        top = report.top_pairs[0]
        lines.append(f"**{top.dataset_a_id} <-> {top.dataset_b_id}**")
        lines.append("")
        for use_case in top.use_cases:
            lines.append(f"- {use_case}")
        lines.append("")

    return "\n".join(lines)


def generate_pairing_json(report: CrossDatasetPairingReport) -> str:
    """Generate JSON pairing report."""
    from dataclasses import asdict

    def serialize_pair(p: DatasetPair) -> dict[str, Any]:
        return {
            "dataset_a_id": p.dataset_a_id,
            "dataset_b_id": p.dataset_b_id,
            "pair_type": p.pair_type,
            "compatibility_score": p.compatibility_score,
            "features": asdict(p.features),
            "rationale": p.rationale,
            "use_cases": p.use_cases,
        }

    payload = {
        "generated_at": report.generated_at,
        "total_datasets": report.total_datasets,
        "total_pairs_evaluated": report.total_pairs_evaluated,
        "top_pairs": [serialize_pair(p) for p in report.top_pairs],
        "pairs_by_type": {
            t: [serialize_pair(p) for p in ps[:5]]  # Limit per type
            for t, ps in report.pairs_by_type.items()
        },
        "summary": report.summary,
    }
    return json.dumps(payload, indent=2, default=str)


def write_pairing_reports(
    report: CrossDatasetPairingReport,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Write pairing reports to files."""
    out = output_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "cross_dataset_pairing_report.md"
    json_path = out / "cross_dataset_pairing_report.json"

    md_path.write_text(generate_pairing_markdown(report), encoding="utf-8")
    json_path.write_text(generate_pairing_json(report), encoding="utf-8")

    return {"markdown": str(md_path), "json": str(json_path)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.evaluation.cross_dataset_pairing",
        description="Run cross-dataset pairing analysis.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top pairs to include.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for reports.",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON to stdout instead of writing files.",
    )
    args = parser.parse_args(argv)

    print("Running cross-dataset pairing analysis...")
    print()

    report = run_cross_dataset_pairing(top_k=args.top_k)

    if args.json_only:
        print(generate_pairing_json(report))
        return 0

    paths = write_pairing_reports(report, args.output_dir)

    # Print summary
    print("=" * 70)
    print("CROSS-DATASET PAIRING RESULTS")
    print("=" * 70)
    print()
    print(f"Total Datasets: {report.total_datasets}")
    print(f"Total Pairs: {report.total_pairs_evaluated}")
    print(f"Mean Compatibility: {report.summary.get('mean_compatibility', 0):.1%}")
    print()

    print("Pairs by Type:")
    for pair_type, pairs in report.pairs_by_type.items():
        print(f"  {pair_type}: {len(pairs)}")
    print()

    if report.top_pairs:
        print("Top 5 Pairs:")
        for pair in report.top_pairs[:5]:
            print(f"  {pair.dataset_a_id} <-> {pair.dataset_b_id}: {pair.compatibility_score:.1%}")
    print()

    print("=" * 70)
    print("Reports written to:")
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON: {paths['json']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
