"""Metadata robustness experiments.

Measures retrieval degradation under metadata perturbations.

Usage:
    python -m neural_search.evaluation.metadata_robustness
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.evaluation.run_benchmark import (
    benchmark_path_for_suite,
    run_full_benchmark,
)
from neural_search.ingestion.demo_seed import build_demo_seed

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


PERTURBATION_TYPES = (
    "drop_description",
    "drop_task_labels",
    "drop_modality_labels",
    "drop_brain_regions",
    "remove_paper_links",
    "corrupt_synonyms",
    "remove_species",
    "inject_ambiguity",
)


@dataclass
class PerturbationConfig:
    """Configuration for a perturbation."""

    name: str
    description: str
    severity: str  # "low", "medium", "high"
    fields_affected: list[str]


PERTURBATION_CONFIGS = {
    "drop_description": PerturbationConfig(
        name="drop_description",
        description="Remove description field",
        severity="high",
        fields_affected=["description"],
    ),
    "drop_task_labels": PerturbationConfig(
        name="drop_task_labels",
        description="Remove task annotations",
        severity="medium",
        fields_affected=["tasks"],
    ),
    "drop_modality_labels": PerturbationConfig(
        name="drop_modality_labels",
        description="Remove modality annotations",
        severity="medium",
        fields_affected=["modalities"],
    ),
    "drop_brain_regions": PerturbationConfig(
        name="drop_brain_regions",
        description="Remove brain region annotations",
        severity="medium",
        fields_affected=["brain_regions"],
    ),
    "remove_paper_links": PerturbationConfig(
        name="remove_paper_links",
        description="Remove DOI/paper associations",
        severity="low",
        fields_affected=["doi", "paper_id", "linked_papers"],
    ),
    "corrupt_synonyms": PerturbationConfig(
        name="corrupt_synonyms",
        description="Replace terms with synonyms",
        severity="low",
        fields_affected=["tasks", "modalities"],
    ),
    "remove_species": PerturbationConfig(
        name="remove_species",
        description="Remove species information",
        severity="medium",
        fields_affected=["species"],
    ),
    "inject_ambiguity": PerturbationConfig(
        name="inject_ambiguity",
        description="Add ambiguous terms",
        severity="low",
        fields_affected=["description"],
    ),
}


def apply_perturbation(
    datasets: list[dict[str, Any]],
    perturbation: str,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Apply a perturbation to a copy of the datasets."""
    random.seed(seed)
    perturbed = copy.deepcopy(datasets)

    for item in perturbed:
        ds = item.get("dataset", item)

        if perturbation == "drop_description":
            ds["description"] = ""

        elif perturbation == "drop_task_labels":
            ds["tasks"] = []

        elif perturbation == "drop_modality_labels":
            ds["modalities"] = []

        elif perturbation == "drop_brain_regions":
            ds["brain_regions"] = []

        elif perturbation == "remove_paper_links":
            ds["doi"] = None
            ds["paper_id"] = None
            ds.pop("linked_papers", None)

        elif perturbation == "corrupt_synonyms":
            # Replace some terms with synonyms
            synonym_map = {
                "neuropixels": "high_density_ephys",
                "go_nogo": "go_no_go_task",
                "reversal_learning": "reversal_paradigm",
                "calcium_imaging": "gcamp_imaging",
            }
            for field_name in ["tasks", "modalities"]:
                values = ds.get(field_name, [])
                corrupted = []
                for v in values:
                    if isinstance(v, str) and v.lower() in synonym_map:
                        corrupted.append(synonym_map[v.lower()])
                    else:
                        corrupted.append(v)
                ds[field_name] = corrupted

        elif perturbation == "remove_species":
            ds["species"] = []

        elif perturbation == "inject_ambiguity":
            # Add ambiguous terms to description
            ambiguous_terms = [
                "neural data",
                "brain recordings",
                "behavior study",
            ]
            desc = ds.get("description", "")
            if desc:
                ds["description"] = desc + " " + random.choice(ambiguous_terms)

    return perturbed


@dataclass
class PerturbationResult:
    """Result of a single perturbation experiment."""

    perturbation: str
    config: PerturbationConfig
    original_metrics: dict[str, float]
    perturbed_metrics: dict[str, float]
    degradation: dict[str, float]
    severity_rating: str


@dataclass
class MetadataRobustnessReport:
    """Complete metadata robustness report."""

    generated_at: str
    suite: str
    seed: int
    total_perturbations: int
    results: list[PerturbationResult]
    most_impactful: str
    least_impactful: str
    summary: dict[str, Any]


def compute_degradation(
    original: dict[str, float],
    perturbed: dict[str, float],
) -> dict[str, float]:
    """Compute metric degradation as (original - perturbed) / original."""
    degradation = {}
    for metric in original:
        orig_val = original[metric]
        pert_val = perturbed.get(metric, 0.0)
        if orig_val > 0:
            degradation[metric] = (orig_val - pert_val) / orig_val
        else:
            degradation[metric] = 0.0
    return degradation


def classify_severity(degradation: dict[str, float]) -> str:
    """Classify severity based on precision degradation."""
    p5_deg = degradation.get("mean_precision_at_5", 0.0)
    if p5_deg > 0.3:
        return "critical"
    elif p5_deg > 0.15:
        return "high"
    elif p5_deg > 0.05:
        return "medium"
    else:
        return "low"


def run_metadata_robustness(
    suite: str = "demo_v02",
    perturbations: list[str] | None = None,
    seed: int = 42,
) -> MetadataRobustnessReport:
    """Run metadata robustness experiments."""
    if perturbations is None:
        perturbations = list(PERTURBATION_TYPES)

    benchmark_path = benchmark_path_for_suite(suite)
    original_datasets = build_demo_seed()

    # Run original benchmark
    original_report = run_full_benchmark(benchmark_path, datasets=original_datasets, suite=suite)
    original_metrics = {
        "mean_precision_at_5": original_report.mean_precision_at_5,
        "mean_recall_at_10": original_report.mean_recall_at_10,
        "mean_mrr": original_report.mean_mrr,
        "mean_ndcg_at_10": original_report.mean_ndcg_at_10,
    }

    results: list[PerturbationResult] = []

    for perturbation in perturbations:
        config = PERTURBATION_CONFIGS.get(
            perturbation,
            PerturbationConfig(
                name=perturbation,
                description=perturbation,
                severity="unknown",
                fields_affected=[],
            ),
        )

        # Apply perturbation
        perturbed_datasets = apply_perturbation(original_datasets, perturbation, seed)

        # Run perturbed benchmark
        perturbed_report = run_full_benchmark(
            benchmark_path, datasets=perturbed_datasets, suite=suite
        )
        perturbed_metrics = {
            "mean_precision_at_5": perturbed_report.mean_precision_at_5,
            "mean_recall_at_10": perturbed_report.mean_recall_at_10,
            "mean_mrr": perturbed_report.mean_mrr,
            "mean_ndcg_at_10": perturbed_report.mean_ndcg_at_10,
        }

        degradation = compute_degradation(original_metrics, perturbed_metrics)
        severity = classify_severity(degradation)

        results.append(
            PerturbationResult(
                perturbation=perturbation,
                config=config,
                original_metrics=original_metrics,
                perturbed_metrics=perturbed_metrics,
                degradation=degradation,
                severity_rating=severity,
            )
        )

    # Sort by impact (precision degradation)
    results.sort(
        key=lambda r: r.degradation.get("mean_precision_at_5", 0.0), reverse=True
    )

    most_impactful = results[0].perturbation if results else "none"
    least_impactful = results[-1].perturbation if results else "none"

    summary = {
        "total_perturbations": len(results),
        "critical_perturbations": [r.perturbation for r in results if r.severity_rating == "critical"],
        "high_impact_perturbations": [r.perturbation for r in results if r.severity_rating == "high"],
        "recommendation": f"Focus robustness on: {most_impactful}" if results else "No experiments run",
    }

    return MetadataRobustnessReport(
        generated_at=datetime.now(UTC).isoformat(),
        suite=suite,
        seed=seed,
        total_perturbations=len(results),
        results=results,
        most_impactful=most_impactful,
        least_impactful=least_impactful,
        summary=summary,
    )


def generate_robustness_markdown(report: MetadataRobustnessReport) -> str:
    """Generate Markdown robustness report."""
    lines = [
        "# Metadata Robustness Report",
        "",
        f"Generated: {report.generated_at}",
        f"Suite: {report.suite}",
        f"Random seed: {report.seed}",
        "",
        "## Summary",
        "",
        f"- **Total Perturbations**: {report.total_perturbations}",
        f"- **Most Impactful**: {report.most_impactful}",
        f"- **Least Impactful**: {report.least_impactful}",
        "",
    ]

    # Degradation table
    lines.extend([
        "## Degradation by Perturbation",
        "",
        "| Perturbation | P@5 Degradation | MRR Degradation | Severity |",
        "| --- | --- | --- | --- |",
    ])

    for r in report.results:
        p5_deg = r.degradation.get("mean_precision_at_5", 0.0)
        mrr_deg = r.degradation.get("mean_mrr", 0.0)
        lines.append(
            f"| {r.perturbation} | {p5_deg:+.1%} | {mrr_deg:+.1%} | {r.severity_rating} |"
        )
    lines.append("")

    # Detailed results
    lines.extend(["## Detailed Results", ""])
    for r in report.results:
        lines.extend([
            f"### {r.perturbation}",
            "",
            f"*{r.config.description}*",
            "",
            f"- Fields affected: {', '.join(r.config.fields_affected)}",
            f"- Original P@5: {r.original_metrics.get('mean_precision_at_5', 0):.1%}",
            f"- Perturbed P@5: {r.perturbed_metrics.get('mean_precision_at_5', 0):.1%}",
            f"- Degradation: {r.degradation.get('mean_precision_at_5', 0):+.1%}",
            "",
        ])

    # Recommendations
    if report.summary.get("critical_perturbations"):
        lines.extend([
            "## Recommendations",
            "",
            "Critical perturbations that severely impact retrieval:",
            "",
        ])
        for p in report.summary["critical_perturbations"]:
            lines.append(f"- {p}")
        lines.append("")

    return "\n".join(lines)


def generate_robustness_json(report: MetadataRobustnessReport) -> str:
    """Generate JSON robustness report."""
    from dataclasses import asdict

    def serialize_result(r: PerturbationResult) -> dict[str, Any]:
        return {
            "perturbation": r.perturbation,
            "config": asdict(r.config),
            "original_metrics": r.original_metrics,
            "perturbed_metrics": r.perturbed_metrics,
            "degradation": r.degradation,
            "severity_rating": r.severity_rating,
        }

    payload = {
        "generated_at": report.generated_at,
        "suite": report.suite,
        "seed": report.seed,
        "total_perturbations": report.total_perturbations,
        "results": [serialize_result(r) for r in report.results],
        "most_impactful": report.most_impactful,
        "least_impactful": report.least_impactful,
        "summary": report.summary,
    }
    return json.dumps(payload, indent=2, default=str)


def write_robustness_reports(
    report: MetadataRobustnessReport,
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Write robustness reports to files."""
    out = output_dir or REPORTS_DIR
    out.mkdir(parents=True, exist_ok=True)

    md_path = out / "metadata_robustness_report.md"
    json_path = out / "metadata_robustness_report.json"

    md_path.write_text(generate_robustness_markdown(report), encoding="utf-8")
    json_path.write_text(generate_robustness_json(report), encoding="utf-8")

    return {"markdown": str(md_path), "json": str(json_path)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.evaluation.metadata_robustness",
        description="Run metadata robustness experiments.",
    )
    parser.add_argument(
        "--suite",
        default="demo_v02",
        help="Benchmark suite to run.",
    )
    parser.add_argument(
        "--perturbations",
        type=str,
        default="all",
        help="Comma-separated perturbations or 'all'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
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

    # Parse perturbations
    if args.perturbations == "all":
        perturbations = list(PERTURBATION_TYPES)
    else:
        perturbations = [p.strip() for p in args.perturbations.split(",")]

    print(f"Running metadata robustness experiments on suite={args.suite}")
    print(f"Perturbations: {', '.join(perturbations)}")
    print()

    report = run_metadata_robustness(args.suite, perturbations, args.seed)

    if args.json_only:
        print(generate_robustness_json(report))
        return 0

    paths = write_robustness_reports(report, args.output_dir)

    # Print summary
    print("=" * 70)
    print("METADATA ROBUSTNESS RESULTS")
    print("=" * 70)
    print()
    print(f"Total Perturbations: {report.total_perturbations}")
    print(f"Most Impactful: {report.most_impactful}")
    print(f"Least Impactful: {report.least_impactful}")
    print()

    print("Degradation Summary:")
    for r in report.results:
        p5_deg = r.degradation.get("mean_precision_at_5", 0.0)
        print(f"  {r.perturbation}: {p5_deg:+.1%} ({r.severity_rating})")
    print()

    print("=" * 70)
    print("Reports written to:")
    print(f"  Markdown: {paths['markdown']}")
    print(f"  JSON: {paths['json']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
