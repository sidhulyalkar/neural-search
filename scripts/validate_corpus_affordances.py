#!/usr/bin/env python3
"""Validate affordances against corpus metadata.

Runs affordance validation on all dataset nodes in the knowledge graph
using metadata-only mode (no file downloads required).

Usage:
    python scripts/validate_corpus_affordances.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_search.affordances.validators import (
    validate_bids_affordances,
    validate_nwb_affordances,
)


@dataclass
class CorpusValidationReport:
    """Summary of corpus-wide affordance validation."""

    timestamp: str
    corpus_size: int
    datasets_validated: int

    # Source breakdown
    datasets_by_source: dict[str, int] = field(default_factory=dict)

    # Affordance coverage
    affordance_support_counts: dict[str, int] = field(default_factory=dict)
    affordance_support_rates: dict[str, float] = field(default_factory=dict)

    # Feature presence
    feature_presence_counts: dict[str, int] = field(default_factory=dict)
    feature_presence_rates: dict[str, float] = field(default_factory=dict)

    # Validation details
    validation_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "corpus_size": self.corpus_size,
            "datasets_validated": self.datasets_validated,
            "datasets_by_source": self.datasets_by_source,
            "affordance_support_counts": self.affordance_support_counts,
            "affordance_support_rates": self.affordance_support_rates,
            "feature_presence_counts": self.feature_presence_counts,
            "feature_presence_rates": self.feature_presence_rates,
            "validation_results": self.validation_results,
        }


def extract_metadata_from_node(node: dict) -> dict:
    """Extract validation-relevant metadata from a dataset node."""
    properties = node.get("properties", {})
    usability = properties.get("usability_flags", {})
    label = node.get("label", "")
    node_id = node.get("node_id", "")

    metadata = {
        "dataset_id": node_id,
        "title": label,
        "source": properties.get("source", "unknown"),
    }

    # Map usability flags to validation metadata
    if usability.get("has_neural_data"):
        metadata["units"] = True
        metadata["n_units"] = 50  # Assume moderate for metadata-only

    if usability.get("has_trials"):
        metadata["trials"] = True
        metadata["n_trials"] = 100

    if usability.get("has_event_timestamps"):
        metadata["spike_times"] = True

    if usability.get("has_behavior"):
        metadata["description"] = f"{label} - behavioral data available"
    else:
        metadata["description"] = label

    # Infer modality from node edges or label
    label_lower = label.lower()
    if "fmri" in label_lower or "bold" in label_lower:
        metadata["modality"] = ["fMRI"]
    elif "eeg" in label_lower:
        metadata["modality"] = ["EEG"]
    elif "meg" in label_lower:
        metadata["modality"] = ["MEG"]
    elif "ephys" in label_lower or "electrophysiology" in label_lower:
        metadata["modality"] = ["ephys"]
    elif "calcium" in label_lower or "imaging" in label_lower:
        metadata["modality"] = ["calcium_imaging"]

    return metadata


def validate_corpus(graph_path: Path) -> CorpusValidationReport:
    """Run affordance validation on entire corpus."""
    print(f"Loading graph from {graph_path}...")
    with open(graph_path) as f:
        data = json.load(f)

    nodes = data.get("nodes", {})
    datasets = [n for n in nodes.values() if n.get("node_type") == "dataset"]

    print(f"Found {len(datasets)} dataset nodes")

    # Track results
    source_counts: Counter = Counter()
    affordance_counts: Counter = Counter()
    feature_counts: Counter = Counter()
    validation_results = []

    for node in datasets:
        properties = node.get("properties", {})
        source = properties.get("source", "unknown")
        source_counts[source] += 1

        # Extract metadata
        metadata = extract_metadata_from_node(node)
        dataset_id = metadata["dataset_id"]

        # Choose validator based on source
        if source in ("dandi", "allen", "nemo", "crcns"):
            result = validate_nwb_affordances(
                metadata=metadata,
                dataset_id=dataset_id,
            )
        elif source in ("openneuro",):
            result = validate_bids_affordances(
                metadata=metadata,
                dataset_id=dataset_id,
            )
        else:
            # Default to NWB for ephys-like sources
            result = validate_nwb_affordances(
                metadata=metadata,
                dataset_id=dataset_id,
            )

        # Aggregate affordance support
        for affordance, supported in result.affordance_support.items():
            if supported:
                affordance_counts[affordance] += 1

        # Aggregate feature presence
        for feature_name, check in result.feature_checks.items():
            if check.present:
                feature_counts[feature_name] += 1

        # Store result summary
        validation_results.append({
            "dataset_id": dataset_id,
            "source": source,
            "affordances_supported": [
                a for a, s in result.affordance_support.items() if s
            ],
            "features_present": [
                f for f, c in result.feature_checks.items() if c.present
            ],
        })

    # Compute rates
    n = len(datasets)
    affordance_rates = {a: c / n for a, c in affordance_counts.items()}
    feature_rates = {f: c / n for f, c in feature_counts.items()}

    return CorpusValidationReport(
        timestamp=datetime.now(UTC).isoformat(),
        corpus_size=n,
        datasets_validated=n,
        datasets_by_source=dict(source_counts),
        affordance_support_counts=dict(affordance_counts),
        affordance_support_rates=affordance_rates,
        feature_presence_counts=dict(feature_counts),
        feature_presence_rates=feature_rates,
        validation_results=validation_results,
    )


def generate_markdown_report(report: CorpusValidationReport) -> str:
    """Generate a markdown report from validation results."""
    lines = [
        "# Corpus Affordance Validation Report",
        "",
        f"**Generated:** {report.timestamp}",
        f"**Corpus Size:** {report.corpus_size} datasets",
        f"**Datasets Validated:** {report.datasets_validated}",
        "",
        "## Dataset Sources",
        "",
        "| Source | Count | Percentage |",
        "|--------|-------|------------|",
    ]

    for source, count in sorted(
        report.datasets_by_source.items(), key=lambda x: -x[1]
    ):
        pct = count / report.corpus_size * 100
        lines.append(f"| {source} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "## Affordance Support",
        "",
        "| Affordance | Datasets | Support Rate |",
        "|------------|----------|--------------|",
    ])

    for affordance, count in sorted(
        report.affordance_support_counts.items(), key=lambda x: -x[1]
    ):
        rate = report.affordance_support_rates.get(affordance, 0)
        lines.append(f"| {affordance} | {count} | {rate:.1%} |")

    lines.extend([
        "",
        "## Feature Presence",
        "",
        "| Feature | Datasets | Presence Rate |",
        "|---------|----------|---------------|",
    ])

    for feature, count in sorted(
        report.feature_presence_counts.items(), key=lambda x: -x[1]
    ):
        rate = report.feature_presence_rates.get(feature, 0)
        lines.append(f"| {feature} | {count} | {rate:.1%} |")

    return "\n".join(lines)


def main():
    project_root = Path(__file__).parent.parent
    graph_path = project_root / "data" / "graph" / "neural_search_graph.real_corpus.json"

    if not graph_path.exists():
        print(f"Error: Graph file not found at {graph_path}")
        sys.exit(1)

    report = validate_corpus(graph_path)

    # Save JSON report
    output_dir = project_root / "data" / "eval" / "affordance_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "validation_results.json"
    with open(json_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print(f"JSON report saved to {json_path}")

    # Save markdown report
    md_path = project_root / "reports" / "affordance_validation_v1.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write(generate_markdown_report(report))
    print(f"Markdown report saved to {md_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Corpus size: {report.corpus_size}")
    print(f"Sources: {report.datasets_by_source}")
    print(f"\nTop affordances by support rate:")
    for aff, rate in sorted(
        report.affordance_support_rates.items(), key=lambda x: -x[1]
    )[:5]:
        print(f"  {aff}: {rate:.1%}")


if __name__ == "__main__":
    main()
