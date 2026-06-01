#!/usr/bin/env python3
"""Run baseline ladder evaluation on benchmark queries.

This script runs the 8-level baseline ladder evaluation to measure
the contribution of each retrieval component.

Usage:
    python scripts/run_baseline_ladder.py [--output REPORT_PATH]
    python scripts/run_baseline_ladder.py --queries PATH_TO_QUERIES.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load_benchmark_queries() -> list[str]:
    """Load benchmark queries from evaluation data."""
    queries_path = PROJECT_ROOT / "data" / "evaluation" / "benchmark_queries.json"

    if queries_path.exists():
        with open(queries_path) as f:
            data = json.load(f)
            return [q["query"] if isinstance(q, dict) else q for q in data.get("queries", data)]

    # Default benchmark queries if file doesn't exist
    return [
        "Find datasets with Neuropixels recordings in mouse visual cortex",
        "Datasets for latent-state modeling with trial structure",
        "Neural recordings during decision-making tasks in primates",
        "Single-cell RNA-seq data from hippocampus",
        "Calcium imaging during locomotion in mice",
        "ECoG recordings during speech production in humans",
        "Datasets suitable for spike sorting analysis",
        "Multi-modal neural and behavioral recordings",
        "Datasets with pose tracking and electrophysiology",
        "fMRI data during working memory tasks",
        "Patch-clamp recordings from cortical interneurons",
        "Datasets for training neural decoding models",
        "Recordings from multiple brain regions simultaneously",
        "Data for studying sensorimotor transformations",
        "Neural activity during reward learning in rodents",
    ]


def load_relevance_labels() -> dict[str, set[str]]:
    """Load human relevance labels mapping queries to relevant dataset IDs."""
    labels_path = PROJECT_ROOT / "data" / "evaluation" / "relevance_labels.json"

    if labels_path.exists():
        with open(labels_path) as f:
            data = json.load(f)
            return {q: set(ids) for q, ids in data.items()}

    # Return empty labels if no file exists
    print("Warning: No relevance labels found. Metrics will be based on empty ground truth.")
    return {}


def load_corpus() -> list[dict]:
    """Load the corpus for evaluation."""
    from neural_search.ingestion.demo_seed import build_combined_corpus

    try:
        corpus = build_combined_corpus()
        print(f"Loaded corpus with {len(corpus)} records")
        return corpus
    except Exception as e:
        print(f"Warning: Could not load full corpus: {e}")
        return []


def run_evaluation(
    queries: list[str],
    corpus: list[dict],
    relevance_labels: dict[str, set[str]],
    output_path: Path | None = None,
) -> None:
    """Run the baseline ladder evaluation."""
    from neural_search.evaluation.baseline_ladder import (
        LadderLevel,
        format_ladder_report_markdown,
        run_baseline_ladder,
    )

    print("\nRunning baseline ladder evaluation...")
    print(f"  Queries: {len(queries)}")
    print(f"  Corpus size: {len(corpus)}")
    print(f"  Labeled queries: {len(relevance_labels)}")
    print()

    # Run evaluation
    report = run_baseline_ladder(
        queries=queries,
        corpus=corpus,
        relevance_labels=relevance_labels,
        levels=list(LadderLevel),
    )

    # Format report
    markdown_report = format_ladder_report_markdown(report)

    # Print to console
    print(markdown_report)

    # Save to file if requested
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(markdown_report)
        print(f"\nReport saved to: {output_path}")

        # Also save JSON version
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        print(f"JSON report saved to: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline ladder evaluation"
    )
    parser.add_argument(
        "--queries",
        type=Path,
        help="Path to custom queries JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / f"baseline_ladder_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.md",
        help="Output path for the report",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        help="Path to custom relevance labels JSON",
    )

    args = parser.parse_args()

    # Load queries
    if args.queries and args.queries.exists():
        with open(args.queries) as f:
            data = json.load(f)
            queries = [q["query"] if isinstance(q, dict) else q for q in data.get("queries", data)]
    else:
        queries = load_benchmark_queries()

    # Load relevance labels
    if args.labels and args.labels.exists():
        with open(args.labels) as f:
            relevance_labels = {q: set(ids) for q, ids in json.load(f).items()}
    else:
        relevance_labels = load_relevance_labels()

    # Load corpus
    corpus = load_corpus()

    if not corpus:
        print("Error: No corpus loaded. Cannot run evaluation.")
        sys.exit(1)

    # Run evaluation
    run_evaluation(queries, corpus, relevance_labels, args.output)


if __name__ == "__main__":
    main()
