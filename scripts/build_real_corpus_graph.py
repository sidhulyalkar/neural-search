#!/usr/bin/env python3
"""Build knowledge graph from expanded real corpus.

Combines all real corpus files (DANDI, OpenNeuro, Allen, NeMO) into a single graph.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from neural_search.graph.builder import build_graph_from_records, split_records
from neural_search.graph.quality import validate_graph_coverage
from neural_search.graph.schema import write_graph_json
from neural_search.normalized import load_normalized_records


CORPUS_DIR = project_root / "data" / "corpus" / "normalized"
GRAPH_OUTPUT = project_root / "data" / "graph" / "neural_search_graph.real_corpus.json"

# Real corpus files to include
REAL_CORPUS_FILES = [
    "real_dandi.jsonl",
    "real_openneuro.jsonl",
    "real_allen.jsonl",
    "real_nemo.jsonl",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build graph from real corpus")
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--output", type=str, default=str(GRAPH_OUTPUT))
    parser.add_argument("--validate", action="store_true", help="Run coverage validation")
    parser.add_argument("--profile", default="ci", help="Validation profile")
    args = parser.parse_args()

    # Load all corpus files
    all_datasets = []
    all_papers = []

    for filename in REAL_CORPUS_FILES:
        filepath = CORPUS_DIR / filename
        if not filepath.exists():
            print(f"Warning: Skipping {filename} (not found)")
            continue

        print(f"Loading {filename}...")
        records = list(load_normalized_records(str(filepath)))
        datasets, papers = split_records(records)
        print(f"  Loaded {len(datasets)} datasets, {len(papers)} papers")
        all_datasets.extend(datasets)
        all_papers.extend(papers)

    print(f"\nTotal: {len(all_datasets)} datasets, {len(all_papers)} papers")

    # Build graph
    print("\nBuilding knowledge graph...")
    graph = build_graph_from_records(
        all_datasets,
        all_papers,
        min_confidence=args.min_confidence,
    )

    print(f"Graph built: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_graph_json(graph, output_path)
    print(f"Wrote graph to {output_path}")

    # Validate if requested
    if args.validate:
        print(f"\nValidating coverage (profile: {args.profile})...")
        report = validate_graph_coverage(graph, profile=args.profile)
        print(f"Validation: {'PASSED' if report.passed else 'FAILED'}")
        print(f"  Errors: {report.error_count}, Warnings: {report.warning_count}")

        if not report.passed:
            print("\nErrors:")
            for issue in report.issues:
                if issue.severity == "error":
                    print(f"  [{issue.code}] {issue.message}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
