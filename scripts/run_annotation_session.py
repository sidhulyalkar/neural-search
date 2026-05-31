#!/usr/bin/env python3
"""Run an annotation session for relevance labeling.

This script provides a CLI interface for collecting human relevance labels
on query-dataset pairs from the benchmark.

Usage:
    # Start a new annotation session
    python scripts/run_annotation_session.py --annotator your_name

    # Resume an existing session
    python scripts/run_annotation_session.py --annotator your_name --resume

    # Use active learning to prioritize uncertain pairs
    python scripts/run_annotation_session.py --annotator your_name --active-learning
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_search.labeling import (
    LabelingSession,
    RelevanceGrade,
    run_labeling_cli,
)


def load_benchmark_queries(benchmark_path: Path) -> list[dict]:
    """Load queries from benchmark file."""
    import yaml

    with open(benchmark_path) as f:
        data = yaml.safe_load(f)

    return data.get("queries", [])


def load_corpus_for_labeling(graph_path: Path) -> list[dict]:
    """Load corpus datasets for labeling."""
    with open(graph_path) as f:
        graph = json.load(f)

    nodes = graph.get("nodes", {})
    datasets = []

    for node_id, node in nodes.items():
        if node.get("node_type") != "dataset":
            continue

        props = node.get("properties", {})
        datasets.append({
            "dataset_id": node_id,
            "title": node.get("label", ""),
            "description": "",
            "source": props.get("source", ""),
            "url": props.get("url", ""),
        })

    return datasets


def main():
    parser = argparse.ArgumentParser(description="Run annotation session")
    parser.add_argument(
        "--annotator", "-a",
        required=True,
        help="Annotator ID/name",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/eval/benchmark_queries_v2.yaml"),
        help="Path to benchmark queries",
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/graph/neural_search_graph.real_corpus.json"),
        help="Path to knowledge graph",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/labels"),
        help="Output directory for labels",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing session",
    )
    parser.add_argument(
        "--active-learning",
        action="store_true",
        help="Use active learning to prioritize pairs",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=50,
        help="Maximum pairs to label per session",
    )

    args = parser.parse_args()

    if not args.benchmark.exists():
        print(f"Error: Benchmark not found: {args.benchmark}")
        sys.exit(1)

    if not args.graph.exists():
        print(f"Error: Graph not found: {args.graph}")
        sys.exit(1)

    print(f"Loading benchmark from {args.benchmark}...")
    queries = load_benchmark_queries(args.benchmark)
    print(f"Loaded {len(queries)} queries")

    print(f"Loading corpus from {args.graph}...")
    datasets = load_corpus_for_labeling(args.graph)
    print(f"Loaded {len(datasets)} datasets")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Session file
    session_file = args.output / f"session_{args.annotator}.json"

    print(f"\n{'='*60}")
    print(f"ANNOTATION SESSION: {args.annotator}")
    print(f"{'='*60}")
    print(f"Queries: {len(queries)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Max pairs: {args.max_pairs}")
    print(f"Active learning: {args.active_learning}")
    print()

    # Run CLI
    print("Starting labeling CLI...")
    print("Commands:")
    print("  0-3: Set relevance grade")
    print("  s: Skip this pair")
    print("  q: Quit and save")
    print()

    run_labeling_cli(
        queries=[q["query"] for q in queries[:10]],  # Start with first 10 queries
        datasets=datasets,
        output_path=args.output / f"labels_{args.annotator}.jsonl",
        annotator_id=args.annotator,
        max_pairs=args.max_pairs,
    )


if __name__ == "__main__":
    main()
