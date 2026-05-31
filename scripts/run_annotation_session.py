#!/usr/bin/env python3
"""Run an annotation session for relevance labeling.

This script provides a CLI interface for collecting human relevance labels
on query-dataset pairs from the benchmark.

Usage:
    # Start a new annotation session
    python scripts/run_annotation_session.py --annotator your_name

    # Label linkage pairs
    python scripts/run_annotation_session.py --annotator your_name --linkage

    # Resume with specific query
    python scripts/run_annotation_session.py --annotator your_name --query "decision-making"
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_corpus(graph_path: Path) -> list[dict]:
    """Load corpus datasets."""
    with open(graph_path) as f:
        graph = json.load(f)

    nodes = graph.get("nodes", {})
    datasets = []

    for node_id, node in nodes.items():
        if node.get("node_type") != "dataset":
            continue

        props = node.get("properties", {})
        datasets.append({
            "id": node_id,
            "title": node.get("label", ""),
            "description": props.get("description", "")[:200] if props.get("description") else "",
            "source": props.get("source", ""),
        })

    return datasets


def load_benchmark_queries(path: Path) -> list[str]:
    """Load queries from benchmark."""
    import yaml

    with open(path) as f:
        data = yaml.safe_load(f)

    queries = []
    for q in data.get("queries", []):
        if isinstance(q, dict):
            queries.append(q.get("query", ""))
        else:
            queries.append(str(q))

    return [q for q in queries if q]


def simple_labeling_cli(
    annotator_id: str,
    queries: list[str],
    datasets: list[dict],
    output_path: Path,
    max_pairs: int = 25,
):
    """Simple CLI for relevance labeling."""
    labels = []
    pairs_labeled = 0

    print("\n" + "=" * 60)
    print("RELEVANCE LABELING SESSION")
    print("=" * 60)
    print(f"Annotator: {annotator_id}")
    print(f"Queries: {len(queries)}")
    print(f"Datasets: {len(datasets)}")
    print(f"Target: {max_pairs} pairs")
    print()
    print("Grade scale:")
    print("  0 = Not relevant")
    print("  1 = Marginally relevant")
    print("  2 = Relevant")
    print("  3 = Highly relevant")
    print()
    print("Commands: 0-3 to grade, s=skip, q=quit")
    print("=" * 60)
    input("\nPress Enter to start...")

    # Sample pairs
    for query in queries:
        if pairs_labeled >= max_pairs:
            break

        # Sample 5 random datasets per query
        sampled = random.sample(datasets, min(5, len(datasets)))

        for dataset in sampled:
            if pairs_labeled >= max_pairs:
                break

            print("\n" + "-" * 60)
            print(f"Query: {query}")
            print()
            print(f"Dataset: {dataset['title']}")
            print(f"  Source: {dataset['source']}")
            if dataset['description']:
                print(f"  Description: {dataset['description'][:150]}...")
            print()

            while True:
                response = input("Grade (0-3), s=skip, q=quit: ").strip().lower()

                if response == 'q':
                    print(f"\nSaving {len(labels)} labels...")
                    _save_labels(labels, output_path)
                    print(f"Labels saved to {output_path}")
                    return

                if response == 's':
                    break

                if response in ('0', '1', '2', '3'):
                    grade = int(response)
                    labels.append({
                        "annotator_id": annotator_id,
                        "query": query,
                        "dataset_id": dataset["id"],
                        "dataset_title": dataset["title"],
                        "grade": grade,
                    })
                    pairs_labeled += 1
                    print(f"  ✓ Labeled ({pairs_labeled}/{max_pairs})")
                    break

                print("  Invalid input. Enter 0-3, s, or q.")

    print(f"\nSession complete! Labeled {len(labels)} pairs.")
    _save_labels(labels, output_path)
    print(f"Labels saved to {output_path}")


def linkage_labeling_cli(
    annotator_id: str,
    benchmark_path: Path,
    output_path: Path,
    max_pairs: int = 25,
):
    """CLI for labeling dataset linkage pairs."""
    from neural_search.evaluation.dataset_linkage import load_benchmark

    benchmark = load_benchmark(benchmark_path)
    labels = []
    pairs_labeled = 0

    print("\n" + "=" * 60)
    print("DATASET LINKAGE LABELING")
    print("=" * 60)
    print(f"Annotator: {annotator_id}")
    print(f"Total pairs: {benchmark.n_pairs}")
    print(f"Target: {max_pairs} pairs")
    print()
    print("Relatedness scale:")
    print("  0 = Unrelated (no meaningful connection)")
    print("  1 = Topical (same general area)")
    print("  2 = Comparable (could compare in meta-analysis)")
    print("  3 = Reusable (analysis pipeline transfers)")
    print()
    print("Commands: 0-3 to grade, s=skip, q=quit")
    print("=" * 60)
    input("\nPress Enter to start...")

    # Shuffle pairs
    pairs = list(benchmark.pairs)
    random.shuffle(pairs)

    for pair in pairs:
        if pairs_labeled >= max_pairs:
            break

        print("\n" + "-" * 60)
        print(f"Linkage type (hint): {pair.linkage_type}")
        print()
        print(f"Dataset A: {pair.source_id}")
        print(f"Dataset B: {pair.target_id}")
        print()

        while True:
            response = input("Relatedness (0-3), s=skip, q=quit: ").strip().lower()

            if response == 'q':
                print(f"\nSaving {len(labels)} labels...")
                _save_linkage_labels(labels, output_path)
                print(f"Labels saved to {output_path}")
                return

            if response == 's':
                break

            if response in ('0', '1', '2', '3'):
                grade = int(response)
                labels.append({
                    "annotator_id": annotator_id,
                    "pair_id": pair.pair_id,
                    "source_id": pair.source_id,
                    "target_id": pair.target_id,
                    "linkage_type": str(pair.linkage_type),
                    "relatedness_score": grade,
                })
                pairs_labeled += 1
                print(f"  ✓ Labeled ({pairs_labeled}/{max_pairs})")
                break

            print("  Invalid input. Enter 0-3, s, or q.")

    print(f"\nSession complete! Labeled {len(labels)} pairs.")
    _save_linkage_labels(labels, output_path)
    print(f"Labels saved to {output_path}")


def _save_labels(labels: list[dict], path: Path):
    """Save labels to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")


def _save_linkage_labels(labels: list[dict], path: Path):
    """Save linkage labels to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for label in labels:
            f.write(json.dumps(label) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run annotation session")
    parser.add_argument("--annotator", "-a", required=True, help="Annotator ID")
    parser.add_argument("--linkage", action="store_true", help="Label linkage pairs")
    parser.add_argument("--max-pairs", "-n", type=int, default=25, help="Max pairs")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("data/eval/benchmark_queries_v2.yaml"),
    )
    parser.add_argument(
        "--linkage-benchmark",
        type=Path,
        default=Path("data/eval/linkage_benchmark_v1.json"),
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/graph/neural_search_graph.real_corpus.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/labels"),
    )

    args = parser.parse_args()

    if args.linkage:
        if not args.linkage_benchmark.exists():
            print(f"Error: Linkage benchmark not found: {args.linkage_benchmark}")
            sys.exit(1)

        linkage_labeling_cli(
            annotator_id=args.annotator,
            benchmark_path=args.linkage_benchmark,
            output_path=args.output / f"linkage_labels_{args.annotator}.jsonl",
            max_pairs=args.max_pairs,
        )
    else:
        if not args.graph.exists():
            print(f"Error: Graph not found: {args.graph}")
            sys.exit(1)

        datasets = load_corpus(args.graph)
        queries = load_benchmark_queries(args.benchmark) if args.benchmark.exists() else [
            "decision-making task with neural recordings",
            "calcium imaging in visual cortex",
            "Neuropixels recordings in mice",
            "hippocampal place cells",
            "motor cortex BMI decoding",
        ]

        simple_labeling_cli(
            annotator_id=args.annotator,
            queries=queries[:10],
            datasets=datasets,
            output_path=args.output / f"relevance_labels_{args.annotator}.jsonl",
            max_pairs=args.max_pairs,
        )


if __name__ == "__main__":
    main()
