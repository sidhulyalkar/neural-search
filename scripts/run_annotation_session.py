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


def _load_dataset_details(graph_path: Path) -> dict[str, dict]:
    """Load rich dataset details for annotation display."""
    with open(graph_path) as f:
        graph = json.load(f)

    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})

    # Build dataset info
    dataset_info: dict[str, dict] = {}

    for node_id, node in nodes.items():
        if node.get("node_type") != "dataset":
            continue

        props = node.get("properties", {})
        dataset_info[node_id] = {
            "title": node.get("label", "Unknown"),
            "source": props.get("source", "unknown"),
            "species": [],
            "modalities": [],
            "tasks": [],
            "regions": [],
        }

    # Enrich from edges
    for edge_id, edge in edges.items():
        source = edge.get("source_node_id", "")
        target = edge.get("target_node_id", "")
        edge_type = edge.get("edge_type", "")

        if source not in dataset_info:
            continue

        # Extract the label from target node
        target_node = nodes.get(target, {})
        target_label = target_node.get("label", target.split(":")[-1])

        if edge_type == "dataset_has_species":
            dataset_info[source]["species"].append(target_label)
        elif edge_type == "dataset_has_modality":
            dataset_info[source]["modalities"].append(target_label)
        elif edge_type == "dataset_has_task":
            dataset_info[source]["tasks"].append(target_label)
        elif edge_type == "dataset_records_region":
            dataset_info[source]["regions"].append(target_label)

    return dataset_info


def _compute_shared_attributes(info_a: dict, info_b: dict) -> dict[str, list[str]]:
    """Find shared attributes between two datasets."""
    shared = {}

    for attr in ["species", "modalities", "tasks", "regions"]:
        set_a = set(info_a.get(attr, []))
        set_b = set(info_b.get(attr, []))
        common = set_a & set_b
        if common:
            shared[attr] = list(common)[:5]

    return shared


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
    graph_path: Path,
    max_pairs: int = 25,
):
    """CLI for labeling dataset linkage pairs."""
    from neural_search.evaluation.dataset_linkage import load_benchmark

    benchmark = load_benchmark(benchmark_path)

    # Load full dataset metadata for display
    dataset_info = _load_dataset_details(graph_path)

    labels = []
    pairs_labeled = 0

    print("\n" + "=" * 70)
    print("DATASET LINKAGE LABELING")
    print("=" * 70)
    print(f"Annotator: {annotator_id}")
    print(f"Total pairs: {benchmark.n_pairs}")
    print(f"Target: {max_pairs} pairs")
    print()
    print("Relatedness scale:")
    print("  0 = Unrelated (no meaningful connection)")
    print("  1 = Topical (same general area, e.g. both neuroscience)")
    print("  2 = Comparable (could compare in meta-analysis)")
    print("  3 = Reusable (same analysis pipeline would work)")
    print()
    print("Commands: 0-3 to grade, s=skip, q=quit")
    print("=" * 70)
    input("\nPress Enter to start...")

    # Shuffle pairs
    pairs = list(benchmark.pairs)
    random.shuffle(pairs)

    for pair in pairs:
        if pairs_labeled >= max_pairs:
            break

        # Get rich metadata for both datasets
        info_a = dataset_info.get(pair.source_id, {})
        info_b = dataset_info.get(pair.target_id, {})

        print("\n" + "=" * 70)
        print(f"PAIR {pairs_labeled + 1}/{max_pairs}  |  Hint: {pair.linkage_type}")
        print("=" * 70)

        # Dataset A
        print("\n[DATASET A]")
        print(f"  Title: {info_a.get('title', 'Unknown')}")
        print(f"  Source: {info_a.get('source', '?')}")
        if info_a.get('species'):
            print(f"  Species: {', '.join(info_a['species'][:3])}")
        if info_a.get('modalities'):
            print(f"  Modality: {', '.join(info_a['modalities'][:3])}")
        if info_a.get('tasks'):
            print(f"  Tasks: {', '.join(info_a['tasks'][:3])}")
        if info_a.get('regions'):
            print(f"  Brain regions: {', '.join(info_a['regions'][:5])}")

        # Dataset B
        print("\n[DATASET B]")
        print(f"  Title: {info_b.get('title', 'Unknown')}")
        print(f"  Source: {info_b.get('source', '?')}")
        if info_b.get('species'):
            print(f"  Species: {', '.join(info_b['species'][:3])}")
        if info_b.get('modalities'):
            print(f"  Modality: {', '.join(info_b['modalities'][:3])}")
        if info_b.get('tasks'):
            print(f"  Tasks: {', '.join(info_b['tasks'][:3])}")
        if info_b.get('regions'):
            print(f"  Brain regions: {', '.join(info_b['regions'][:5])}")

        # Show shared attributes
        shared = _compute_shared_attributes(info_a, info_b)
        if shared:
            print("\n[SHARED]")
            for attr, values in shared.items():
                print(f"  {attr}: {', '.join(values)}")

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
            graph_path=args.graph,
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
