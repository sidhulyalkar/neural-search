#!/usr/bin/env python3
"""Generate dataset linkage pairs for annotation.

This script generates candidate pairs for the dataset linkage benchmark,
stratified by linkage type. Pairs are exported for human annotation.

Usage:
    python scripts/generate_linkage_pairs.py --output data/eval/linkage_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_search.evaluation.dataset_linkage import (
    DatasetPair,
    LinkageBenchmark,
    LinkageType,
    save_benchmark,
)


def load_corpus_graph(graph_path: Path) -> dict:
    """Load the knowledge graph."""
    with open(graph_path) as f:
        return json.load(f)


def extract_dataset_metadata(graph: dict) -> list[dict]:
    """Extract dataset nodes with their metadata."""
    nodes = graph.get("nodes", {})
    edges = graph.get("edges", {})

    datasets = []
    for node_id, node in nodes.items():
        if node.get("node_type") != "dataset":
            continue

        props = node.get("properties", {})
        usability = props.get("usability_flags", {})

        datasets.append({
            "dataset_id": node_id,
            "label": node.get("label", ""),
            "source": props.get("source", "unknown"),
            "has_neural_data": usability.get("has_neural_data", False),
            "has_behavior": usability.get("has_behavior", False),
            "has_trials": usability.get("has_trials", False),
        })

    # Extract relationships from edges
    dataset_tasks = defaultdict(set)
    dataset_modalities = defaultdict(set)
    dataset_species = defaultdict(set)
    dataset_regions = defaultdict(set)

    for _edge_id, edge in edges.items():
        source = edge.get("source_node_id", "")
        target = edge.get("target_node_id", "")
        edge_type = edge.get("edge_type", "")

        # Find dataset -> task edges
        if edge_type == "dataset_has_task":
            dataset_tasks[source].add(target)

        # Find dataset -> modality edges
        if edge_type == "dataset_has_modality":
            dataset_modalities[source].add(target)

        # Find dataset -> species edges
        if edge_type == "dataset_has_species":
            dataset_species[source].add(target)

        # Find dataset -> brain_region edges
        if edge_type == "dataset_records_region":
            dataset_regions[source].add(target)

    # Enrich dataset metadata (use node_id format to match edges)
    for d in datasets:
        node_id = d["dataset_id"]  # This is already "node:dataset:..." format
        d["tasks"] = list(dataset_tasks.get(node_id, set()))
        d["modalities"] = list(dataset_modalities.get(node_id, set()))
        d["species"] = list(dataset_species.get(node_id, set()))
        d["brain_regions"] = list(dataset_regions.get(node_id, set()))

    return datasets


def find_same_task_pairs(datasets: list[dict], n: int = 100) -> list[tuple[str, str]]:
    """Find pairs studying the same task."""
    task_to_datasets = defaultdict(list)
    for d in datasets:
        for task in d["tasks"]:
            task_to_datasets[task].append(d["dataset_id"])

    pairs = []
    for _task, dids in task_to_datasets.items():
        if len(dids) >= 2:
            for i in range(len(dids)):
                for j in range(i + 1, len(dids)):
                    pairs.append((dids[i], dids[j]))

    random.shuffle(pairs)
    return pairs[:n]


def find_same_modality_pairs(datasets: list[dict], n: int = 100) -> list[tuple[str, str]]:
    """Find pairs using the same modality."""
    mod_to_datasets = defaultdict(list)
    for d in datasets:
        for mod in d["modalities"]:
            mod_to_datasets[mod].append(d["dataset_id"])

    pairs = []
    for _mod, dids in mod_to_datasets.items():
        if len(dids) >= 2:
            for i in range(min(len(dids), 20)):
                for j in range(i + 1, min(len(dids), 20)):
                    pairs.append((dids[i], dids[j]))

    random.shuffle(pairs)
    return pairs[:n]


def find_same_species_pairs(datasets: list[dict], n: int = 100) -> list[tuple[str, str]]:
    """Find pairs from the same species."""
    species_to_datasets = defaultdict(list)
    for d in datasets:
        for sp in d["species"]:
            species_to_datasets[sp].append(d["dataset_id"])

    pairs = []
    for _sp, dids in species_to_datasets.items():
        if len(dids) >= 2:
            for i in range(min(len(dids), 20)):
                for j in range(i + 1, min(len(dids), 20)):
                    pairs.append((dids[i], dids[j]))

    random.shuffle(pairs)
    return pairs[:n]


def find_same_region_pairs(datasets: list[dict], n: int = 100) -> list[tuple[str, str]]:
    """Find pairs recording from the same brain region."""
    region_to_datasets = defaultdict(list)
    for d in datasets:
        for region in d["brain_regions"]:
            region_to_datasets[region].append(d["dataset_id"])

    pairs = []
    for _region, dids in region_to_datasets.items():
        if len(dids) >= 2:
            for i in range(min(len(dids), 15)):
                for j in range(i + 1, min(len(dids), 15)):
                    pairs.append((dids[i], dids[j]))

    random.shuffle(pairs)
    return pairs[:n]


def find_unrelated_pairs(datasets: list[dict], n: int = 100) -> list[tuple[str, str]]:
    """Find pairs that share nothing in common."""
    pairs = []
    random.shuffle(datasets)

    for i in range(len(datasets)):
        for j in range(i + 1, len(datasets)):
            d1, d2 = datasets[i], datasets[j]

            # Check for no overlap
            tasks_overlap = bool(set(d1["tasks"]) & set(d2["tasks"]))
            mod_overlap = bool(set(d1["modalities"]) & set(d2["modalities"]))
            species_overlap = bool(set(d1["species"]) & set(d2["species"]))

            if not tasks_overlap and not mod_overlap and not species_overlap:
                pairs.append((d1["dataset_id"], d2["dataset_id"]))
                if len(pairs) >= n:
                    return pairs

    return pairs


def generate_linkage_benchmark(
    graph_path: Path,
    pairs_per_type: int = 100,
) -> LinkageBenchmark:
    """Generate a complete linkage benchmark."""
    print(f"Loading graph from {graph_path}...")
    graph = load_corpus_graph(graph_path)

    print("Extracting dataset metadata...")
    datasets = extract_dataset_metadata(graph)
    print(f"Found {len(datasets)} datasets")

    all_pairs: list[DatasetPair] = []

    # Generate pairs by type
    print(f"\nGenerating {pairs_per_type} pairs per type...")

    same_task = find_same_task_pairs(datasets, pairs_per_type)
    print(f"  same_task: {len(same_task)} candidate pairs")
    for s, t in same_task:
        all_pairs.append(DatasetPair(
            source_id=s, target_id=t,
            linkage_type=LinkageType.SAME_TASK,
            relatedness_score=0,  # To be annotated
        ))

    same_mod = find_same_modality_pairs(datasets, pairs_per_type)
    print(f"  same_modality: {len(same_mod)} candidate pairs")
    for s, t in same_mod:
        all_pairs.append(DatasetPair(
            source_id=s, target_id=t,
            linkage_type=LinkageType.SAME_MODALITY,
            relatedness_score=0,
        ))

    same_species = find_same_species_pairs(datasets, pairs_per_type)
    print(f"  same_species: {len(same_species)} candidate pairs")
    for s, t in same_species:
        all_pairs.append(DatasetPair(
            source_id=s, target_id=t,
            linkage_type=LinkageType.SAME_SPECIES,
            relatedness_score=0,
        ))

    same_region = find_same_region_pairs(datasets, pairs_per_type)
    print(f"  same_brain_region: {len(same_region)} candidate pairs")
    for s, t in same_region:
        all_pairs.append(DatasetPair(
            source_id=s, target_id=t,
            linkage_type=LinkageType.SAME_BRAIN_REGION,
            relatedness_score=0,
        ))

    unrelated = find_unrelated_pairs(datasets, pairs_per_type)
    print(f"  unrelated: {len(unrelated)} candidate pairs")
    for s, t in unrelated:
        all_pairs.append(DatasetPair(
            source_id=s, target_id=t,
            linkage_type=LinkageType.UNRELATED,
            relatedness_score=0,
        ))

    return LinkageBenchmark(
        benchmark_id="linkage_v1",
        version="1.0",
        description="Dataset linkage benchmark for Neural Search evaluation",
        pairs=all_pairs,
        target_by_type={
            "same_task": pairs_per_type,
            "same_modality": pairs_per_type,
            "same_species": pairs_per_type,
            "same_brain_region": pairs_per_type,
            "unrelated": pairs_per_type,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Generate dataset linkage pairs")
    parser.add_argument(
        "--graph", "-g",
        type=Path,
        default=Path("data/graph/neural_search_graph.real_corpus.json"),
        help="Path to knowledge graph",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/eval/linkage_benchmark_v1.json"),
        help="Output path for benchmark",
    )
    parser.add_argument(
        "--pairs-per-type", "-n",
        type=int,
        default=100,
        help="Number of pairs per linkage type",
    )

    args = parser.parse_args()

    if not args.graph.exists():
        print(f"Error: Graph file not found: {args.graph}")
        sys.exit(1)

    benchmark = generate_linkage_benchmark(args.graph, args.pairs_per_type)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_benchmark(benchmark, args.output)

    print(f"\n{'='*60}")
    print("BENCHMARK GENERATED")
    print(f"{'='*60}")
    print(f"Total pairs: {benchmark.n_pairs}")
    print(f"Output: {args.output}")
    print("\nNext steps:")
    print("1. Export pairs for annotation: neural_search.labeling.cli")
    print("2. Collect 2+ annotator labels per pair")
    print("3. Compute agreement and adjudicate disagreements")


if __name__ == "__main__":
    main()
