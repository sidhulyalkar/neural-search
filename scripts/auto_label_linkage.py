#!/usr/bin/env python3
"""Automatically label dataset linkage benchmark using graph-based scoring.

This replaces human annotation with empirical relatedness scoring based on:
- Modality alignment (can same analysis pipeline process both?)
- Task compatibility (do they study compatible questions?)
- Species alignment (are findings transferable?)
- Brain region overlap (do they cover comparable circuits?)
- Affordance compatibility (do they support similar analyses?)
- Graph proximity (how connected are they in the knowledge graph?)
- Provenance quality (are both well-documented?)
- Statistical power (do both have sufficient data?)
- Pipeline transferability (would code transfer?)

Usage:
    python scripts/auto_label_linkage.py

    # With custom paths
    python scripts/auto_label_linkage.py \
        --graph data/graph/neural_search_graph.real_corpus.json \
        --benchmark data/eval/linkage_benchmark_v1.json \
        --output data/eval/linkage_benchmark_v1_labeled.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from neural_search.evaluation.relatedness_scorer import (
    RelatednessScorer,
    auto_label_linkage_benchmark,
)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label linkage benchmark using graph-based relatedness"
    )
    parser.add_argument(
        "--graph", "-g",
        type=Path,
        default=Path("data/graph/neural_search_graph.real_corpus.json"),
        help="Path to knowledge graph",
    )
    parser.add_argument(
        "--benchmark", "-b",
        type=Path,
        default=Path("data/eval/linkage_benchmark_v1.json"),
        help="Path to input benchmark",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/eval/linkage_benchmark_v1_labeled.json"),
        help="Path to output labeled benchmark",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Show detailed analysis of scoring",
    )

    args = parser.parse_args()

    if not args.graph.exists():
        print(f"Error: Graph not found: {args.graph}")
        sys.exit(1)

    if not args.benchmark.exists():
        print(f"Error: Benchmark not found: {args.benchmark}")
        sys.exit(1)

    # Run auto-labeling
    auto_label_linkage_benchmark(args.graph, args.benchmark, args.output)

    if args.analyze:
        print("\n" + "=" * 60)
        print("DETAILED ANALYSIS")
        print("=" * 60)

        # Load and analyze
        with open(args.output) as f:
            labeled = json.load(f)

        pairs = labeled.get("pairs", [])

        # Show score distribution by original linkage type
        by_type: dict[str, list[int]] = {}
        for pair in pairs:
            lt = pair.get("linkage_type", "unknown")
            grade = pair.get("relatedness_score", 0)
            if lt not in by_type:
                by_type[lt] = []
            by_type[lt].append(grade)

        print("\nGrade distribution by linkage type:")
        for lt, grades in sorted(by_type.items()):
            avg = sum(grades) / len(grades) if grades else 0
            print(f"\n  {lt}:")
            print(f"    Count: {len(grades)}")
            print(f"    Avg grade: {avg:.2f}")
            print(f"    Distribution: {[grades.count(g) for g in range(4)]}")

        # Show some high-scoring pairs
        high_pairs = [p for p in pairs if p.get("relatedness_score", 0) >= 2]
        print(f"\nHigh-relatedness pairs (grade >= 2): {len(high_pairs)}")

        if high_pairs[:3]:
            print("\nTop 3 examples:")
            for p in high_pairs[:3]:
                attrs = p.get("shared_attributes", {})
                mod = attrs.get("modality_alignment", {})
                print(f"  {p['source_id'][:30]} <-> {p['target_id'][:30]}")
                print(f"    Grade: {p['relatedness_score']}")
                if mod.get("shared_items"):
                    print(f"    Shared modalities: {mod['shared_items']}")


if __name__ == "__main__":
    main()
