"""Ingest cross-finding and region co-occurrence relationships into the knowledge graph.

Usage: python scripts/literature/ingest_relationships_to_kg.py
Input:  artifacts/literature/relationships/finding_edges.jsonl
        artifacts/literature/relationships/region_cooccurrence.jsonl
Output: data/graph/relationships_kg.jsonl (KG JSONL format)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.graph.schema import KnowledgeGraph, write_graph_jsonl
from neural_search.literature.relationship_kg_builder import (
    add_finding_relationships_to_graph,
    add_region_cooccurrence_to_graph,
)

FINDING_EDGES_PATH = REPO_ROOT / "artifacts/literature/relationships/finding_edges.jsonl"
REGION_COOCCURRENCE_PATH = REPO_ROOT / "artifacts/literature/relationships/region_cooccurrence.jsonl"
OUTPUT_PATH = REPO_ROOT / "data/graph/relationships_kg.jsonl"


def main() -> None:
    print(f"Ingesting finding relationships from {FINDING_EDGES_PATH}...")
    graph = KnowledgeGraph()
    finding_stats = add_finding_relationships_to_graph(graph, FINDING_EDGES_PATH)
    print(f"  {finding_stats['finding_nodes_added']} finding nodes added")
    print(f"  {finding_stats['edges_added']} finding relationship edges added")

    print(f"Ingesting region co-occurrence from {REGION_COOCCURRENCE_PATH}...")
    region_stats = add_region_cooccurrence_to_graph(graph, REGION_COOCCURRENCE_PATH)
    print(f"  {region_stats['region_nodes_added']} region nodes added")
    print(f"  {region_stats['edges_added']} region co-occurrence edges added")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_graph_jsonl(graph, OUTPUT_PATH)
    print(f"Done. KG written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
