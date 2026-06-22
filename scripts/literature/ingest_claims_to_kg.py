"""Ingest validated claims into the knowledge graph.

Usage: python scripts/literature/ingest_claims_to_kg.py
Input:  artifacts/claims/claims_validated.jsonl
Output: data/graph/claims_kg.jsonl (KG JSONL format)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.graph.schema import KnowledgeGraph, write_graph_jsonl
from neural_search.literature.claim_kg_builder import add_claims_to_graph

INPUT_PATH = REPO_ROOT / "artifacts/claims/claims_validated.jsonl"
OUTPUT_PATH = REPO_ROOT / "data/graph/claims_kg.jsonl"


def main() -> None:
    print(f"Ingesting claims from {INPUT_PATH}...")
    graph = KnowledgeGraph()
    stats = add_claims_to_graph(graph, INPUT_PATH)
    print(f"  {stats['claims_added']} claim nodes added")
    print(f"  {stats['edges_added']} edges added")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_graph_jsonl(graph, OUTPUT_PATH)
    print(f"Done. KG written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
