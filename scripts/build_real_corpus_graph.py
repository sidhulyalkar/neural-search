#!/usr/bin/env python3
"""Build knowledge graph directly from full_corpus_v09.jsonl with real corpus IDs.

Reads the combined corpus JSONL, builds a KnowledgeGraph where each record
becomes a dataset node with id = node:dataset:{source}:{source_id}, matching
what _resolve_dataset_node_id expects in search_features.py.

Outputs: data/graph/neural_search_graph.real_corpus.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

script_dir = Path(__file__).resolve().parents[1]

# In a git worktree, large data files live in the main repo, not the worktree.
# Walk up to find the directory that actually has data/corpus/normalized/.
def _find_project_root() -> Path:
    candidates = [script_dir]
    # Also check if the worktree .git file points to a parent
    git_file = script_dir / ".git"
    if git_file.is_file():
        # worktree: .git is a file like "gitdir: ../../.git/worktrees/..."
        content = git_file.read_text()
        if content.startswith("gitdir:"):
            rel = content.split(":", 1)[1].strip()
            # .git/worktrees/xxx → main repo root is 3 levels up
            worktree_git = (script_dir / rel).resolve()
            candidates.append(worktree_git.parents[2])
    for candidate in candidates:
        corpus_check = candidate / "data" / "corpus" / "normalized" / "combined_corpus.jsonl"
        if corpus_check.exists():
            return candidate
    return script_dir

project_root = _find_project_root()
sys.path.insert(0, str(project_root))
# Also insert the worktree root so neural_search package is found
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from neural_search.graph.schema import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
    write_graph_json,
)

CORPUS_PATH = (
    project_root
    / "data"
    / "corpus"
    / "normalized"
    / "combined_corpus.jsonl"
    / "full_corpus_v09.jsonl"
)
OUTPUT_PATH = project_root / "data" / "graph" / "neural_search_graph.real_corpus.json"

BUILDER_NAME = "build_real_corpus_graph"
BUILDER_VERSION = "v1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_str_list(items: Any) -> list[str]:
    """Normalize a field value that may be a list of strings or dicts."""
    if not items:
        return []
    result: list[str] = []
    for item in items:
        if isinstance(item, str):
            v = item.strip()
        elif isinstance(item, dict):
            v = str(
                item.get("label") or item.get("id") or item.get("name") or ""
            ).strip()
        else:
            v = str(item).strip()
        if v:
            result.append(v)
    return result


def _safe_id_part(value: str) -> str:
    """Make a safe node ID part (no colons or whitespace issues)."""
    return value.strip().replace(" ", "_")


def _concept_node_id(node_type: str, value: str) -> str:
    """Stable node ID for a concept node like modality/task/species/region."""
    # Normalize: lowercase, spaces to underscores
    normalized = value.lower().replace(" ", "_").replace("-", "_")
    return f"node:{node_type}:{normalized}"


def _make_evidence(
    *,
    source_id: str,
    source_field: str,
    evidence_text: str,
    confidence: float = 0.9,
) -> GraphEvidence:
    return GraphEvidence(
        evidence_id=f"evidence:corpus:{source_id}:{source_field}:{evidence_text[:80]}",
        source_type="corpus",
        source_id=source_id,
        source_field=source_field,
        evidence_text=evidence_text,
        confidence=confidence,
        extractor_name=BUILDER_NAME,
        extractor_version=BUILDER_VERSION,
    )


def _make_edge(
    source_node_id: str,
    edge_type: str,
    target_node_id: str,
    confidence: float = 1.0,
) -> KnowledgeGraphEdge:
    edge_id = make_edge_id(source_node_id, edge_type, target_node_id)
    return KnowledgeGraphEdge(
        edge_id=edge_id,
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        directed=True,
        confidence=confidence,
        evidence=[],
        properties={},
    )


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_graph(corpus: list[dict[str, Any]]) -> KnowledgeGraph:
    nodes: dict[str, KnowledgeGraphNode] = {}
    edges: dict[str, KnowledgeGraphEdge] = {}

    # Track concept nodes that already exist (avoid re-creating)
    concept_nodes_seen: set[str] = set()

    # For cross-dataset edges: map concept-pair → list of dataset node IDs
    # key: frozenset({modality_node_id, region_node_id})
    region_modality_datasets: dict[frozenset[str], list[str]] = defaultdict(list)
    # key: frozenset({species_node_id, task_node_id})
    species_task_datasets: dict[frozenset[str], list[str]] = defaultdict(list)

    FIELD_MAP = [
        ("modalities", "modality", "dataset_has_modality"),
        ("tasks", "task", "dataset_has_task"),
        ("species", "species", "dataset_has_species"),
        ("brain_regions", "brain_region", "dataset_records_region"),
    ]

    for record in corpus:
        source = str(record.get("source", "unknown"))
        source_id = str(record.get("source_id") or "")
        if not source_id:
            # Fall back to dataset_id parsing
            did = str(record.get("dataset_id", ""))
            parts = did.split(":")
            if len(parts) >= 3:
                source_id = parts[-1]
            else:
                source_id = did

        dataset_node_id = f"node:dataset:{source}:{source_id}"
        title = str(record.get("title") or f"{source}:{source_id}")
        dataset_id = str(record.get("dataset_id") or f"dataset:{source}:{source_id}")

        # Dataset node
        dataset_node = KnowledgeGraphNode(
            node_id=dataset_node_id,
            node_type="dataset",
            label=title,
            # aliases let _resolve_dataset_node_id find this node by
            # "dandi:000785" or "000785" or "dataset:dandi:000785"
            aliases=[f"{source}:{source_id}", source_id, dataset_id],
            source_ids=[dataset_id],
            properties={
                "source": source,
                "source_id": source_id,
                "url": record.get("url", ""),
            },
            evidence=[
                _make_evidence(
                    source_id=dataset_id,
                    source_field="title",
                    evidence_text=title,
                    confidence=1.0,
                )
            ],
            confidence=1.0,
        )
        nodes[dataset_node_id] = dataset_node

        # Concept nodes + edges
        dataset_modality_ids: list[str] = []
        dataset_region_ids: list[str] = []
        dataset_task_ids: list[str] = []
        dataset_species_ids: list[str] = []

        for field_name, node_type, edge_type in FIELD_MAP:
            values = extract_str_list(record.get(field_name) or [])
            for val in values:
                concept_id = _concept_node_id(node_type, val)

                if concept_id not in concept_nodes_seen:
                    concept_node = KnowledgeGraphNode(
                        node_id=concept_id,
                        node_type=node_type,
                        label=val,
                        aliases=[val.lower(), val],
                        source_ids=[],
                        properties={},
                        evidence=[],
                        confidence=0.9,
                    )
                    nodes[concept_id] = concept_node
                    concept_nodes_seen.add(concept_id)

                edge = _make_edge(dataset_node_id, edge_type, concept_id)
                edges[edge.edge_id] = edge

                if node_type == "modality":
                    dataset_modality_ids.append(concept_id)
                elif node_type == "brain_region":
                    dataset_region_ids.append(concept_id)
                elif node_type == "task":
                    dataset_task_ids.append(concept_id)
                elif node_type == "species":
                    dataset_species_ids.append(concept_id)

        # Accumulate cross-dataset pairs
        for mod_id in dataset_modality_ids:
            for reg_id in dataset_region_ids:
                pair = frozenset([mod_id, reg_id])
                region_modality_datasets[pair].append(dataset_node_id)

        for sp_id in dataset_species_ids:
            for task_id in dataset_task_ids:
                pair = frozenset([sp_id, task_id])
                species_task_datasets[pair].append(dataset_node_id)

    # Cross-dataset edges: same modality+region
    cross_edge_count = 0
    for pair, ds_ids in region_modality_datasets.items():
        if len(ds_ids) < 2:
            continue
        # Cap at 50 partners per pair to avoid edge blowup on popular concepts
        partners = ds_ids[:50]
        for i in range(len(partners)):
            for j in range(i + 1, len(partners)):
                a, b = partners[i], partners[j]
                edge_id = make_edge_id(a, "dataset_similar_to_dataset", b)
                if edge_id not in edges:
                    edge = KnowledgeGraphEdge(
                        edge_id=edge_id,
                        source_node_id=a,
                        target_node_id=b,
                        edge_type="dataset_similar_to_dataset",
                        directed=False,
                        confidence=0.5,
                        evidence=[],
                        properties={"cross_type": "same_region_cross_modality"},
                    )
                    edges[edge_id] = edge
                    cross_edge_count += 1

    # Cross-dataset edges: same species+task
    for pair, ds_ids in species_task_datasets.items():
        if len(ds_ids) < 2:
            continue
        partners = ds_ids[:50]
        for i in range(len(partners)):
            for j in range(i + 1, len(partners)):
                a, b = partners[i], partners[j]
                edge_id = make_edge_id(a, "dataset_similar_to_dataset", b)
                if edge_id not in edges:
                    edge = KnowledgeGraphEdge(
                        edge_id=edge_id,
                        source_node_id=a,
                        target_node_id=b,
                        edge_type="dataset_similar_to_dataset",
                        directed=False,
                        confidence=0.5,
                        evidence=[],
                        properties={"cross_type": "same_task_cross_species"},
                    )
                    edges[edge_id] = edge
                    cross_edge_count += 1

    print(f"  Cross-dataset edges added: {cross_edge_count}")

    return KnowledgeGraph(nodes=nodes, edges=edges, metadata={
        "builder": BUILDER_NAME,
        "builder_version": BUILDER_VERSION,
        "corpus": str(CORPUS_PATH),
        "record_count": len(corpus),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    if not CORPUS_PATH.exists():
        print(f"ERROR: Corpus not found: {CORPUS_PATH}", file=sys.stderr)
        return 1

    print(f"Reading corpus from {CORPUS_PATH} ...")
    t0 = time.time()
    corpus = [
        json.loads(line)
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"  {len(corpus)} records in {time.time() - t0:.1f}s")

    print("Building graph ...")
    t1 = time.time()
    graph = build_graph(corpus)
    print(f"  Done in {time.time() - t1:.1f}s")

    # Summary by node type
    node_counts: dict[str, int] = defaultdict(int)
    for node in graph.nodes.values():
        node_counts[node.node_type] += 1

    edge_counts: dict[str, int] = defaultdict(int)
    for edge in graph.edges.values():
        key = edge.properties.get("cross_type") or edge.edge_type
        edge_counts[key] += 1

    print("\nNode counts by type:")
    for ntype, count in sorted(node_counts.items(), key=lambda x: -x[1]):
        print(f"  {ntype:<25} {count:>6}")

    print("\nEdge counts by type:")
    for etype, count in sorted(edge_counts.items(), key=lambda x: -x[1]):
        print(f"  {etype:<40} {count:>6}")

    print(f"\nTotal nodes: {len(graph.nodes)}, Total edges: {len(graph.edges)}")

    print(f"\nWriting graph to {OUTPUT_PATH} ...")
    t2 = time.time()
    write_graph_json(graph, OUTPUT_PATH)
    print(f"  Written in {time.time() - t2:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
