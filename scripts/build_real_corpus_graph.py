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

from neural_search.graph.builder import build_taxonomy_requirement_subgraph
from neural_search.graph.evidence_tier_upgrader import apply_file_validation_upgrades
from neural_search.graph.method_registry_builder import build_method_registry_subgraph
from neural_search.graph.reanalysis_bridge_builder import build_reanalysis_bridge_edges
from neural_search.graph.reanalysis_candidates import build_reanalysis_candidate_edges
from neural_search.graph.reinterpretation_candidate_builder import (
    build_reinterpretation_candidate_edges,
)
from neural_search.graph.schema import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
    resolve_dangling_edges,
    write_graph_json,
)
from neural_search.graph.paper_node_builder import attach_retraction_status, build_paper_nodes_and_links
from neural_search.graph.reprocessing_candidate_builder import attach_reprocessing_candidate_status
from neural_search.ingestion.allen_connectivity_builder import build_allen_connectivity_kg
from neural_search.ingestion.citation_builder import build_citation_kg_for_graph
from neural_search.ingestion.concept_builder import build_concept_kg
from neural_search.ingestion.disorder_builder import build_disorder_kg
from neural_search.ingestion.hcp_connectivity import build_hcp_kg
from neural_search.ingestion.methods_builder import build_methods_kg
from neural_search.ingestion.oscillation_builder import build_oscillation_kg
from neural_search.ingestion.paradigm_builder import build_paradigm_kg
from neural_search.ingestion.scholarpedia_builder import build_scholarpedia_kg
from neural_search.ingestion.species_homology_builder import build_homology_kg

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

    # ── Reanalysis bridge edges ─────────────────────────────────────────────
    # Evidence-backed variant of the reanalysis-candidate signal: "a similar
    # dataset was actually analyzed with method X (per a real paper); this
    # dataset hasn't been." Must run after cross-dataset dataset_similar_to_dataset
    # edges exist (above) since it traverses them. See
    # neural_search.graph.reanalysis_bridge_builder for the join logic.
    try:
        bridge_edges = build_reanalysis_bridge_edges(KnowledgeGraph(nodes=nodes, edges=edges))
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, this is a build script
        print(f"  WARNING: reanalysis bridge layer unavailable: {exc}")
        bridge_edges = []
    for edge in bridge_edges:
        edges.setdefault(edge.edge_id, edge)
    print(f"  Reanalysis bridge edges added: {len(bridge_edges)}")

    # ── Reinterpretation candidates ──────────────────────────────────────────
    # Real literature-contradiction-backed signal: "this dataset's linked
    # paper's finding is directly contradicted by another paper linked to a
    # different dataset — reinterpreting/reanalyzing through that
    # contradicting framing is a genuine, literature-motivated opportunity."
    # See neural_search.graph.reinterpretation_candidate_builder.
    try:
        reinterpretation_edges = build_reinterpretation_candidate_edges(
            KnowledgeGraph(nodes=nodes, edges=edges)
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, this is a build script
        print(f"  WARNING: reinterpretation candidate layer unavailable: {exc}")
        reinterpretation_edges = []
    for edge in reinterpretation_edges:
        edges.setdefault(edge.edge_id, edge)
    print(f"  Reinterpretation candidate edges added: {len(reinterpretation_edges)}")

    # ── Methodology registry layer ─────────────────────────────────────────
    # Brings named-technique nodes (formulas/assumptions/key papers), the
    # analysis_affordance nodes + analysis_requires_* requirement edges, and
    # the method<->analysis_family bridge into the actual production graph
    # for the first time — these previously only lived in the separate
    # build_graph_from_records() dev/CLI pipeline.
    taxonomy_kg = build_taxonomy_requirement_subgraph()
    for node in taxonomy_kg.nodes.values():
        nodes.setdefault(node.node_id, node)
    for edge in taxonomy_kg.edges.values():
        edges.setdefault(edge.edge_id, edge)
    print(f"  Taxonomy requirement KG merged: {len(taxonomy_kg.nodes)} nodes, {len(taxonomy_kg.edges)} edges")

    methods_kg = build_methods_kg()
    for node in methods_kg.nodes.values():
        nodes.setdefault(node.node_id, node)
    for edge in methods_kg.edges.values():
        edges.setdefault(edge.edge_id, edge)
    print(f"  Methods KG merged: {len(methods_kg.nodes)} nodes, {len(methods_kg.edges)} edges")

    registry_kg = build_method_registry_subgraph()
    for edge in registry_kg.edges.values():
        edges.setdefault(edge.edge_id, edge)
    print(f"  Method registry merged: {len(registry_kg.edges)} edges")

    # ── Reanalysis candidates ──────────────────────────────────────────────
    candidate_edges = build_reanalysis_candidate_edges(corpus)
    for edge in candidate_edges:
        edges.setdefault(edge.edge_id, edge)
    print(f"  Reanalysis candidate edges added: {len(candidate_edges)}")

    # ── Evidence-tier upgrades from live file validation ────────────────────
    # Reads artifacts/validation/top_suggestions_file_validation.jsonl
    # (produced by scripts/validate_top_reanalysis_suggestions.py, a separate
    # live-network script, not run on every build) and promotes matching
    # candidate/bridge edges from heuristic_candidate/evidence_backed_bridge
    # to file_validated. A no-op if that artifact doesn't exist yet.
    edges, evidence_tier_upgrades = apply_file_validation_upgrades(edges)
    print(f"  Evidence tier upgrades (file_validated): {evidence_tier_upgrades}")

    # ── Previously-orphaned KG layers ───────────────────────────────────────
    # These builders are fully implemented and tested but, until now, only
    # ever wrote to artifacts/kg/composed_kg.jsonl (via compose_kg.py), which
    # nothing merges into the graph search actually reads — see
    # reports/architecture_connectivity_audit_2026-07-01.md. Each is wrapped
    # in try/except so a missing/changed data file degrades gracefully
    # instead of failing the whole production build.
    orphaned_layers: list[tuple[str, Any]] = [
        ("disorder", build_disorder_kg),
        ("allen_connectivity", build_allen_connectivity_kg),
        ("paradigm", build_paradigm_kg),
        ("oscillation", build_oscillation_kg),
        ("species_homology", build_homology_kg),
        ("hcp_connectivity", build_hcp_kg),
        ("scholarpedia", build_scholarpedia_kg),
        ("concept", build_concept_kg),
        # paper_links takes the in-progress graph (unlike the other layers
        # above, which are self-contained) so it can scope itself to
        # datasets that actually exist in the corpus being built, rather
        # than blindly injecting edges for every dataset in the real,
        # environment-wide paper_dataset_links.jsonl — see
        # paper_node_builder.build_paper_nodes_and_links's docstring.
        (
            "paper_links",
            lambda: build_paper_nodes_and_links(KnowledgeGraph(nodes=nodes, edges=edges)),
        ),
        # citations must run after paper_links (list order = merge order, see
        # loop below) so the openalex paper nodes it scopes itself to already
        # exist in `nodes`/`edges` by the time this lambda executes. See
        # build_citation_edges_for_graph's docstring for the two real bugs
        # (node ID prefix, unscoped known-ids) this fixes vs. the older
        # build_citation_kg() still used standalone by compose_kg.py.
        (
            "citations",
            lambda: build_citation_kg_for_graph(KnowledgeGraph(nodes=nodes, edges=edges)),
        ),
    ]
    layer_edge_counts: dict[str, int] = {}
    for layer_name, builder_fn in orphaned_layers:
        try:
            layer_kg = builder_fn()
        except Exception as exc:  # noqa: BLE001 - degrade gracefully, this is a build script
            print(f"  WARNING: {layer_name} layer unavailable: {exc}")
            continue
        for node in layer_kg.nodes.values():
            nodes.setdefault(node.node_id, node)
        for edge in layer_kg.edges.values():
            edges.setdefault(edge.edge_id, edge)
        layer_edge_counts[layer_name] = len(layer_kg.edges)
        print(f"  {layer_name} layer merged: {len(layer_kg.nodes)} nodes, {len(layer_kg.edges)} edges")

    # ── Resolve dangling edges ──────────────────────────────────────────────
    # The builder modules above use inconsistent, independently-authored node
    # id conventions and frequently reference concepts (regions, circuits,
    # topics, oscillation bands, paradigms) that no single layer creates.
    # Rather than hand-reconciling each builder's vocabulary, create minimal
    # placeholder nodes for any edge endpoint still missing after all layers
    # are merged, so the shipped graph never has dangling edges regardless of
    # which layers are present. See neural_search.graph.schema.resolve_dangling_edges.
    merged = KnowledgeGraph(nodes=nodes, edges=edges, metadata={
        "builder": BUILDER_NAME,
        "builder_version": BUILDER_VERSION,
        "corpus": str(CORPUS_PATH),
        "record_count": len(corpus),
        "taxonomy_requirement_edges": len(taxonomy_kg.edges),
        "methods_kg_nodes": len(methods_kg.nodes),
        "method_supports_analysis_edges": len(registry_kg.edges),
        "reanalysis_candidate_edges": len(candidate_edges),
        "reanalysis_bridge_edges": len(bridge_edges),
        "reinterpretation_candidate_edges": len(reinterpretation_edges),
        "evidence_tier_file_validated_upgrades": evidence_tier_upgrades,
        **{f"{name}_edges": count for name, count in layer_edge_counts.items()},
    })

    # Retraction-status enrichment (2026-07-02): a durable, build-time step
    # that only reads a precomputed artifact (scripts/check_paper_retraction_status.py,
    # a separate live-network script, not run on every build) -- adds a
    # property to existing paper nodes, never a new node/edge, so it cannot
    # introduce a dangling-edge or graph_degree risk. A no-op if that
    # artifact doesn't exist yet.
    merged = attach_retraction_status(merged)
    retracted_count = sum(
        1
        for n in merged.nodes.values()
        if n.node_type == "paper" and n.properties.get("retraction_status", {}).get("status") == "retracted"
    )
    print(f"  Papers flagged retracted: {retracted_count}")

    # Reprocessing-candidate enrichment (2026-07-04): same durable,
    # artifact-only pattern as attach_retraction_status above -- reads
    # scripts/validate_top_reanalysis_suggestions.py's output (which already
    # streams nwb_version per DANDI row), flags datasets whose NWB schema
    # predates the heuristic threshold. Only a node property, never a new
    # node/edge. A no-op if that artifact doesn't exist yet.
    merged = attach_reprocessing_candidate_status(merged)
    reprocessing_candidate_count = sum(
        1 for n in merged.nodes.values() if n.properties.get("reprocessing_candidate")
    )
    print(f"  Datasets flagged as reprocessing candidates: {reprocessing_candidate_count}")

    resolved, stub_count = resolve_dangling_edges(merged)
    print(f"  Dangling-edge placeholder nodes created: {stub_count}")
    resolved.metadata["stub_nodes_created"] = stub_count
    return resolved


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
