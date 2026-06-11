"""Build, write, and read the Graph-Indexed Concept Memory artifacts."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import networkx as nx

from neural_search.field_state.concept_memory.loaders import (
    LoadResult,
    load_corpus,
    load_field_state_artifacts,
    load_obsidian_notes,
    merge_load_results,
)
from neural_search.field_state.concept_memory.schema import ConceptNode, EvidenceLink

# ---------------------------------------------------------------------------
# Repo root
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]  # neural_search/field_state/concept_memory/graph_builder.py


def _repo_root() -> Path:
    return _REPO_ROOT


# ---------------------------------------------------------------------------
# Artifact paths
# ---------------------------------------------------------------------------

_ARTIFACT_DIR = Path("artifacts/field_state/concept_memory")

_CONCEPTS_JSONL = _ARTIFACT_DIR / "concepts.jsonl"
_EVIDENCE_JSONL = _ARTIFACT_DIR / "evidence_links.jsonl"
_CONCEPT_INDEX = _ARTIFACT_DIR / "concept_index.json"
_CONCEPT_GRAPH_JSON = _ARTIFACT_DIR / "concept_graph.json"
_CONCEPT_GRAPH_GRAPHML = _ARTIFACT_DIR / "concept_graph.graphml"


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_concept_graph(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
) -> nx.DiGraph:
    """Build a NetworkX DiGraph from ConceptNode and EvidenceLink lists."""
    g: nx.DiGraph = nx.DiGraph()

    for node in concepts:
        g.add_node(node.concept_id, **node.model_dump())

    for link in evidence_links:
        src = link.source_concept_id
        tgt = link.target_concept_id
        if tgt is None:
            continue
        if src not in g or tgt not in g:
            continue
        g.add_edge(src, tgt, **link.model_dump())

    return g


# ---------------------------------------------------------------------------
# Concept index builder
# ---------------------------------------------------------------------------


def build_concept_index(concepts: list[ConceptNode]) -> dict[str, Any]:
    """Build an in-memory index for fast concept lookup."""
    by_id: dict[str, dict[str, Any]] = {}
    by_type: dict[str, list[str]] = defaultdict(list)
    by_canonical_name: dict[str, str] = {}
    by_alias: dict[str, str] = {}
    total_by_type: dict[str, int] = defaultdict(int)

    for node in concepts:
        dumped: dict[str, Any] = node.model_dump()
        by_id[node.concept_id] = dumped
        by_type[node.concept_type].append(node.concept_id)
        total_by_type[node.concept_type] += 1
        by_canonical_name[node.canonical_name.lower()] = node.concept_id
        for alias in node.aliases:
            by_alias[alias.lower()] = node.concept_id

    return {
        "by_id": by_id,
        "by_type": dict(by_type),
        "by_canonical_name": by_canonical_name,
        "by_alias": by_alias,
        "total_concepts": len(concepts),
        "total_by_type": dict(total_by_type),
    }


# ---------------------------------------------------------------------------
# Write all artifacts
# ---------------------------------------------------------------------------


def write_concept_artifacts(
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    root: Path | None = None,
) -> dict[str, Path]:
    """Write all concept memory artifacts. Returns label→absolute path dict."""
    base = root if root is not None else _repo_root()
    paths: dict[str, Path] = {}

    # 1. concepts.jsonl
    concepts_path = base / _CONCEPTS_JSONL
    _atomic_write(concepts_path, "\n".join(c.to_jsonl() for c in concepts))
    paths["concepts"] = concepts_path

    # 2. evidence_links.jsonl
    evidence_path = base / _EVIDENCE_JSONL
    _atomic_write(evidence_path, "\n".join(lnk.to_jsonl() for lnk in evidence_links))
    paths["evidence_links"] = evidence_path

    # 3. concept_index.json
    index = build_concept_index(concepts)
    index_path = base / _CONCEPT_INDEX
    _atomic_write(index_path, json.dumps(index, indent=2))
    paths["concept_index"] = index_path

    # 4. concept_graph.json (adjacency-list format)
    graph_json = {
        "nodes": [
            {
                "concept_id": c.concept_id,
                "canonical_name": c.canonical_name,
                "concept_type": c.concept_type,
                "evidence_count": c.evidence_count,
                **{
                    k: v
                    for k, v in c.model_dump().items()
                    if k not in {"concept_id", "canonical_name", "concept_type", "evidence_count"}
                },
            }
            for c in concepts
        ],
        "edges": [
            {
                "source": lnk.source_concept_id,
                "target": lnk.target_concept_id,
                "relation_type": lnk.relation_type,
                "confidence": lnk.confidence,
                "review_status": lnk.review_status,
            }
            for lnk in evidence_links
            if lnk.target_concept_id is not None
        ],
    }
    graph_json_path = base / _CONCEPT_GRAPH_JSON
    _atomic_write(graph_json_path, json.dumps(graph_json, indent=2))
    paths["concept_graph"] = graph_json_path

    # 5. concept_graph.graphml (optional — skip if networkx graphml export fails)
    tmp_graphml: Path | None = None
    try:
        g = build_concept_graph(concepts, evidence_links)
        graphml_path = base / _CONCEPT_GRAPH_GRAPHML
        graphml_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_graphml = graphml_path.with_name(f".{graphml_path.name}.tmp")
        nx.write_graphml(g, str(tmp_graphml))
        tmp_graphml.replace(graphml_path)
        paths["concept_graph_graphml"] = graphml_path
    except Exception:  # noqa: BLE001
        # Clean up any partial temp file
        if tmp_graphml is not None:
            try:
                tmp_graphml.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass

    return paths


# ---------------------------------------------------------------------------
# Read artifacts
# ---------------------------------------------------------------------------


def read_concept_artifacts(
    root: Path | None = None,
) -> tuple[list[ConceptNode], list[EvidenceLink]]:
    """Read concepts.jsonl and evidence_links.jsonl from disk."""
    base = root if root is not None else _repo_root()

    concepts: list[ConceptNode] = []
    concepts_path = base / _CONCEPTS_JSONL
    if concepts_path.exists():
        with concepts_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        concepts.append(ConceptNode.from_jsonl(line))
                    except Exception:
                        pass

    evidence_links: list[EvidenceLink] = []
    evidence_path = base / _EVIDENCE_JSONL
    if evidence_path.exists():
        with evidence_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        evidence_links.append(EvidenceLink.from_jsonl(line))
                    except Exception:
                        pass

    return concepts, evidence_links


# ---------------------------------------------------------------------------
# Load index
# ---------------------------------------------------------------------------


def load_concept_index(root: Path | None = None) -> dict[str, Any]:
    """Load concept_index.json from disk."""
    base = root if root is not None else _repo_root()
    index_path = base / _CONCEPT_INDEX
    return cast("dict[str, Any]", json.loads(index_path.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Neighborhood query
# ---------------------------------------------------------------------------


def get_concept_neighborhood(
    graph: nx.DiGraph,
    concept_id_str: str,
    depth: int = 2,
) -> dict[str, Any]:
    """Return the neighborhood of a concept up to `depth` hops (in + out)."""
    # Build undirected view for ego_graph so we get both directions
    undirected = graph.to_undirected()
    if concept_id_str not in undirected:
        return {
            "center": concept_id_str,
            "nodes": [],
            "edges": [],
            "depth": depth,
        }

    ego = nx.ego_graph(undirected, concept_id_str, radius=depth)
    node_ids = set(ego.nodes())

    nodes = [
        {
            "concept_id": nid,
            "canonical_name": graph.nodes[nid].get("canonical_name", ""),
            "concept_type": graph.nodes[nid].get("concept_type", ""),
        }
        for nid in node_ids
        if nid in graph
    ]

    edges = [
        {
            "source": u,
            "target": v,
            "relation_type": graph.edges[u, v].get("relation_type", ""),
        }
        for u, v in graph.edges()
        if u in node_ids and v in node_ids
    ]

    return {
        "center": concept_id_str,
        "nodes": nodes,
        "edges": edges,
        "depth": depth,
    }


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------


def build_full_concept_memory(
    field: str = "neuroscience_dataset_reuse",
    vault_path: Path | None = None,
    corpus_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, Path]:
    """Orchestrate the full concept memory build from all available sources."""
    base = root if root is not None else _repo_root()

    # 1. Field-state artifacts (claims, gaps, opportunities)
    field_state_result = load_field_state_artifacts(base)

    # 2. Corpus datasets
    effective_corpus = corpus_path
    if effective_corpus is None:
        default_corpus = base / "data" / "corpus" / "normalized" / "combined_corpus.jsonl"
        if default_corpus.exists():
            effective_corpus = default_corpus
    if effective_corpus is not None:
        try:
            corpus_result = load_corpus(effective_corpus)
        except Exception as exc:  # noqa: BLE001
            corpus_result = LoadResult(warnings=[f"Corpus load failed: {exc}"])
    else:
        corpus_result = LoadResult()

    # 3. Obsidian notes (optional)
    if vault_path is not None:
        obsidian_result = load_obsidian_notes(vault_path, field)
    else:
        obsidian_result = LoadResult()

    # 4. Merge all results
    merged = merge_load_results([field_state_result, corpus_result, obsidian_result])
    concepts = merged.concepts
    evidence_links = merged.evidence_links

    # 5. Write artifacts
    paths = write_concept_artifacts(concepts, evidence_links, base)

    # 6. Print summary
    type_counts: dict[str, int] = defaultdict(int)
    for c in concepts:
        type_counts[c.concept_type] += 1

    print("Concept memory built:")
    print(f"  total concepts     : {len(concepts)}")
    print(f"  total evidence links: {len(evidence_links)}")
    print("  by type:")
    for ctype, count in sorted(type_counts.items()):
        print(f"    {ctype:<30} {count}")
    if merged.warnings:
        print(f"  warnings ({len(merged.warnings)}):")
        for w in merged.warnings[:10]:
            print(f"    {w}")
        if len(merged.warnings) > 10:
            print(f"    ... and {len(merged.warnings) - 10} more")

    return paths
