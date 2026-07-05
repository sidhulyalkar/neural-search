"""Build real `paper` nodes and paper<->dataset edges for the production
graph from `paper_dataset_links.jsonl`-shaped literature-linking output.

Confirmed directly against `data/graph/neural_search_graph.real_corpus.json`
(2026-07-02): the production graph has zero paper nodes today, even though
`SUPPORTED_NODE_TYPES` already includes `"paper"` and the dev/CLI-only
`neural_search/graph/builder.py` pipeline already exercises the
`paper:{source}:{source_id}` convention. This module is the first to bring
real paper nodes into the actual production pipeline
(`scripts/build_real_corpus_graph.py`), following the exact reconnection
pattern used for the 8 KG layers already wired in this session (registered
in `orphaned_layers`, merged via `resolve_dangling_edges()`).

Edge direction is `paper -> dataset` (`source_node_id=paper`,
`target_node_id=dataset`), confirmed directly against
`neural_search/graph/query.py`'s `PAPER_DATASET_EDGES` /
`find_papers_for_dataset()`, which reads these edge types as incoming edges
to the dataset. Using any other direction would make new links invisible to
existing query/search-feature code.

`edge_type` distinguishes two different strengths of claim:
- `paper_uses_dataset` for DataCite-declared relations
  (`match_method == "datacite_related_identifier"`) -- the data publisher
  directly declared the paper used/described/is-supplemented-by this
  dataset, a stronger claim than a fuzzy or DOI-string match.
- `paper_mentions_dataset` for everything else (OpenAlex doi_exact/
  title_fuzzy/title_fuzzy_local today; Crossref/Semantic Scholar/PubMed in a
  later phase) -- the paper is linked to the dataset's record by DOI or
  title similarity, a weaker claim than a publisher-declared relation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neural_search.graph.schema import (
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)
from neural_search.kg.schemas.evidence_tier import EvidenceTier

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
LITERATURE_DIR = PROJECT_ROOT / "artifacts" / "literature"

# Reads each source's link file directly rather than routing through
# neural_search.literature.merge_links.merge_link_sources() -- a dataset can
# legitimately have real matches from multiple sources (different
# paper_source values), which is a union, not a conflict needing dedup, so
# no merge step is required for correctness here. Deliberately NOT made a
# dependency of this builder: merge_link_sources() is a separate, optional
# reporting utility (used by scripts/build_artifact_manifest.py) -- putting
# it on this critical path would add a "forgot to re-run the merge" failure
# mode with no correctness benefit, unlike the durable, build-time-embedded
# evidence_tier_upgrader.apply_file_validation_upgrades() step elsewhere in
# scripts/build_real_corpus_graph.py.
DEFAULT_LINK_PATHS = (
    LITERATURE_DIR / "paper_dataset_links.jsonl",  # legacy OpenAlex master
    LITERATURE_DIR / "paper_dataset_links.datacite.jsonl",
    LITERATURE_DIR / "paper_dataset_links.crossref.jsonl",
    LITERATURE_DIR / "paper_dataset_links.semantic_scholar.jsonl",
    LITERATURE_DIR / "paper_dataset_links.pubmed.jsonl",
)

# Precomputed by scripts/check_paper_retraction_status.py (a live-network,
# Crossref-only script run separately, same pattern as
# scripts/validate_top_reanalysis_suggestions.py) -- attach_retraction_status
# below only reads this artifact, it never makes live API calls itself, so
# a production graph rebuild never blocks on network access.
DEFAULT_RETRACTION_STATUS_PATH = (
    PROJECT_ROOT / "artifacts" / "literature" / "paper_retraction_status.jsonl"
)

_NOT_REAL_MATCH_METHODS = {"not_found", "not_applicable_no_dataset_doi"}
_DATASET_USES_RELATION_METHODS = {"datacite_related_identifier"}

BUILDER_NAME = "paper_node_builder"
BUILDER_VERSION = "v1.0.0"


def _dataset_node_id(dataset_record_id: str) -> str:
    source, _, source_id = dataset_record_id.partition(":")
    return f"node:dataset:{source}:{source_id}"


def _paper_label(row: dict) -> str:
    return row.get("paper_title") or row.get("paper_doi") or row.get("paper_source_id") or "Unknown paper"


def build_paper_nodes_and_links(
    graph: KnowledgeGraph,
    link_paths: tuple[Path, ...] = DEFAULT_LINK_PATHS,
) -> KnowledgeGraph:
    """Build paper nodes + paper<->dataset edges from real (non-"not found")
    rows across the given literature-linking output files.

    Takes the in-progress `graph` (already containing all dataset nodes for
    the corpus being built) and skips any row whose dataset isn't present in
    it -- the same scoping pattern used by
    `reanalysis_bridge_builder.build_reanalysis_bridge_edges` (`if
    precedent_node_id not in graph.nodes: continue`). Without this, this
    builder would inject edges for every dataset in the real, environment-
    wide `paper_dataset_links.jsonl` regardless of what corpus is actually
    being processed -- caught by a fixture-scale test asserting exactly 2
    dataset nodes on a 2-record fixture corpus, which failed with 405
    (2 real + 403 stub nodes fabricated by resolve_dangling_edges for
    real-corpus dataset IDs the fixture never declared).

    Each file that doesn't exist yet is skipped (logged, not fatal) --
    matches the house style of the other `orphaned_layers` builders in
    `scripts/build_real_corpus_graph.py`, which degrade gracefully rather
    than failing the whole production build.
    """

    nodes: dict[str, KnowledgeGraphNode] = {}
    edges: dict[str, KnowledgeGraphEdge] = {}

    for path in link_paths:
        if not path.exists():
            log.info("paper_node_builder: %s not found, skipping", path)
            continue

        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("match_method") in _NOT_REAL_MATCH_METHODS:
                    continue

                dataset_node_id = _dataset_node_id(row["dataset_record_id"])
                if dataset_node_id not in graph.nodes:
                    continue

                paper_source = row.get("paper_source") or "openalex"
                paper_source_id = row.get("paper_source_id") or row.get("paper_openalex_id")
                if not paper_source_id:
                    continue

                paper_node_id = make_node_id("paper", paper_source, paper_source_id)
                nodes.setdefault(
                    paper_node_id,
                    KnowledgeGraphNode(
                        node_id=paper_node_id,
                        node_type="paper",
                        label=_paper_label(row),
                        source_ids=[f"{paper_source}:{paper_source_id}"],
                        properties={
                            "source": paper_source,
                            "doi": row.get("paper_doi"),
                            "year": row.get("paper_year"),
                        },
                        confidence=row.get("confidence", 0.5),
                    ),
                )

                edge_type = (
                    "paper_uses_dataset"
                    if row.get("match_method") in _DATASET_USES_RELATION_METHODS
                    else "paper_mentions_dataset"
                )
                edge_id = make_edge_id(paper_node_id, edge_type, dataset_node_id)
                edges.setdefault(
                    edge_id,
                    KnowledgeGraphEdge(
                        edge_id=edge_id,
                        source_node_id=paper_node_id,
                        target_node_id=dataset_node_id,
                        edge_type=edge_type,
                        directed=True,
                        confidence=row.get("confidence", 0.5),
                        properties={
                            "match_method": row.get("match_method"),
                            "paper_source": paper_source,
                            "evidence_tier": (
                                EvidenceTier.SOURCE_DECLARED.value
                                if paper_source == "datacite"
                                else None
                            ),
                        },
                    ),
                )

    return KnowledgeGraph(nodes=nodes, edges=edges)


# Per-graph-instance cache, same pattern as search_features.py's
# _ALIAS_INDEX_CACHE -- keyed by id(graph) since KnowledgeGraph instances are
# immutable (model_copy always returns a new object), so a stale entry can
# only ever be for a graph that's no longer reachable.
_DOI_INDEX_CACHE: dict[int, dict[str, str]] = {}


def build_doi_to_paper_node_id_index(graph: KnowledgeGraph) -> dict[str, str]:
    """Index paper node_id by DOI, for cheap retraction/evidence-tier lookups
    at search/card time without a linear scan over every paper node."""

    gid = id(graph)
    if gid in _DOI_INDEX_CACHE:
        return _DOI_INDEX_CACHE[gid]
    index: dict[str, str] = {}
    for node_id, node in graph.nodes.items():
        if node.node_type != "paper":
            continue
        doi = node.properties.get("doi")
        if doi and doi not in index:
            index[doi] = node_id
    _DOI_INDEX_CACHE[gid] = index
    return index


def get_paper_trust_signals(
    graph: KnowledgeGraph,
    *,
    doi: str | None,
    dataset_node_id: str | None = None,
) -> dict[str, Any]:
    """Best-effort retraction status + link evidence tier for a paper.

    Returns `{}` when the paper has no DOI, isn't in the graph, or carries no
    trust signal yet -- callers should treat an empty dict as "nothing
    additional to show", never as an error. This is the one place search
    results and dataset cards should go to surface `retraction_status`
    (currently a graph-node-only property, see `attach_retraction_status`)
    and a paper<->dataset link's `evidence_tier` (currently edge-only,
    written by `build_paper_nodes_and_links`) to an actual user.
    """

    if not doi:
        return {}
    doi_index = build_doi_to_paper_node_id_index(graph)
    paper_node_id = doi_index.get(doi)
    if paper_node_id is None:
        return {}

    node = graph.nodes[paper_node_id]
    signals: dict[str, Any] = {}
    retraction_status = node.properties.get("retraction_status")
    if retraction_status:
        signals["retraction_status"] = retraction_status

    if dataset_node_id:
        # Import locally to avoid a module-level cycle: query.py doesn't
        # import paper_node_builder, but keeping this import scoped makes
        # that invariant obvious rather than relying on import order.
        from neural_search.graph.query import get_edges_between

        for edge in get_edges_between(graph, paper_node_id, dataset_node_id):
            tier = edge.properties.get("evidence_tier")
            if tier:
                signals["evidence_tier"] = tier
                break

    return signals


def attach_retraction_status(
    graph: KnowledgeGraph,
    retraction_path: Path = DEFAULT_RETRACTION_STATUS_PATH,
) -> KnowledgeGraph:
    """Set `properties["retraction_status"]` on paper nodes whose DOI appears
    in a precomputed retraction-status artifact (see
    `scripts/check_paper_retraction_status.py`).

    Immutable: returns a new KnowledgeGraph, does not mutate the input.
    A no-op (returns `graph` unchanged) if the artifact doesn't exist yet --
    matches the house style of `evidence_tier_upgrader.apply_file_validation_upgrades`.
    Only ever adds a property to an existing node; never creates new nodes or
    edges, so this cannot introduce a dangling-edge or graph_degree risk.
    """

    if not retraction_path.exists():
        log.info("paper_node_builder: %s not found, skipping retraction status", retraction_path)
        return graph

    status_by_doi: dict[str, dict] = {}
    with retraction_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("doi"):
                status_by_doi[row["doi"]] = row

    if not status_by_doi:
        return graph

    updated_nodes = dict(graph.nodes)
    for node_id, node in graph.nodes.items():
        if node.node_type != "paper":
            continue
        doi = node.properties.get("doi")
        status = status_by_doi.get(doi) if doi else None
        if status is None:
            continue
        updated_nodes[node_id] = node.model_copy(
            update={"properties": {**node.properties, "retraction_status": status}}
        )

    return graph.model_copy(update={"nodes": updated_nodes})
