"""Bridge between FieldStateGraphStore and the retrieval pipeline.

Provides lazy-loaded store access and memory-graph scoring for search results.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from neural_search.field_state.graph_store import FieldStateGraphStore

logger = logging.getLogger(__name__)

# Score bounds
_MIN_SCORE = -0.20
_MAX_SCORE = 0.40

# Per-match positive weights
_MODALITY_MATCH_SCORE = 0.10
_SPECIES_MATCH_SCORE = 0.10
_REGION_MATCH_SCORE = 0.05
_AFFORDANCE_MATCH_SCORE = 0.08
_RAW_SIGNAL_BONUS = 0.06

# Penalty weights
_LACKS_EVIDENCE_PENALTY = 0.04
_LACKS_EVIDENCE_MAX_PENALTY = 0.12
_CONTRAINDICATED_PENALTY = 0.10

# Positive edge types used to look up dataset neighbors
_MODALITY_EDGE = "dataset_has_modality"
_SPECIES_EDGE = "dataset_has_species"
_REGION_EDGE = "dataset_records_region"
_AFFORDANCE_EDGE = "dataset_supports_analysis"
_RAW_SIGNAL_EDGE = "dataset_has_raw_signal"

# Penalty edge types
_LACKS_EVIDENCE_EDGE = "dataset_lacks_required_evidence"
_CONTRAINDICATED_EDGE = "dataset_contraindicated_for"


@lru_cache(maxsize=4)
def load_memory_graph_store(
    nodes_path: str,
    edges_path: str,
) -> FieldStateGraphStore | None:
    """Load and cache a FieldStateGraphStore from JSONL artifact files.

    Returns None if either file does not exist, so callers can gate
    gracefully without crashing.
    """
    nodes = Path(nodes_path)
    edges = Path(edges_path)
    if not nodes.exists() or not edges.exists():
        logger.debug(
            "memory graph artifacts not found at %s / %s — skipping",
            nodes_path,
            edges_path,
        )
        return None
    try:
        store = FieldStateGraphStore.from_jsonl(nodes, edges)
        logger.debug(
            "loaded memory graph store: %d nodes, %d edges",
            store.node_count,
            store.edge_count,
        )
        return store
    except Exception:
        logger.exception("failed to load memory graph store — skipping")
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _node_terms(node: Any) -> set[str]:
    """Return a lower-cased set of the node label and all aliases."""
    terms: set[str] = set()
    label = getattr(node, "label", None)
    if label:
        terms.add(str(label).lower())
    for alias in getattr(node, "aliases", []):
        if alias:
            terms.add(str(alias).lower())
    return terms


def _query_terms(values: list[str]) -> set[str]:
    """Return a lower-cased set of query values."""
    return {v.lower() for v in values if v}


def _has_match(store: FieldStateGraphStore, dataset_node_id: str, edge_type: str, query_values: list[str]) -> bool:
    """Return True if any neighbor reachable via edge_type matches a query value."""
    if not query_values:
        return False
    q_terms = _query_terms(query_values)
    for _edge, neighbor in store.get_neighbors(dataset_node_id, edge_types=[edge_type]):
        if _node_terms(neighbor) & q_terms:
            return True
    return False


def _match_count(store: FieldStateGraphStore, dataset_node_id: str, edge_type: str, query_values: list[str]) -> int:
    """Count how many distinct query values match a neighbor via edge_type."""
    if not query_values:
        return 0
    q_terms = _query_terms(query_values)
    matched: set[str] = set()
    for _edge, neighbor in store.get_neighbors(dataset_node_id, edge_types=[edge_type]):
        node_t = _node_terms(neighbor)
        hit = node_t & q_terms
        matched.update(hit)
    return len(matched)


# ---------------------------------------------------------------------------
# Public scoring function
# ---------------------------------------------------------------------------


def compute_memory_graph_score(
    store: FieldStateGraphStore,
    result: Any,
    parsed_query: Mapping[str, Any],
) -> float:
    """Compute a memory-graph evidence score for a single search result.

    Score contributions:
    - +0.10 per matching modality (query vs dataset modality nodes)
    - +0.10 per matching species
    - +0.05 per matching brain region
    - +0.08 per matching affordance
    - +0.06 if query asks for raw data and dataset has a raw_signal edge
    - -0.04 per lacks_required_evidence edge (capped at -0.12)
    - -0.10 per contraindicated_for edge relevant to query terms

    Final score is clamped to [-0.20, 0.40].

    Parameters
    ----------
    store:
        The loaded FieldStateGraphStore.
    result:
        A SearchResult; must have a ``dataset_id`` attribute.
    parsed_query:
        The parsed query dict from ``parse_query``; keys used:
        ``modalities``, ``species``, ``brain_regions``, ``affordances``,
        and the raw query text under ``_raw_query`` (optional).

    Returns
    -------
    float
        Score delta in [-0.20, 0.40].
    """
    dataset_id = str(getattr(result, "dataset_id", "") or "")
    if not dataset_id:
        return 0.0

    dataset_node = store.query_by_dataset_id(dataset_id)
    if dataset_node is None:
        return 0.0

    node_id = dataset_node.node_id
    score = 0.0

    # --- Positive signals ---

    modalities: list[str] = list(parsed_query.get("modalities", []))
    for _modality in modalities:
        if _has_match(store, node_id, _MODALITY_EDGE, [_modality]):
            score += _MODALITY_MATCH_SCORE

    species: list[str] = list(parsed_query.get("species", []))
    for _sp in species:
        if _has_match(store, node_id, _SPECIES_EDGE, [_sp]):
            score += _SPECIES_MATCH_SCORE

    regions: list[str] = list(parsed_query.get("brain_regions", []))
    for _region in regions:
        if _has_match(store, node_id, _REGION_EDGE, [_region]):
            score += _REGION_MATCH_SCORE

    affordances: list[str] = list(parsed_query.get("affordances", []))
    for _affordance in affordances:
        if _has_match(store, node_id, _AFFORDANCE_EDGE, [_affordance]):
            score += _AFFORDANCE_MATCH_SCORE

    # Raw signal bonus: look for "raw" in modalities list or query text
    raw_indicators = {"raw"}
    all_query_terms = _query_terms(modalities + list(parsed_query.get("tasks", [])))
    query_text = str(parsed_query.get("_raw_query", "")).lower()
    if raw_indicators & all_query_terms or "raw" in query_text:
        neighbors = store.get_neighbors(node_id, edge_types=[_RAW_SIGNAL_EDGE])
        if neighbors:
            score += _RAW_SIGNAL_BONUS

    # --- Penalty signals ---

    lacks_edges = store.get_neighbors(node_id, edge_types=[_LACKS_EVIDENCE_EDGE])
    evidence_penalty = min(len(lacks_edges) * _LACKS_EVIDENCE_PENALTY, _LACKS_EVIDENCE_MAX_PENALTY)
    score -= evidence_penalty

    # Contraindicated penalties: check if any contraindicated neighbor matches query terms
    all_q = _query_terms(modalities + species + regions + affordances)
    for _edge, contra_node in store.get_neighbors(node_id, edge_types=[_CONTRAINDICATED_EDGE]):
        if _node_terms(contra_node) & all_q:
            score -= _CONTRAINDICATED_PENALTY

    return max(_MIN_SCORE, min(_MAX_SCORE, score))
