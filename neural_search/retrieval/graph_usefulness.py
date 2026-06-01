"""Graph-derived usefulness signals.

Functions here accept plain dicts (compatible with KnowledgeGraph.model_dump()
output) so they work without importing heavy schema classes.
"""
from __future__ import annotations

from collections import defaultdict


def _jaccard(a: list[str], b: list[str]) -> float:
    sa = {x.lower() for x in a}
    sb = {x.lower() for x in b}
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def affordance_overlap(dataset_a: dict, dataset_b: dict) -> float:
    """Jaccard similarity of analysis affordances."""
    return _jaccard(
        dataset_a.get("affordances", []),
        dataset_b.get("affordances", []),
    )


def pipeline_overlap(dataset_a: dict, dataset_b: dict) -> float:
    """Jaccard similarity of data standards (pipeline compatibility proxy)."""
    return _jaccard(
        dataset_a.get("data_standards", []),
        dataset_b.get("data_standards", []),
    )


def complementarity_score(dataset_a: dict, dataset_b: dict) -> float:
    """Score how complementary two datasets are.

    High when each dataset has affordances the other lacks AND they share some
    common ground (pure disjoint is noise; pure overlap is redundant).
    """
    sa = {x.lower() for x in dataset_a.get("affordances", [])}
    sb = {x.lower() for x in dataset_b.get("affordances", [])}
    if not sa or not sb:
        return 0.0
    union = sa | sb
    intersection = sa & sb
    only_a = sa - sb
    only_b = sb - sa

    if not union:
        return 0.0

    unique_fraction = (len(only_a) + len(only_b)) / len(union)
    overlap_fraction = len(intersection) / len(union)

    if overlap_fraction == 0.0:
        return unique_fraction * 0.3
    return min(1.0, unique_fraction * 0.7 + overlap_fraction * 0.3)


def normalized_metapath_score(
    graph: dict | None,
    source_id: str,
    target_id: str,
    metapath_type: str,
) -> float:
    """PathSim-inspired similarity normalized against hub-node degree.

    Returns 0.0 when graph is None or nodes are absent.
    """
    if not graph:
        return 0.0
    edges = graph.get("edges", {})

    neighbors: dict[str, set[str]] = defaultdict(set)
    for edge in edges.values():
        if edge.get("edge_type") != metapath_type:
            continue
        src = edge.get("source_node_id", "")
        tgt = edge.get("target_node_id", "")
        if src:
            neighbors[src].add(tgt)
        if tgt:
            neighbors[tgt].add(src)

    n_src = neighbors.get(source_id, set())
    n_tgt = neighbors.get(target_id, set())

    if not n_src or not n_tgt:
        return 0.0

    shared = n_src & n_tgt
    if not shared:
        return 0.0

    # PathSim: 2 * |shared| / (|n_src| + |n_tgt|)
    # Automatically down-weights hub nodes because they inflate the denominator.
    return 2.0 * len(shared) / (len(n_src) + len(n_tgt))


def graph_usefulness_features(
    query_context: dict | None,
    candidate: dict | None,
    graph: dict | None,
) -> dict[str, float]:
    """Compute all graph-derived usefulness features as a flat dict."""
    q = query_context or {}
    c = candidate or {}

    aff = affordance_overlap(q, c)
    pip = pipeline_overlap(q, c)
    comp = complementarity_score(q, c)

    q_id = q.get("dataset_id", "")
    c_id = c.get("dataset_id", "")
    meta = normalized_metapath_score(graph, q_id, c_id, "dataset_has_task") if graph else 0.0

    return {
        "affordance_overlap": aff,
        "pipeline_overlap": pip,
        "complementarity": comp,
        "metapath_score": meta,
    }
