"""Provenance-aware scientific knowledge graph primitives."""

from neural_search.graph.builder import (
    build_dataset_subgraph,
    build_graph_from_records,
    build_paper_subgraph,
    build_taxonomy_requirement_subgraph,
    dataset_node_id,
    merge_graphs,
    paper_node_id,
    split_records,
)
from neural_search.graph.experimental_design import (
    DesignMatch,
    ExperimentalDesign,
    find_datasets_for_experimental_design,
    get_experimental_design,
    list_experimental_designs,
    load_experimental_design_seeds,
)
from neural_search.graph.quality import (
    GraphQualityIssue,
    GraphQualityReport,
    audit_graph_quality,
)
from neural_search.graph.query import (
    DEFAULT_WEIGHTS,
    RelatedItem,
    explain_connection,
    find_datasets_for_analysis,
    find_datasets_for_paper,
    find_datasets_for_task,
    find_datasets_with_constraints,
    find_nodes_by_label,
    find_nodes_by_type,
    find_papers_for_dataset,
    find_paths,
    get_edges_between,
    get_neighbors,
    get_node,
    rank_related_datasets,
    rank_related_papers,
)
from neural_search.graph.schema import (
    SUPPORTED_EDGE_TYPES,
    SUPPORTED_NODE_TYPES,
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    graph_from_dict,
    graph_to_dict,
    make_edge_id,
    make_node_id,
    normalize_edge_type,
    normalize_node_type,
    read_graph_json,
    read_graph_jsonl,
    resolve_dangling_edges,
    validate_graph,
    write_graph_json,
    write_graph_jsonl,
)
from neural_search.graph.search_features import (
    DEFAULT_GRAPH_SEARCH_WEIGHTS,
    compute_graph_features_for_result,
    graph_context_score,
    load_graph_if_exists,
)
from neural_search.graph.semantic_edges import (
    SemanticEdgeConfig,
    SemanticEdgeResult,
    add_semantic_edges_to_graph,
    build_concept_similarity_edges,
    build_semantic_dataset_edges,
    get_semantic_neighbors,
    load_and_add_semantic_edges,
)
from neural_search.graph.transitive import (
    TransitiveMatch,
    expand_query_with_graph,
    find_related_affordances,
    find_related_tasks,
    find_transitive_concepts,
    get_transitive_boost,
)

_REPORT_EXPORTS = {
    "generate_graph_reports",
    "graph_gap_report",
    "graph_linking_report",
    "graph_requirement_report",
    "graph_scientific_coverage_report",
    "graph_summary_report",
    "write_graph_reports",
}


def __getattr__(name: str):
    if name in _REPORT_EXPORTS:
        from neural_search.graph import reports

        return getattr(reports, name)
    raise AttributeError(f"module 'neural_search.graph' has no attribute {name!r}")


__all__ = [
    # Schema
    "SUPPORTED_EDGE_TYPES",
    "SUPPORTED_NODE_TYPES",
    "GraphEvidence",
    "GraphQualityIssue",
    "GraphQualityReport",
    "KnowledgeGraph",
    "KnowledgeGraphEdge",
    "KnowledgeGraphNode",
    "graph_from_dict",
    "graph_to_dict",
    "make_edge_id",
    "make_node_id",
    "normalize_edge_type",
    "normalize_node_type",
    "read_graph_json",
    "read_graph_jsonl",
    "resolve_dangling_edges",
    "validate_graph",
    "write_graph_json",
    "write_graph_jsonl",
    # Builder
    "build_dataset_subgraph",
    "build_graph_from_records",
    "build_paper_subgraph",
    "build_taxonomy_requirement_subgraph",
    "dataset_node_id",
    "merge_graphs",
    "paper_node_id",
    "split_records",
    # Query
    "DEFAULT_GRAPH_SEARCH_WEIGHTS",
    "DEFAULT_WEIGHTS",
    "RelatedItem",
    "compute_graph_features_for_result",
    "explain_connection",
    "find_datasets_for_analysis",
    "find_datasets_for_paper",
    "find_datasets_for_task",
    "find_datasets_with_constraints",
    "find_nodes_by_label",
    "find_nodes_by_type",
    "find_papers_for_dataset",
    "find_paths",
    "generate_graph_reports",
    "get_edges_between",
    "get_neighbors",
    "get_node",
    "graph_context_score",
    "graph_gap_report",
    "graph_linking_report",
    "graph_requirement_report",
    "graph_scientific_coverage_report",
    "graph_summary_report",
    "load_graph_if_exists",
    "rank_related_datasets",
    "rank_related_papers",
    "write_graph_reports",
    # Experimental Design
    "DesignMatch",
    "ExperimentalDesign",
    "find_datasets_for_experimental_design",
    "get_experimental_design",
    "list_experimental_designs",
    "load_experimental_design_seeds",
    # Transitive Matching
    "TransitiveMatch",
    "expand_query_with_graph",
    "find_related_affordances",
    "find_related_tasks",
    "find_transitive_concepts",
    "get_transitive_boost",
    "audit_graph_quality",
    # Semantic Edges
    "SemanticEdgeConfig",
    "SemanticEdgeResult",
    "add_semantic_edges_to_graph",
    "build_concept_similarity_edges",
    "build_semantic_dataset_edges",
    "get_semantic_neighbors",
    "load_and_add_semantic_edges",
]
