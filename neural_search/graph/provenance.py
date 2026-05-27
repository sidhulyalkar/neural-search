"""Source-specific provenance tracking for knowledge graphs.

This module provides functionality to:
1. Track which source each node/edge originated from
2. Generate reports on source balance and coverage
3. Add source quality metrics to graph metadata
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from neural_search.graph.schema import KnowledgeGraph


@dataclass
class SourceStats:
    """Statistics for a single data source."""

    source_name: str
    node_count: int = 0
    edge_count: int = 0
    dataset_count: int = 0
    paper_count: int = 0
    concept_node_count: int = 0
    node_types: dict[str, int] = field(default_factory=dict)
    edge_types: dict[str, int] = field(default_factory=dict)
    avg_confidence: float = 0.0
    low_confidence_count: int = 0


@dataclass
class ProvenanceReport:
    """Comprehensive provenance report for a knowledge graph."""

    total_nodes: int
    total_edges: int
    source_stats: dict[str, SourceStats]
    source_balance_score: float  # 0-1, higher = more balanced
    dominant_source: str
    underrepresented_sources: list[str]
    node_type_by_source: dict[str, dict[str, int]]
    edge_type_by_source: dict[str, dict[str, int]]
    warnings: list[str]


def _extract_source_from_node(node: Any) -> str:
    """Extract the primary source from a node."""
    # Check properties for source
    props = node.properties if hasattr(node, "properties") else {}

    if "source" in props:
        return str(props["source"])

    # Check source_ids for patterns
    source_ids = node.source_ids if hasattr(node, "source_ids") else []
    for sid in source_ids:
        sid_str = str(sid).lower()
        if "dandi" in sid_str:
            return "dandi"
        if "openneuro" in sid_str:
            return "openneuro"
        if "openalex" in sid_str:
            return "openalex"
        if "allen" in sid_str:
            return "allen"
        if "nemo" in sid_str:
            return "nemo"

    # Check node_id patterns
    node_id = str(node.node_id) if hasattr(node, "node_id") else ""
    if "dandi" in node_id.lower():
        return "dandi"
    if "openneuro" in node_id.lower():
        return "openneuro"

    # Default based on node type
    node_type = str(node.node_type) if hasattr(node, "node_type") else ""
    if node_type in {"task", "modality", "brain_region", "behavioral_event", "species"}:
        return "taxonomy"
    if node_type == "analysis_affordance":
        return "taxonomy"

    return "unknown"


def _extract_source_from_edge(edge: Any, nodes: dict[str, Any]) -> str:
    """Extract the primary source from an edge based on its source node."""
    source_node_id = str(edge.source_node_id) if hasattr(edge, "source_node_id") else ""

    if source_node_id in nodes:
        return _extract_source_from_node(nodes[source_node_id])

    # Fallback to edge type patterns
    edge_type = str(edge.edge_type) if hasattr(edge, "edge_type") else ""
    if edge_type.startswith("analysis_requires"):
        return "taxonomy"

    return "unknown"


def analyze_graph_provenance(graph: KnowledgeGraph) -> ProvenanceReport:
    """Analyze source provenance across the entire graph.

    Args:
        graph: Knowledge graph to analyze

    Returns:
        ProvenanceReport with comprehensive statistics
    """
    source_stats: dict[str, SourceStats] = {}
    node_type_by_source: dict[str, dict[str, int]] = {}
    edge_type_by_source: dict[str, dict[str, int]] = {}
    warnings: list[str] = []

    # Analyze nodes
    for node in graph.nodes.values():
        source = _extract_source_from_node(node)
        node_type = str(node.node_type)
        confidence = float(node.confidence) if hasattr(node, "confidence") else 1.0

        if source not in source_stats:
            source_stats[source] = SourceStats(source_name=source)
            node_type_by_source[source] = {}

        stats = source_stats[source]
        stats.node_count += 1

        if node_type == "dataset":
            stats.dataset_count += 1
        elif node_type == "paper":
            stats.paper_count += 1
        else:
            stats.concept_node_count += 1

        stats.node_types[node_type] = stats.node_types.get(node_type, 0) + 1
        node_type_by_source[source][node_type] = node_type_by_source[source].get(node_type, 0) + 1

        if confidence < 0.5:
            stats.low_confidence_count += 1

    # Analyze edges
    for edge in graph.edges.values():
        source = _extract_source_from_edge(edge, graph.nodes)
        edge_type = str(edge.edge_type)

        if source not in source_stats:
            source_stats[source] = SourceStats(source_name=source)
            edge_type_by_source[source] = {}

        if source not in edge_type_by_source:
            edge_type_by_source[source] = {}

        stats = source_stats[source]
        stats.edge_count += 1
        stats.edge_types[edge_type] = stats.edge_types.get(edge_type, 0) + 1
        edge_type_by_source[source][edge_type] = edge_type_by_source[source].get(edge_type, 0) + 1

    # Calculate average confidence per source
    for source, stats in source_stats.items():
        total_confidence = 0.0
        count = 0
        for node in graph.nodes.values():
            if _extract_source_from_node(node) == source:
                total_confidence += float(node.confidence) if hasattr(node, "confidence") else 1.0
                count += 1
        stats.avg_confidence = round(total_confidence / count, 3) if count > 0 else 0.0

    # Calculate source balance
    dataset_sources = [s for s, stats in source_stats.items() if stats.dataset_count > 0]
    if dataset_sources:
        dataset_counts = [source_stats[s].dataset_count for s in dataset_sources]
        total_datasets = sum(dataset_counts)
        if total_datasets > 0:
            # Calculate normalized entropy as balance score
            import math
            proportions = [c / total_datasets for c in dataset_counts]
            entropy = -sum(p * math.log(p) if p > 0 else 0 for p in proportions)
            max_entropy = math.log(len(dataset_sources)) if len(dataset_sources) > 1 else 1
            balance_score = entropy / max_entropy if max_entropy > 0 else 1.0
        else:
            balance_score = 0.0
    else:
        balance_score = 0.0

    # Find dominant and underrepresented sources
    sorted_sources = sorted(
        [(s, stats.dataset_count) for s, stats in source_stats.items() if stats.dataset_count > 0],
        key=lambda x: x[1],
        reverse=True,
    )
    dominant_source = sorted_sources[0][0] if sorted_sources else "none"

    total_datasets = sum(stats.dataset_count for stats in source_stats.values())
    underrepresented = []
    for source, count in sorted_sources:
        if total_datasets > 0 and count / total_datasets < 0.1:  # Less than 10%
            underrepresented.append(source)

    # Generate warnings
    if balance_score < 0.5:
        warnings.append(f"Source imbalance detected: {dominant_source} dominates the corpus")

    for source, stats in source_stats.items():
        if stats.low_confidence_count > stats.node_count * 0.3:
            warnings.append(f"High proportion of low-confidence nodes from {source}")

    if len(underrepresented) > 0:
        warnings.append(f"Underrepresented sources: {', '.join(underrepresented)}")

    return ProvenanceReport(
        total_nodes=len(graph.nodes),
        total_edges=len(graph.edges),
        source_stats=source_stats,
        source_balance_score=round(balance_score, 3),
        dominant_source=dominant_source,
        underrepresented_sources=underrepresented,
        node_type_by_source=node_type_by_source,
        edge_type_by_source=edge_type_by_source,
        warnings=warnings,
    )


def format_provenance_report(report: ProvenanceReport) -> str:
    """Format a provenance report for human reading."""
    lines = []
    lines.append("=" * 60)
    lines.append("GRAPH PROVENANCE REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Total Nodes: {report.total_nodes}")
    lines.append(f"Total Edges: {report.total_edges}")
    lines.append(f"Source Balance Score: {report.source_balance_score:.2%}")
    lines.append(f"Dominant Source: {report.dominant_source}")
    lines.append("")

    lines.append("Source Statistics:")
    for source, stats in sorted(report.source_stats.items(), key=lambda x: -x[1].dataset_count):
        lines.append(f"\n  {source.upper()}:")
        lines.append(f"    Datasets: {stats.dataset_count}")
        lines.append(f"    Papers: {stats.paper_count}")
        lines.append(f"    Concept Nodes: {stats.concept_node_count}")
        lines.append(f"    Total Nodes: {stats.node_count}")
        lines.append(f"    Total Edges: {stats.edge_count}")
        lines.append(f"    Avg Confidence: {stats.avg_confidence:.2%}")
        if stats.low_confidence_count > 0:
            lines.append(f"    Low Confidence: {stats.low_confidence_count}")

    if report.warnings:
        lines.append("\n" + "-" * 40)
        lines.append("WARNINGS:")
        for warning in report.warnings:
            lines.append(f"  - {warning}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def add_provenance_metadata(graph: KnowledgeGraph) -> None:
    """Add provenance statistics to graph metadata in place.

    Args:
        graph: Knowledge graph to modify
    """
    report = analyze_graph_provenance(graph)

    graph.metadata["provenance"] = {
        "source_balance_score": report.source_balance_score,
        "dominant_source": report.dominant_source,
        "underrepresented_sources": report.underrepresented_sources,
        "source_dataset_counts": {
            source: stats.dataset_count
            for source, stats in report.source_stats.items()
        },
        "source_paper_counts": {
            source: stats.paper_count
            for source, stats in report.source_stats.items()
        },
        "warnings": report.warnings,
    }


def get_nodes_by_source(
    graph: KnowledgeGraph,
    source: str,
    node_type: str | None = None,
) -> list[Any]:
    """Get all nodes from a specific source.

    Args:
        graph: Knowledge graph
        source: Source name to filter by
        node_type: Optional node type to filter

    Returns:
        List of matching nodes
    """
    results = []
    for node in graph.nodes.values():
        if _extract_source_from_node(node) == source:
            if node_type is None or node.node_type == node_type:
                results.append(node)
    return results


def get_source_coverage(
    graph: KnowledgeGraph,
    concept_type: str,
) -> dict[str, list[str]]:
    """Get coverage of a concept type across sources.

    Args:
        graph: Knowledge graph
        concept_type: Node type to check (e.g., "task", "modality")

    Returns:
        Dict mapping source -> list of concept labels
    """
    coverage: dict[str, set[str]] = {}

    for node in graph.nodes.values():
        if node.node_type != concept_type:
            continue

        source = _extract_source_from_node(node)
        if source not in coverage:
            coverage[source] = set()
        coverage[source].add(node.label)

    return {source: sorted(labels) for source, labels in coverage.items()}
