"""Quality gates for provenance-aware Neural Search graphs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from neural_search.graph.schema import KnowledgeGraph

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class GraphQualityIssue:
    """One graph quality finding."""

    code: str
    severity: Severity
    message: str
    node_id: str | None = None
    edge_id: str | None = None
    node_type: str | None = None
    edge_type: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class GraphQualityReport:
    """Deterministic graph QA output."""

    node_count: int
    edge_count: int
    issue_count: int
    error_count: int
    warning_count: int
    node_type_counts: dict[str, int]
    edge_type_counts: dict[str, int]
    issues: tuple[GraphQualityIssue, ...]

    @property
    def passed(self) -> bool:
        """Return True when no error-severity issues were found."""

        return self.error_count == 0


def _items_by_id(values: Any) -> dict[str, Any]:
    if isinstance(values, Mapping):
        return {str(key): value for key, value in values.items()}
    output: dict[str, Any] = {}
    for item in values or []:
        item_id = _get(item, "node_id") or _get(item, "edge_id") or _get(item, "id")
        if item_id:
            output[str(item_id)] = item
    return output


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _properties(obj: Any) -> Mapping[str, Any]:
    value = _get(obj, "properties", {})
    return value if isinstance(value, Mapping) else {}


def _confidence(obj: Any) -> float | None:
    value = _get(obj, "confidence", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _graph_parts(graph: KnowledgeGraph | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(graph, KnowledgeGraph):
        return dict(graph.nodes), dict(graph.edges)
    return _items_by_id(graph.get("nodes", {})), _items_by_id(graph.get("edges", {}))


def audit_graph_quality(
    graph: KnowledgeGraph | Mapping[str, Any],
    *,
    required_node_types: Sequence[str] = (),
    required_edge_types: Sequence[str] = (),
    weak_confidence_threshold: float = 0.5,
) -> GraphQualityReport:
    """Run deterministic graph QA gates over a validated graph or raw graph dict."""

    nodes, edges = _graph_parts(graph)
    node_type_counts = Counter(str(_get(node, "node_type", "unknown")) for node in nodes.values())
    edge_type_counts = Counter(str(_get(edge, "edge_type", "unknown")) for edge in edges.values())
    issues: list[GraphQualityIssue] = []

    incident_node_ids: set[str] = set()
    for edge_id, edge in edges.items():
        source_id = str(_get(edge, "source_node_id", ""))
        target_id = str(_get(edge, "target_node_id", ""))
        edge_type = str(_get(edge, "edge_type", "unknown"))
        confidence = _confidence(edge)
        if confidence is None or not 0.0 <= confidence <= 1.0:
            issues.append(
                GraphQualityIssue(
                    code="invalid_edge_confidence",
                    severity="error",
                    message="Edge confidence must be between 0 and 1.",
                    edge_id=edge_id,
                    edge_type=edge_type,
                    confidence=confidence,
                )
            )
        elif confidence < weak_confidence_threshold:
            issues.append(
                GraphQualityIssue(
                    code="weak_edge",
                    severity="warning",
                    message="Edge confidence is below the configured weak-link threshold.",
                    edge_id=edge_id,
                    edge_type=edge_type,
                    confidence=confidence,
                )
            )
        for endpoint, node_id in (("source", source_id), ("target", target_id)):
            if node_id:
                incident_node_ids.add(node_id)
            if node_id and node_id not in nodes:
                issues.append(
                    GraphQualityIssue(
                        code="dangling_edge_reference",
                        severity="error",
                        message=f"Edge {endpoint} node is not present in graph nodes.",
                        node_id=node_id,
                        edge_id=edge_id,
                        edge_type=edge_type,
                    )
                )

    for node_id, node in nodes.items():
        node_type = str(_get(node, "node_type", "unknown"))
        confidence = _confidence(node)
        if confidence is None or not 0.0 <= confidence <= 1.0:
            issues.append(
                GraphQualityIssue(
                    code="invalid_node_confidence",
                    severity="error",
                    message="Node confidence must be between 0 and 1.",
                    node_id=node_id,
                    node_type=node_type,
                    confidence=confidence,
                )
            )
        if _properties(node).get("placeholder") is True:
            issues.append(
                GraphQualityIssue(
                    code="unresolved_placeholder",
                    severity="warning",
                    message="Placeholder node should be resolved with source metadata.",
                    node_id=node_id,
                    node_type=node_type,
                    confidence=confidence,
                )
            )
        if edges and node_id not in incident_node_ids:
            issues.append(
                GraphQualityIssue(
                    code="orphan_node",
                    severity="warning",
                    message="Node has no incident graph edges.",
                    node_id=node_id,
                    node_type=node_type,
                    confidence=confidence,
                )
            )

    for node_type in required_node_types:
        if node_type_counts.get(node_type, 0) == 0:
            issues.append(
                GraphQualityIssue(
                    code="missing_required_node_type",
                    severity="error",
                    message=f"Required node type is absent: {node_type}.",
                    node_type=node_type,
                )
            )

    for edge_type in required_edge_types:
        if edge_type_counts.get(edge_type, 0) == 0:
            issues.append(
                GraphQualityIssue(
                    code="missing_required_edge_type",
                    severity="error",
                    message=f"Required edge type is absent: {edge_type}.",
                    edge_type=edge_type,
                )
            )

    issues.sort(
        key=lambda issue: (
            issue.severity,
            issue.code,
            issue.node_id or "",
            issue.edge_id or "",
        )
    )
    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = len(issues) - error_count
    return GraphQualityReport(
        node_count=len(nodes),
        edge_count=len(edges),
        issue_count=len(issues),
        error_count=error_count,
        warning_count=warning_count,
        node_type_counts=dict(sorted(node_type_counts.items())),
        edge_type_counts=dict(sorted(edge_type_counts.items())),
        issues=tuple(issues),
    )
