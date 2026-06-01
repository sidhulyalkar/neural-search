"""Quality gates for provenance-aware Neural Search graphs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from neural_search.graph.schema import KnowledgeGraph

Severity = Literal["error", "warning"]

COVERAGE_CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config" / "graph_coverage.yaml"


@dataclass(frozen=True)
class CoverageRequirement:
    """A node or edge type coverage requirement."""

    type_name: str
    min_count: int
    severity: Severity
    description: str = ""


@dataclass(frozen=True)
class CoverageThreshold:
    """A coverage percentage threshold for dataset-to-concept connectivity."""

    metric_name: str
    threshold: float
    actual: float
    passed: bool


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
    actual_count: int | None = None
    required_count: int | None = None


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
    coverage_thresholds: tuple[CoverageThreshold, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        """Return True when no error-severity issues were found."""

        return self.error_count == 0

    @property
    def coverage_summary(self) -> dict[str, Any]:
        """Return a summary of coverage metrics."""

        return {
            "node_types": dict(self.node_type_counts),
            "edge_types": dict(self.edge_type_counts),
            "coverage_thresholds": [
                {
                    "metric": ct.metric_name,
                    "threshold": ct.threshold,
                    "actual": ct.actual,
                    "passed": ct.passed,
                }
                for ct in self.coverage_thresholds
            ],
        }


_COVERAGE_CONFIG_CACHE: dict[str, Any] | None = None


def _load_coverage_config() -> dict[str, Any]:
    """Load the graph coverage configuration."""

    global _COVERAGE_CONFIG_CACHE
    if _COVERAGE_CONFIG_CACHE is None:
        if COVERAGE_CONFIG_PATH.exists():
            with open(COVERAGE_CONFIG_PATH, encoding="utf-8") as f:
                _COVERAGE_CONFIG_CACHE = yaml.safe_load(f) or {}
        else:
            _COVERAGE_CONFIG_CACHE = {}
    return _COVERAGE_CONFIG_CACHE


def get_required_node_types(profile: str | None = None) -> list[CoverageRequirement]:
    """Get required node types from config with minimum counts."""

    config = _load_coverage_config()
    requirements = []
    for type_name, spec in config.get("required_node_types", {}).items():
        if isinstance(spec, dict):
            requirements.append(
                CoverageRequirement(
                    type_name=type_name,
                    min_count=spec.get("min_count", 0),
                    severity=spec.get("severity", "warning"),
                    description=spec.get("description", ""),
                )
            )
    return requirements


def get_required_edge_types(profile: str | None = None) -> list[CoverageRequirement]:
    """Get required edge types from config with minimum counts."""

    config = _load_coverage_config()
    requirements = []
    for type_name, spec in config.get("required_edge_types", {}).items():
        if isinstance(spec, dict):
            requirements.append(
                CoverageRequirement(
                    type_name=type_name,
                    min_count=spec.get("min_count", 0),
                    severity=spec.get("severity", "warning"),
                    description=spec.get("description", ""),
                )
            )
    return requirements


def get_coverage_thresholds() -> dict[str, float]:
    """Get coverage threshold percentages from config."""

    config = _load_coverage_config()
    return config.get("coverage_thresholds", {})


def get_quality_gate_profile(profile: str = "ci") -> dict[str, Any]:
    """Get quality gate settings for a profile."""

    config = _load_coverage_config()
    profiles = config.get("profiles", {})
    return profiles.get(profile, profiles.get("ci", {}))


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
    use_coverage_config: bool = False,
    profile: str = "ci",
) -> GraphQualityReport:
    """Run deterministic graph QA gates over a validated graph or raw graph dict.

    Args:
        graph: Knowledge graph to validate
        required_node_types: Legacy list of required node type names
        required_edge_types: Legacy list of required edge type names
        weak_confidence_threshold: Threshold below which edges are flagged
        use_coverage_config: If True, load requirements from config file
        profile: Config profile to use (ci, local, release)
    """

    nodes, edges = _graph_parts(graph)
    node_type_counts = Counter(str(_get(node, "node_type", "unknown")) for node in nodes.values())
    edge_type_counts = Counter(str(_get(edge, "edge_type", "unknown")) for edge in edges.values())
    issues: list[GraphQualityIssue] = []
    coverage_results: list[CoverageThreshold] = []

    # Load coverage requirements from config if requested
    node_requirements: list[CoverageRequirement] = []
    edge_requirements: list[CoverageRequirement] = []
    coverage_thresholds: dict[str, float] = {}

    if use_coverage_config:
        node_requirements = get_required_node_types(profile)
        edge_requirements = get_required_edge_types(profile)
        coverage_thresholds = get_coverage_thresholds()
        profile_config = get_quality_gate_profile(profile)
        if "weak_confidence_threshold" in profile_config:
            weak_confidence_threshold = profile_config["weak_confidence_threshold"]

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

    # Check legacy required node types (simple presence check)
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

    # Check config-based node requirements (with min count thresholds)
    for req in node_requirements:
        actual = node_type_counts.get(req.type_name, 0)
        if actual < req.min_count:
            issues.append(
                GraphQualityIssue(
                    code="below_min_node_count",
                    severity=req.severity,
                    message=f"Node type '{req.type_name}' has {actual} nodes, requires {req.min_count}. {req.description}",
                    node_type=req.type_name,
                    actual_count=actual,
                    required_count=req.min_count,
                )
            )

    # Check legacy required edge types
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

    # Check config-based edge requirements (with min count thresholds)
    for req in edge_requirements:
        actual = edge_type_counts.get(req.type_name, 0)
        if actual < req.min_count:
            issues.append(
                GraphQualityIssue(
                    code="below_min_edge_count",
                    severity=req.severity,
                    message=f"Edge type '{req.type_name}' has {actual} edges, requires {req.min_count}. {req.description}",
                    edge_type=req.type_name,
                    actual_count=actual,
                    required_count=req.min_count,
                )
            )

    # Check coverage thresholds (percentage of datasets with certain edge types)
    dataset_count = node_type_counts.get("dataset", 0)
    if dataset_count > 0 and coverage_thresholds:
        # Build map of dataset nodes that have specific edge types
        dataset_edge_coverage: dict[str, set[str]] = {}
        for edge in edges.values():
            source_id = str(_get(edge, "source_node_id", ""))
            edge_type = str(_get(edge, "edge_type", "unknown"))
            if source_id.startswith("node:dataset:"):
                if edge_type not in dataset_edge_coverage:
                    dataset_edge_coverage[edge_type] = set()
                dataset_edge_coverage[edge_type].add(source_id)

        # Check coverage thresholds
        threshold_map = {
            "dataset_task_coverage": "dataset_has_task",
            "dataset_modality_coverage": "dataset_has_modality",
            "dataset_species_coverage": "dataset_has_species",
            "dataset_region_coverage": "dataset_records_region",
        }
        for metric_name, edge_type in threshold_map.items():
            if metric_name in coverage_thresholds:
                threshold = coverage_thresholds[metric_name]
                datasets_with_edge = len(dataset_edge_coverage.get(edge_type, set()))
                actual_coverage = datasets_with_edge / dataset_count
                passed = actual_coverage >= threshold
                coverage_results.append(
                    CoverageThreshold(
                        metric_name=metric_name,
                        threshold=threshold,
                        actual=round(actual_coverage, 3),
                        passed=passed,
                    )
                )
                if not passed:
                    issues.append(
                        GraphQualityIssue(
                            code="below_coverage_threshold",
                            severity="warning",
                            message=f"Coverage '{metric_name}' is {actual_coverage:.1%}, requires {threshold:.1%}.",
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
        coverage_thresholds=tuple(coverage_results),
    )


def validate_graph_coverage(
    graph: KnowledgeGraph | Mapping[str, Any],
    profile: str = "ci",
) -> GraphQualityReport:
    """Validate graph coverage using configuration-based gates.

    This is the recommended entry point for CI/CD pipelines.

    Args:
        graph: Knowledge graph to validate
        profile: Config profile (ci, local, release)

    Returns:
        GraphQualityReport with coverage validation results
    """

    return audit_graph_quality(
        graph,
        use_coverage_config=True,
        profile=profile,
    )
