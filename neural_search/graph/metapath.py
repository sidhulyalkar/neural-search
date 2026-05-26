"""Metapath-based graph reasoning and path scoring.

Metapaths define semantically meaningful traversal patterns through heterogeneous
knowledge graphs. This module implements metapath template matching, path finding,
and confidence-weighted path scoring.

Mathematical foundation:
    A metapath P is a sequence of node types connected by edge types:
    P = T_1 --[R_1]--> T_2 --[R_2]--> ... --[R_{l-1}]--> T_l

    PathSim similarity between nodes v_i and v_j along metapath P:
    PathSim(v_i, v_j | P) = 2 * |paths_P(v_i, v_j)| / (|paths_P(v_i, v_i)| + |paths_P(v_j, v_j)|)

    Confidence-weighted path score:
    score(path) = product_{edge in path} confidence(edge)

Reference: Sun et al., "PathSim: Meta Path-based Top-K Similarity Search in
Heterogeneous Information Networks" (VLDB 2011)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from neural_search.graph.schema import (
    KnowledgeGraph,
    KnowledgeGraphEdge,
)


class QueryIntent(StrEnum):
    """Query intents that map to specific metapath templates."""

    DATASET_FOR_ANALYSIS = "dataset_for_analysis"
    SIMILAR_DATASET = "similar_dataset"
    PAPER_TO_DATASET = "paper_to_dataset"
    DATASET_TO_PAPER = "dataset_to_paper"
    EXPERIMENTAL_DESIGN = "experimental_design"
    ANALYSIS_REQUIREMENTS = "analysis_requirements"
    CROSS_SPECIES = "cross_species"
    TASK_RELATED = "task_related"


@dataclass
class MetapathStep:
    """A single step in a metapath template."""

    source_type: str
    edge_type: str
    target_type: str
    direction: str = "forward"  # "forward" or "backward"
    required: bool = True  # If False, step can be skipped


@dataclass
class MetapathTemplate:
    """A reusable metapath template for graph traversal."""

    name: str
    description: str
    steps: list[MetapathStep]
    applicable_intents: list[QueryIntent]
    weight: float = 1.0  # Template importance weight


@dataclass
class PathInstance:
    """A concrete path through the graph matching a template."""

    template_name: str
    nodes: list[str]  # Node IDs in traversal order
    edges: list[str]  # Edge IDs used
    confidence: float  # Product of edge confidences
    evidence: list[str]  # Evidence texts from edges
    length: int


@dataclass
class MetapathMatch:
    """Result of matching metapaths from a source node."""

    source_node_id: str
    target_node_id: str
    template_name: str
    paths: list[PathInstance]
    best_confidence: float
    path_count: int
    explanation: str


@dataclass
class MetapathScoreResult:
    """Complete metapath scoring result for a query."""

    query_node_id: str
    matches: list[MetapathMatch]
    total_score: float
    template_contributions: dict[str, float]
    explanations: list[str]


# Predefined metapath templates for scientific dataset search
ANALYSIS_METAPATHS: list[MetapathTemplate] = [
    MetapathTemplate(
        name="dataset_event_analysis",
        description="Dataset -> BehavioralEvent -> Analysis (via requirements)",
        steps=[
            MetapathStep("dataset", "dataset_has_behavioral_event", "behavioral_event"),
            MetapathStep("behavioral_event", "analysis_requires_behavioral_event", "analysis_affordance", direction="backward"),
        ],
        applicable_intents=[QueryIntent.DATASET_FOR_ANALYSIS, QueryIntent.ANALYSIS_REQUIREMENTS],
        weight=1.2,
    ),
    MetapathTemplate(
        name="dataset_modality_analysis",
        description="Dataset -> Modality -> Analysis (via requirements)",
        steps=[
            MetapathStep("dataset", "dataset_has_modality", "modality"),
            MetapathStep("modality", "analysis_requires_modality", "analysis_affordance", direction="backward"),
        ],
        applicable_intents=[QueryIntent.DATASET_FOR_ANALYSIS, QueryIntent.ANALYSIS_REQUIREMENTS],
        weight=1.0,
    ),
    MetapathTemplate(
        name="dataset_task_analysis",
        description="Dataset -> Task -> Analysis (via task structure)",
        steps=[
            MetapathStep("dataset", "dataset_has_task", "task"),
            MetapathStep("task", "analysis_requires_task_structure", "analysis_affordance", direction="backward"),
        ],
        applicable_intents=[QueryIntent.DATASET_FOR_ANALYSIS, QueryIntent.TASK_RELATED],
        weight=1.1,
    ),
]

SIMILARITY_METAPATHS: list[MetapathTemplate] = [
    MetapathTemplate(
        name="task_similarity",
        description="Dataset -> Task <- Dataset (shared task)",
        steps=[
            MetapathStep("dataset", "dataset_has_task", "task"),
            MetapathStep("task", "dataset_has_task", "dataset", direction="backward"),
        ],
        applicable_intents=[QueryIntent.SIMILAR_DATASET],
        weight=1.0,
    ),
    MetapathTemplate(
        name="modality_similarity",
        description="Dataset -> Modality <- Dataset (shared modality)",
        steps=[
            MetapathStep("dataset", "dataset_has_modality", "modality"),
            MetapathStep("modality", "dataset_has_modality", "dataset", direction="backward"),
        ],
        applicable_intents=[QueryIntent.SIMILAR_DATASET],
        weight=0.8,
    ),
    MetapathTemplate(
        name="species_similarity",
        description="Dataset -> Species <- Dataset (shared species)",
        steps=[
            MetapathStep("dataset", "dataset_has_species", "species"),
            MetapathStep("species", "dataset_has_species", "dataset", direction="backward"),
        ],
        applicable_intents=[QueryIntent.SIMILAR_DATASET, QueryIntent.CROSS_SPECIES],
        weight=0.9,
    ),
    MetapathTemplate(
        name="region_similarity",
        description="Dataset -> BrainRegion <- Dataset (shared region)",
        steps=[
            MetapathStep("dataset", "dataset_records_region", "brain_region"),
            MetapathStep("brain_region", "dataset_records_region", "dataset", direction="backward"),
        ],
        applicable_intents=[QueryIntent.SIMILAR_DATASET],
        weight=0.7,
    ),
]

PAPER_METAPATHS: list[MetapathTemplate] = [
    MetapathTemplate(
        name="paper_uses_dataset",
        description="Paper -> Dataset (direct link)",
        steps=[
            MetapathStep("paper", "paper_uses_dataset", "dataset"),
        ],
        applicable_intents=[QueryIntent.PAPER_TO_DATASET],
        weight=1.5,
    ),
    MetapathTemplate(
        name="paper_mentions_dataset",
        description="Paper -> Dataset (mention)",
        steps=[
            MetapathStep("paper", "paper_mentions_dataset", "dataset"),
        ],
        applicable_intents=[QueryIntent.PAPER_TO_DATASET],
        weight=1.0,
    ),
    MetapathTemplate(
        name="dataset_to_paper",
        description="Dataset <- Paper (reverse link)",
        steps=[
            MetapathStep("dataset", "paper_uses_dataset", "paper", direction="backward"),
        ],
        applicable_intents=[QueryIntent.DATASET_TO_PAPER],
        weight=1.5,
    ),
    MetapathTemplate(
        name="paper_task_dataset",
        description="Paper -> Task <- Dataset (paper studies task used by dataset)",
        steps=[
            MetapathStep("paper", "paper_studies_task", "task"),
            MetapathStep("task", "dataset_has_task", "dataset", direction="backward"),
        ],
        applicable_intents=[QueryIntent.PAPER_TO_DATASET],
        weight=0.8,
    ),
]

EXPERIMENTAL_DESIGN_METAPATHS: list[MetapathTemplate] = [
    MetapathTemplate(
        name="design_requires_dataset",
        description="ExperimentalDesign -> Task/Modality/Behavior -> Dataset",
        steps=[
            MetapathStep("experimental_design", "experimental_design_requires_task", "task"),
            MetapathStep("task", "dataset_has_task", "dataset", direction="backward"),
        ],
        applicable_intents=[QueryIntent.EXPERIMENTAL_DESIGN],
        weight=1.0,
    ),
    MetapathTemplate(
        name="design_can_use_dataset",
        description="ExperimentalDesign -> Dataset (direct compatibility)",
        steps=[
            MetapathStep("experimental_design", "experimental_design_can_use_dataset", "dataset"),
        ],
        applicable_intents=[QueryIntent.EXPERIMENTAL_DESIGN],
        weight=1.3,
    ),
]

ALL_TEMPLATES: list[MetapathTemplate] = (
    ANALYSIS_METAPATHS + SIMILARITY_METAPATHS + PAPER_METAPATHS + EXPERIMENTAL_DESIGN_METAPATHS
)


class MetapathScorer:
    """Score graph paths using metapath templates.

    Provides path finding, confidence propagation, and explanation generation
    for metapath-based graph reasoning.

    Example:
        >>> scorer = MetapathScorer(graph)
        >>> result = scorer.score_paths("node:dataset:dandi:000001", QueryIntent.SIMILAR_DATASET)
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        templates: list[MetapathTemplate] | None = None,
        max_paths_per_template: int = 10,
        min_confidence: float = 0.1,
    ):
        """Initialize the metapath scorer.

        Args:
            graph: The knowledge graph to traverse.
            templates: Custom templates (default: use built-in templates).
            max_paths_per_template: Maximum paths to find per template.
            min_confidence: Minimum confidence threshold for paths.
        """
        self.graph = graph
        self.templates = templates or ALL_TEMPLATES
        self.max_paths_per_template = max_paths_per_template
        self.min_confidence = min_confidence

        # Build adjacency indexes for efficient traversal
        self._forward_index: dict[str, dict[str, list[KnowledgeGraphEdge]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._backward_index: dict[str, dict[str, list[KnowledgeGraphEdge]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build adjacency indexes for graph traversal."""
        for edge in self.graph.edges.values():
            # Forward: source -> edge_type -> edges
            self._forward_index[edge.source_node_id][edge.edge_type].append(edge)
            # Backward: target -> edge_type -> edges
            self._backward_index[edge.target_node_id][edge.edge_type].append(edge)

    def _get_node_type(self, node_id: str) -> str | None:
        """Extract node type from node ID or graph."""
        if node_id in self.graph.nodes:
            return self.graph.nodes[node_id].node_type
        # Try to parse from ID format: node:type:...
        parts = node_id.split(":")
        if len(parts) >= 2 and parts[0] == "node":
            return parts[1]
        return None

    def _follow_step(
        self,
        current_node_id: str,
        step: MetapathStep,
    ) -> list[tuple[str, KnowledgeGraphEdge]]:
        """Follow a single metapath step from a node.

        Returns list of (target_node_id, edge) tuples.
        """
        results: list[tuple[str, KnowledgeGraphEdge]] = []

        if step.direction == "forward":
            edges = self._forward_index[current_node_id].get(step.edge_type, [])
            for edge in edges:
                target_type = self._get_node_type(edge.target_node_id)
                if target_type == step.target_type:
                    results.append((edge.target_node_id, edge))
        else:  # backward
            edges = self._backward_index[current_node_id].get(step.edge_type, [])
            for edge in edges:
                source_type = self._get_node_type(edge.source_node_id)
                if source_type == step.target_type:
                    results.append((edge.source_node_id, edge))

        return results

    def _find_paths(
        self,
        start_node_id: str,
        template: MetapathTemplate,
    ) -> list[PathInstance]:
        """Find all paths from a start node matching a template."""
        paths: list[PathInstance] = []

        # Check start node type matches template
        start_type = self._get_node_type(start_node_id)
        if not template.steps:
            return paths
        if start_type != template.steps[0].source_type:
            return paths

        # BFS-style path finding with confidence tracking
        # State: (current_node, step_index, node_path, edge_path, confidence)
        queue: list[tuple[str, int, list[str], list[str], float, list[str]]] = [
            (start_node_id, 0, [start_node_id], [], 1.0, [])
        ]

        while queue and len(paths) < self.max_paths_per_template:
            current, step_idx, node_path, edge_path, conf, evidence = queue.pop(0)

            if step_idx >= len(template.steps):
                # Completed the template
                if conf >= self.min_confidence:
                    paths.append(PathInstance(
                        template_name=template.name,
                        nodes=node_path,
                        edges=edge_path,
                        confidence=conf,
                        evidence=evidence,
                        length=len(edge_path),
                    ))
                continue

            step = template.steps[step_idx]
            next_nodes = self._follow_step(current, step)

            for next_node_id, edge in next_nodes:
                # Avoid cycles
                if next_node_id in node_path:
                    continue

                new_conf = conf * edge.confidence
                if new_conf < self.min_confidence:
                    continue

                new_evidence = evidence.copy()
                for ev in edge.evidence:
                    if ev.evidence_text:
                        new_evidence.append(ev.evidence_text)

                queue.append((
                    next_node_id,
                    step_idx + 1,
                    node_path + [next_node_id],
                    edge_path + [edge.edge_id],
                    new_conf,
                    new_evidence,
                ))

        return paths

    def find_paths_to_target(
        self,
        start_node_id: str,
        target_node_id: str,
        template: MetapathTemplate,
    ) -> list[PathInstance]:
        """Find paths from start to a specific target node."""
        all_paths = self._find_paths(start_node_id, template)
        return [p for p in all_paths if p.nodes[-1] == target_node_id]

    def score_paths(
        self,
        source_node_id: str,
        intent: QueryIntent,
        target_type: str | None = None,
    ) -> MetapathScoreResult:
        """Score all metapaths from a source node for a given intent.

        Args:
            source_node_id: Starting node ID.
            intent: Query intent to filter templates.
            target_type: Optional target node type filter.

        Returns:
            MetapathScoreResult with scored matches.
        """
        applicable_templates = [
            t for t in self.templates
            if intent in t.applicable_intents
        ]

        matches: list[MetapathMatch] = []
        template_contributions: dict[str, float] = {}
        total_score = 0.0
        explanations: list[str] = []

        # Group paths by target node
        target_paths: dict[str, dict[str, list[PathInstance]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for template in applicable_templates:
            paths = self._find_paths(source_node_id, template)

            for path in paths:
                target_node_id = path.nodes[-1]

                # Filter by target type if specified
                if target_type:
                    actual_type = self._get_node_type(target_node_id)
                    if actual_type != target_type:
                        continue

                target_paths[target_node_id][template.name].append(path)

        # Build matches from grouped paths
        for target_node_id, template_paths in target_paths.items():
            for template_name, paths in template_paths.items():
                if not paths:
                    continue

                template = next(t for t in applicable_templates if t.name == template_name)
                best_conf = max(p.confidence for p in paths)
                weighted_score = best_conf * template.weight

                match = MetapathMatch(
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    template_name=template_name,
                    paths=paths,
                    best_confidence=best_conf,
                    path_count=len(paths),
                    explanation=self._generate_explanation(template, paths[0]),
                )
                matches.append(match)

                template_contributions[template_name] = (
                    template_contributions.get(template_name, 0.0) + weighted_score
                )
                total_score += weighted_score

                if match.explanation:
                    explanations.append(match.explanation)

        # Sort matches by confidence
        matches.sort(key=lambda m: (-m.best_confidence, m.target_node_id))

        return MetapathScoreResult(
            query_node_id=source_node_id,
            matches=matches,
            total_score=total_score,
            template_contributions=template_contributions,
            explanations=explanations[:10],  # Limit explanations
        )

    def _generate_explanation(
        self,
        template: MetapathTemplate,
        path: PathInstance,
    ) -> str:
        """Generate a human-readable explanation for a path."""
        if not path.nodes:
            return ""

        # Get node labels
        labels: list[str] = []
        for node_id in path.nodes:
            if node_id in self.graph.nodes:
                labels.append(self.graph.nodes[node_id].label)
            else:
                # Extract from ID
                parts = node_id.split(":")
                labels.append(parts[-1] if parts else node_id)

        if len(labels) < 2:
            return ""

        # Build explanation
        parts = [f"'{labels[0]}'"]
        for i, step in enumerate(template.steps):
            if i + 1 < len(labels):
                relation = step.edge_type.replace("_", " ")
                parts.append(f"--[{relation}]-->")
                parts.append(f"'{labels[i + 1]}'")

        confidence_str = f"(confidence: {path.confidence:.2f})"
        return " ".join(parts) + f" {confidence_str}"

    def compute_pathsim(
        self,
        node_i: str,
        node_j: str,
        template: MetapathTemplate,
    ) -> float:
        """Compute PathSim similarity between two nodes along a metapath.

        PathSim(i, j | P) = 2 * |paths(i,j)| / (|paths(i,i)| + |paths(j,j)|)
        """
        paths_ij = self.find_paths_to_target(node_i, node_j, template)
        paths_ii = self.find_paths_to_target(node_i, node_i, template)
        paths_jj = self.find_paths_to_target(node_j, node_j, template)

        numerator = 2 * len(paths_ij)
        denominator = len(paths_ii) + len(paths_jj)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_templates_for_intent(self, intent: QueryIntent) -> list[MetapathTemplate]:
        """Get all templates applicable to an intent."""
        return [t for t in self.templates if intent in t.applicable_intents]


def score_metapaths(
    graph: KnowledgeGraph,
    source_node_id: str,
    intent: QueryIntent,
    templates: list[MetapathTemplate] | None = None,
    target_type: str | None = None,
) -> MetapathScoreResult:
    """Convenience function to score metapaths.

    Args:
        graph: Knowledge graph to traverse.
        source_node_id: Starting node ID.
        intent: Query intent.
        templates: Optional custom templates.
        target_type: Optional target node type filter.

    Returns:
        MetapathScoreResult with all matches.
    """
    scorer = MetapathScorer(graph, templates=templates)
    return scorer.score_paths(source_node_id, intent, target_type)


def explain_path(
    graph: KnowledgeGraph,
    path: PathInstance,
) -> dict[str, Any]:
    """Generate detailed explanation for a path instance.

    Args:
        graph: Knowledge graph for node labels.
        path: Path instance to explain.

    Returns:
        Dictionary with detailed path explanation.
    """
    nodes_info: list[dict[str, Any]] = []
    edges_info: list[dict[str, Any]] = []

    for node_id in path.nodes:
        if node_id in graph.nodes:
            node = graph.nodes[node_id]
            nodes_info.append({
                "id": node_id,
                "type": node.node_type,
                "label": node.label,
                "confidence": node.confidence,
            })
        else:
            nodes_info.append({"id": node_id, "type": "unknown", "label": node_id})

    for edge_id in path.edges:
        if edge_id in graph.edges:
            edge = graph.edges[edge_id]
            edges_info.append({
                "id": edge_id,
                "type": edge.edge_type,
                "confidence": edge.confidence,
                "evidence": [e.evidence_text for e in edge.evidence if e.evidence_text],
            })
        else:
            edges_info.append({"id": edge_id, "type": "unknown"})

    return {
        "template": path.template_name,
        "length": path.length,
        "confidence": path.confidence,
        "nodes": nodes_info,
        "edges": edges_info,
        "evidence": path.evidence,
    }
