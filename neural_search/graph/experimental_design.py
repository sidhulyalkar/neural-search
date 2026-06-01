"""Experimental design seed loading and dataset matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from neural_search.graph.query import find_nodes_by_type
from neural_search.graph.schema import KnowledgeGraph, normalize_node_type

DEFAULT_SEED_PATH = Path(__file__).resolve().parents[2] / "data" / "graph" / "experimental_design_seeds.yaml"


@dataclass
class ExperimentalDesign:
    """An experimental design template with requirements."""

    id: str
    name: str
    description: str
    requires: dict[str, list[str]]
    helpful: dict[str, list[str]]
    caveats: list[str]
    minimum_requirements: dict[str, Any]


@dataclass
class DesignMatch:
    """Result of matching a dataset against an experimental design."""

    design_id: str
    dataset_id: str
    score: float
    satisfied_requirements: list[str]
    missing_requirements: list[str]
    helpful_present: list[str]
    caveats_applicable: list[str]
    explanation: list[str] = field(default_factory=list)


def load_experimental_design_seeds(path: str | Path | None = None) -> list[ExperimentalDesign]:
    """Load experimental design seeds from YAML file."""
    seed_path = Path(path) if path else DEFAULT_SEED_PATH

    if not seed_path.exists():
        return []

    with seed_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    designs: list[ExperimentalDesign] = []
    for item in data.get("experimental_designs", []):
        designs.append(
            ExperimentalDesign(
                id=item.get("id", ""),
                name=item.get("name", ""),
                description=item.get("description", "").strip(),
                requires=_normalize_requirement_mapping(item.get("requires", {})),
                helpful=_normalize_requirement_mapping(item.get("helpful", {})),
                caveats=item.get("caveats", []),
                minimum_requirements=_normalize_minimum_requirements(
                    item.get("minimum_requirements", {})
                ),
            )
        )

    return designs


def get_experimental_design(
    design_id: str,
    path: str | Path | None = None,
) -> ExperimentalDesign | None:
    """Get a specific experimental design by ID."""
    designs = load_experimental_design_seeds(path)
    for design in designs:
        if design.id == design_id:
            return design
    return None


def _normalize_requirement_mapping(value: Any) -> dict[str, list[str]]:
    """Accept either mapping or list-of-singleton-mapping seed syntax."""

    if isinstance(value, dict):
        return {
            str(key): [str(item) for item in items]
            for key, items in value.items()
            if isinstance(items, list)
        }
    if isinstance(value, list):
        merged: dict[str, list[str]] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            for key, items in item.items():
                if isinstance(items, list):
                    merged.setdefault(str(key), []).extend(str(value) for value in items)
                elif items is not None:
                    merged.setdefault(str(key), []).append(str(items))
        return merged
    return {}


def _normalize_minimum_requirements(value: Any) -> dict[str, Any]:
    """Accept mapping or list-of-singleton-mapping minimum requirement syntax."""

    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        merged: dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict):
                merged.update(item)
        return merged
    return {}


def _get_dataset_labels(
    graph: KnowledgeGraph,
    dataset_id: str,
    label_type: str,
) -> set[str]:
    """Get label values connected to a dataset node."""
    labels: set[str] = set()
    edge_type_map = {
        "tasks": "dataset_has_task",
        "modalities": "dataset_has_modality",
        "brain_regions": "dataset_records_region",
        "species": "dataset_has_species",
        "behavioral_events": "dataset_has_behavioral_event",
        "analysis_affordances": "dataset_supports_analysis",
        "data_standards": "dataset_uses_standard",
        "stimulus_types": "dataset_has_stimulus_type",
        "subject_states": "dataset_has_subject_state",
        "disease_states": "dataset_has_disease_state",
    }

    edge_type = edge_type_map.get(label_type)
    if not edge_type:
        return labels

    for edge in graph.edges.values():
        if edge.source_node_id == dataset_id and edge.edge_type == edge_type:
            target = graph.nodes.get(edge.target_node_id)
            if target:
                # Extract the label from the node
                labels.add(normalize_node_type(target.label))
                labels.update(normalize_node_type(a) for a in target.aliases if a)

    return labels


def _check_requirements(
    graph: KnowledgeGraph,
    dataset_id: str,
    requirements: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Check which requirements are satisfied and which are missing."""
    satisfied: list[str] = []
    missing: list[str] = []

    for req_type, req_values in requirements.items():
        if not req_values:
            continue

        dataset_labels = _get_dataset_labels(graph, dataset_id, req_type)
        normalized_reqs = {normalize_node_type(v) for v in req_values}

        for req in normalized_reqs:
            if req in dataset_labels:
                satisfied.append(f"{req_type}:{req}")
            else:
                missing.append(f"{req_type}:{req}")

    return satisfied, missing


def _check_minimum_requirements(
    graph: KnowledgeGraph,
    dataset_id: str,
    minimum_requirements: dict[str, Any],
) -> tuple[list[str], list[str]]:
    dataset = graph.nodes.get(dataset_id)
    if dataset is None:
        return [], []
    flags = dataset.properties.get("usability_flags", {})
    satisfied: list[str] = []
    missing: list[str] = []
    for key, expected in minimum_requirements.items():
        actual = flags.get(key)
        marker = f"minimum:{key}"
        if actual == expected:
            satisfied.append(marker)
        else:
            missing.append(marker)
    return satisfied, missing


def find_datasets_for_experimental_design(
    graph: KnowledgeGraph,
    design_id: str,
    path: str | Path | None = None,
    min_score: float = 0.0,
) -> list[DesignMatch]:
    """Find datasets matching an experimental design's requirements."""
    design = get_experimental_design(design_id, path)
    if not design:
        return []

    matches: list[DesignMatch] = []
    datasets = find_nodes_by_type(graph, "dataset")

    for dataset in datasets:
        # Check required labels
        satisfied_reqs, missing_reqs = _check_requirements(
            graph,
            dataset.node_id,
            design.requires,
        )

        # Check helpful labels
        helpful_present, _ = _check_requirements(
            graph,
            dataset.node_id,
            design.helpful,
        )
        satisfied_minimum, missing_minimum = _check_minimum_requirements(
            graph,
            dataset.node_id,
            design.minimum_requirements,
        )
        satisfied_reqs.extend(satisfied_minimum)
        missing_reqs.extend(missing_minimum)

        # Calculate score
        total_required = (
            sum(len(v) for v in design.requires.values())
            + len(design.minimum_requirements)
        )
        if total_required > 0:
            score = len(satisfied_reqs) / total_required
        else:
            score = 1.0

        # Add bonus for helpful fields
        total_helpful = sum(len(v) for v in design.helpful.values())
        if total_helpful > 0:
            score += 0.2 * (len(helpful_present) / total_helpful)

        if score < min_score:
            continue

        # Generate explanation
        explanation: list[str] = []
        if satisfied_reqs:
            explanation.append(f"Satisfies: {', '.join(satisfied_reqs[:5])}")
        if missing_reqs:
            explanation.append(f"Missing: {', '.join(missing_reqs[:5])}")
        if helpful_present:
            explanation.append(f"Has helpful: {', '.join(helpful_present[:3])}")

        matches.append(
            DesignMatch(
                design_id=design_id,
                dataset_id=dataset.node_id,
                score=round(score, 3),
                satisfied_requirements=satisfied_reqs,
                missing_requirements=missing_reqs,
                helpful_present=helpful_present,
                caveats_applicable=design.caveats,
                explanation=explanation,
            )
        )

    return sorted(matches, key=lambda m: (-m.score, len(m.missing_requirements)))


def list_experimental_designs(path: str | Path | None = None) -> list[dict[str, Any]]:
    """List all available experimental designs with summaries."""
    designs = load_experimental_design_seeds(path)
    return [
        {
            "id": d.id,
            "name": d.name,
            "requires": {k: len(v) for k, v in d.requires.items() if v},
            "helpful": {k: len(v) for k, v in d.helpful.items() if v},
            "caveats_count": len(d.caveats),
        }
        for d in designs
    ]
