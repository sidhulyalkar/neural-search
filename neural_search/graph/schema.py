"""File-backed scientific knowledge graph schema."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_NODE_TYPES = {
    "dataset",
    "paper",
    "task",
    "modality",
    "recording_scale",
    "brain_region",
    "species",
    "taxon_group",
    "organism_model",
    "behavioral_event",
    "analysis_affordance",
    "required_signal",
    "data_standard",
    "file_format",
    "modeling_method",
    "disease_state",
    "clinical_condition",
    "stimulus_type",
    "subject_state",
    "recording_context",
    "finding",
    "experimental_design",
    "analysis_method",
    "institution",
    "author",
    "venue",
    # Field-state memory graph node types
    "source_archive",
    "concept",
    "pipeline",
    "file_artifact",
    "raw_data_signal",
    "processed_data_signal",
    "query",
    "query_intent",
    "retrieval_run",
    "neuro_judge_evidence_packet",
    "neuro_judge_judgment",
    "feedback_signal",
    "curation_issue",
    "snapshot_manifest",
}

SUPPORTED_EDGE_TYPES = {
    "dataset_has_task",
    "dataset_has_modality",
    "dataset_has_recording_scale",
    "dataset_records_region",
    "dataset_has_species",
    "dataset_has_behavioral_event",
    "dataset_supports_analysis",
    "dataset_uses_standard",
    "dataset_has_file_format",
    "dataset_has_subject_state",
    "dataset_has_stimulus_type",
    "paper_mentions_dataset",
    "paper_uses_dataset",
    "paper_studies_task",
    "paper_uses_modality",
    "paper_reports_finding",
    "paper_published_in",
    "paper_mentions_region",
    "paper_uses_method",
    "paper_has_author",
    "paper_from_institution",
    "species_in_taxon_group",
    "species_has_model_role",
    "species_has_animal_type",
    "task_has_behavioral_event",
    "task_uses_stimulus_type",
    "analysis_requires_modality",
    "analysis_requires_behavioral_event",
    "analysis_requires_task_structure",
    "analysis_applicable_to_dataset",
    "method_supports_analysis",
    "region_related_to_task",
    "dataset_similar_to_dataset",
    "paper_related_to_paper",
    "finding_supported_by_dataset",
    "finding_reported_by_paper",
    "finding_involves_region",
    "finding_involves_task",
    "finding_involves_modality",
    "finding_involves_species",
    "experimental_design_requires_task",
    "experimental_design_requires_modality",
    "experimental_design_requires_behavior",
    "experimental_design_can_use_dataset",
    # Field-state memory graph edge types
    "dataset_from_source",
    "dataset_has_file_artifact",
    "dataset_has_raw_signal",
    "dataset_has_processed_signal",
    "dataset_lacks_required_evidence",
    "dataset_contraindicated_for",
    "dataset_linked_to_paper",
    "paper_supports_method",
    "method_requires_modality",
    "method_requires_raw_data",
    "method_requires_region",
    "concept_related_to_concept",
    "concept_requires_affordance",
    "query_requires_modality",
    "query_requires_recording_scale",
    "query_requires_species",
    "query_requires_region",
    "query_requires_task",
    "query_requires_affordance",
    "query_has_hard_negative",
    "retrieval_returned_dataset",
    "judgment_labels_query_dataset",
    "feedback_marks_result",
    "snapshot_contains_node",
    "snapshot_contains_edge",
}

TOKEN_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _require_text(value: str, field_name: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{field_name} cannot be empty")
    return cleaned


def _normalize_identifier(value: str) -> str:
    cleaned = _require_text(value, "identifier").strip().lower()
    cleaned = cleaned.replace("-", "_").replace(" ", "_")
    cleaned = TOKEN_RE.sub("_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("identifier cannot be empty")
    return cleaned


def _safe_id_part(value: str) -> str:
    cleaned = _require_text(value, "ID part").strip()
    cleaned = TOKEN_RE.sub("_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("ID part cannot be empty")
    return cleaned


def normalize_node_type(value: str) -> str:
    """Normalize a node type to a stable snake_case identifier."""

    return _normalize_identifier(value)


def normalize_edge_type(value: str) -> str:
    """Normalize an edge type to a stable snake_case identifier."""

    return _normalize_identifier(value)


def make_node_id(node_type: str, *parts: str) -> str:
    """Make a stable graph node ID such as ``node:dataset:dandi:000026``."""

    if not parts:
        raise ValueError("node ID requires at least one ID part")
    normalized_type = normalize_node_type(node_type)
    safe_parts = [_safe_id_part(part) for part in parts]
    return ":".join(["node", normalized_type, *safe_parts])


def _split_node_id(node_id: str) -> tuple[str, list[str]]:
    parts = _require_text(node_id, "node_id").split(":")
    if len(parts) < 3 or parts[0] != "node":
        raise ValueError(f"invalid node ID: {node_id}")
    return parts[1], parts[2:]


def _compact_edge_relation(source_type: str, edge_type: str) -> str:
    normalized = normalize_edge_type(edge_type)
    prefix = f"{source_type}_"
    if normalized.startswith(prefix):
        return normalized.removeprefix(prefix)
    return normalized


def make_edge_id(source_node_id: str, edge_type: str, target_node_id: str) -> str:
    """Make a stable graph edge ID from source, relation, and target IDs."""

    source_type, source_parts = _split_node_id(source_node_id)
    _, target_parts = _split_node_id(target_node_id)
    relation = _compact_edge_relation(source_type, edge_type)
    return ":".join(["edge", source_type, *source_parts, relation, *target_parts])


class GraphEvidence(BaseModel):
    """Evidence supporting a graph node or edge."""

    evidence_id: str
    source_type: str
    source_id: str
    source_field: str | None = None
    evidence_text: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    extractor_name: str
    extractor_version: str

    @field_validator("evidence_id", "source_id", "extractor_name", "extractor_version")
    @classmethod
    def required_text(cls, value: str) -> str:
        return _require_text(value, "evidence field")

    @field_validator("source_type")
    @classmethod
    def normalized_source_type(cls, value: str) -> str:
        return _normalize_identifier(value)


class KnowledgeGraphNode(BaseModel):
    """A scientific concept or record in the knowledge graph."""

    node_id: str
    node_type: str
    label: str
    aliases: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    evidence: list[GraphEvidence] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str | None = None

    @field_validator("node_id")
    @classmethod
    def required_node_id(cls, value: str) -> str:
        return _require_text(value, "node_id")

    @field_validator("node_type")
    @classmethod
    def normalized_node_type(cls, value: str) -> str:
        return normalize_node_type(value)

    @field_validator("label")
    @classmethod
    def required_label(cls, value: str) -> str:
        return _require_text(value, "label")


class KnowledgeGraphEdge(BaseModel):
    """A provenance-backed relationship between two graph nodes."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    directed: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: list[GraphEvidence] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str | None = None

    @field_validator("edge_id", "source_node_id", "target_node_id")
    @classmethod
    def required_ids(cls, value: str) -> str:
        return _require_text(value, "edge identifier")

    @field_validator("edge_type")
    @classmethod
    def normalized_edge_type(cls, value: str) -> str:
        return normalize_edge_type(value)


class KnowledgeGraph(BaseModel):
    """A lightweight, file-backed knowledge graph container."""

    nodes: dict[str, KnowledgeGraphNode] = Field(default_factory=dict)
    edges: dict[str, KnowledgeGraphEdge] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_complete_graph(self) -> KnowledgeGraph:
        validate_graph(self)
        return self


def validate_graph(graph: KnowledgeGraph) -> KnowledgeGraph:
    """Validate graph identity maps and edge references."""

    node_ids: set[str] = set()
    for key, node in graph.nodes.items():
        if key != node.node_id:
            raise ValueError(f"node map key does not match node_id: {key}")
        if node.node_id in node_ids:
            raise ValueError(f"duplicate node ID: {node.node_id}")
        node_ids.add(node.node_id)

    edge_ids: set[str] = set()
    for key, edge in graph.edges.items():
        if key != edge.edge_id:
            raise ValueError(f"edge map key does not match edge_id: {key}")
        if edge.edge_id in edge_ids:
            raise ValueError(f"duplicate edge ID: {edge.edge_id}")
        edge_ids.add(edge.edge_id)
        if edge.source_node_id not in graph.nodes:
            raise ValueError(f"edge source does not resolve: {edge.source_node_id}")
        if edge.target_node_id not in graph.nodes:
            raise ValueError(f"edge target does not resolve: {edge.target_node_id}")

    return graph


def graph_to_dict(graph: KnowledgeGraph) -> dict[str, Any]:
    """Convert a graph to a JSON-serializable dictionary."""

    validate_graph(graph)
    return graph.model_dump(mode="json")


def graph_from_dict(payload: Mapping[str, Any]) -> KnowledgeGraph:
    """Validate a graph dictionary as a ``KnowledgeGraph``."""

    return KnowledgeGraph.model_validate(dict(payload))


def write_graph_json(graph: KnowledgeGraph, path: str | Path) -> Path:
    """Write a graph as a single JSON document."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(graph_to_dict(graph), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output


def read_graph_json(path: str | Path) -> KnowledgeGraph:
    """Read a graph from a single JSON document."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return graph_from_dict(payload)


def write_graph_jsonl(graph: KnowledgeGraph, path: str | Path) -> Path:
    """Write graph metadata, nodes, and edges as JSONL records."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_graph(graph)
    with output.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "metadata", "metadata": graph.metadata}))
        handle.write("\n")
        for node in graph.nodes.values():
            handle.write(
                json.dumps(
                    {"record_type": "node", "node": node.model_dump(mode="json")},
                    sort_keys=True,
                )
            )
            handle.write("\n")
        for edge in graph.edges.values():
            handle.write(
                json.dumps(
                    {"record_type": "edge", "edge": edge.model_dump(mode="json")},
                    sort_keys=True,
                )
            )
            handle.write("\n")
    return output


def read_graph_jsonl(path: str | Path) -> KnowledgeGraph:
    """Read a graph from JSONL records and reject duplicate IDs."""

    metadata: dict[str, Any] = {}
    nodes: dict[str, KnowledgeGraphNode] = {}
    edges: dict[str, KnowledgeGraphEdge] = {}

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            record_type = payload.get("record_type")
            if record_type == "metadata":
                metadata.update(payload.get("metadata", {}))
                continue
            if record_type == "node":
                node = KnowledgeGraphNode.model_validate(payload.get("node", {}))
                if node.node_id in nodes:
                    raise ValueError(f"duplicate node ID in JSONL: {node.node_id}")
                nodes[node.node_id] = node
                continue
            if record_type == "edge":
                edge = KnowledgeGraphEdge.model_validate(payload.get("edge", {}))
                if edge.edge_id in edges:
                    raise ValueError(f"duplicate edge ID in JSONL: {edge.edge_id}")
                edges[edge.edge_id] = edge
                continue
            raise ValueError(f"unknown graph JSONL record on line {line_number}")

    return KnowledgeGraph(nodes=nodes, edges=edges, metadata=metadata)
