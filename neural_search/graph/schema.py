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
    "claim",
    "frequency_band",
    "temporal_pattern",
    "spatial_frame",
    "experimental_design",
    "analysis_method",
    "institution",
    "author",
    "venue",
    "atlas_structure",   # A raw Allen structure node
    "topic",             # A canonical research topic from topic_taxonomy.yaml
    # ── Multi-scale KG extensions ──────────────────────────────────────────
    "method",            # Analysis method (FFT, PAC, Granger causality, etc.)
    "math_concept",      # Mathematical concept / theorem (Cramér-Rao, Bayes theorem)
    "paradigm",          # Behavioral/cognitive experimental paradigm
    "oscillation",       # Named oscillatory phenomenon (theta, beta burst, ripple)
    "disorder",          # Clinical neuropsychiatric disorder
    "cell_type",         # Neuron type (PV interneuron, D1-MSN, Purkinje cell)
    "white_matter_tract", # Structural connectivity pathway (arcuate fasciculus, CC)
    "ontology_region",   # Named anatomical region in our region ontology vocabulary
    "circuit",           # Named functional circuit (e.g. hippocampal_circuit, fear_circuit)
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
    # Spectral phenotype / aperiodic reanalysis node types
    "spectral_estimate",
    "aperiodic_component",
    "periodic_peak",
    "spectral_run",
    "spectral_qc_assessment",
    "spectral_feature_bundle",
    "task_state_epoch",
    "channel",
    "electrode",
    "probe",
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
    "same_region_cross_modality",
    "same_task_cross_species",
    "same_region_same_task",
    "dataset_reanalysis_bridge_dataset",
    "dataset_old_dataset_new_method_candidate",
    "dataset_reinterpretation_candidate",
    "dataset_reprocessing_candidate",
    "paper_related_to_paper",
    "finding_supported_by_dataset",
    "finding_reported_by_paper",
    "finding_involves_region",
    "finding_involves_task",
    "finding_involves_modality",
    "finding_involves_species",
    "claim_supports_finding",
    "claim_contradicts_claim",
    "claim_supported_by_dataset",
    "claim_supported_by_paper",
    "claim_derived_from_finding",
    "finding_supports_finding",
    "finding_contradicts_finding",
    "region_co_occurs_with_region",
    "finding_has_frequency_band",
    "finding_has_temporal_pattern",
    "finding_has_spatial_frame",
    "experimental_design_requires_task",
    "experimental_design_requires_modality",
    "experimental_design_requires_behavior",
    "experimental_design_can_use_dataset",
    "region_is_child_of_region",          # Hierarchical parent-child in Allen CCF
    "region_structurally_adjacent_to",    # Siblings in Allen CCF hierarchy
    "ontology_region_maps_to_atlas",      # Our ontology_id → Allen structure
    "paper_cites_paper",                  # Citation relationship (citing → cited)
    "topic_encompasses_task",             # Topic includes a task
    "topic_encompasses_region",           # Topic involves a brain region
    "paper_foundational_for_topic",       # High-citation paper within a topic
    "finding_advances_topic",             # Finding contributes to a topic
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
    "concept_has_alias",
    "concept_in_domain",
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
    # Spectral phenotype / aperiodic reanalysis edge types
    "dataset_has_spectral_feature_bundle",
    "dataset_has_spectral_estimate",
    "spectral_estimate_generated_by_run",
    "spectral_estimate_has_aperiodic_component",
    "spectral_estimate_has_periodic_peak",
    "spectral_estimate_has_qc_assessment",
    "spectral_estimate_measured_in_region",
    "spectral_estimate_measured_during_state",
    "spectral_estimate_measured_from_channel",
    "aperiodic_component_estimated_by_method",
    "periodic_peak_estimated_by_method",
    "dataset_reanalyzable_by_pipeline",
    "dataset_missing_aperiodic_requirement",
    "dataset_supports_aperiodic_reanalysis",
    # ── Multi-scale KG extension edge types ───────────────────────────────
    # Methods / math layer
    "method_computes",                      # method → what it computes
    "method_assumes",                       # method → mathematical assumption
    "method_related_to_method",             # method ↔ related method
    "method_used_for_topic",                # method → topic it's commonly used in
    "method_measures_oscillation",          # method → oscillation it quantifies
    "circuit_studied_by_method",            # circuit → canonical analysis method
    # Species homology layer
    "region_has_homolog_in",               # region (species A) → region (species B)
    "finding_transfers_to_species",        # finding (rodent) → species (human)
    "paradigm_validated_cross_species",    # paradigm → species pair validation
    # Oscillation / spectral signatures
    "region_generates_oscillation",        # region → oscillation (frequency_band, condition)
    "oscillation_indexes_circuit",         # oscillation → functional circuit
    "oscillation_measured_by_method",      # oscillation → analysis method
    # HCP structural connectivity
    "region_structurally_connected",       # region → region (FA weight, pathway)
    # NeuroSynth topic-activation
    "topic_activates_region",              # topic → region (forward inference)
    "region_implicated_in_topic",          # region → topic (reverse inference)
    # Paradigm layer
    "paradigm_engages_circuit",            # paradigm → circuit
    "paradigm_targets_topic",              # paradigm → topic
    "paradigm_uses_method",                # paradigm → measurement method
    "paper_uses_paradigm",                 # paper → paradigm
    # Allen Mouse Connectivity
    "region_projects_to",                  # region → region (anterograde tracer, mouse)
    # Concept / theory hierarchy (Scholarpedia-inspired)
    "concept_narrower_than",               # concept → broader concept
    "concept_broader_than",                # concept → narrower concepts
    "concept_motivates_method",            # concept → analysis method it motivates
    "concept_testable_with_dataset",       # concept → dataset type that can test it
    "concept_related_to_topic",            # concept → research topic
    # Disorder-circuit mapping
    "disorder_disrupts_circuit",           # disorder → circuit it disrupts
    "disorder_has_biomarker",              # disorder → oscillation biomarker
    "disorder_modeled_by_paradigm",        # disorder → animal model paradigm
    "paper_involves_disorder",             # paper → disorder it studies
    "dataset_models_disorder",             # dataset → disorder model
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
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    sentence_id: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_span_order(self) -> GraphEvidence:
        if self.char_start is not None and self.char_end is not None and self.char_end < self.char_start:
            raise ValueError("char_end cannot be before char_start")
        return self

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
    """A lightweight, file-backed knowledge graph container.

    Accepts nodes/edges as either dicts (keyed by ID) or lists (auto-indexed).
    All builders pass lists; the validator normalises to dicts.
    """

    nodes: dict[str, KnowledgeGraphNode] | list[KnowledgeGraphNode] = Field(default_factory=dict)
    edges: dict[str, KnowledgeGraphEdge] | list[KnowledgeGraphEdge] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def coerce_lists_to_dicts(cls, values: Any) -> Any:
        if isinstance(values, dict):
            raw_nodes = values.get("nodes", {})
            raw_edges = values.get("edges", {})
            if isinstance(raw_nodes, list):
                values["nodes"] = {n["node_id"] if isinstance(n, dict) else n.node_id: n for n in raw_nodes}
            if isinstance(raw_edges, list):
                values["edges"] = {e["edge_id"] if isinstance(e, dict) else e.edge_id: e for e in raw_edges}
        return values

    @model_validator(mode="after")
    def validate_complete_graph(self) -> KnowledgeGraph:
        validate_graph(self)
        return self


def validate_graph(graph: KnowledgeGraph, strict: bool = False) -> KnowledgeGraph:
    """Validate graph identity maps and optionally edge references.

    When ``strict=False`` (default) dangling edge endpoints are allowed so that
    partial KG layers (e.g. a builder that only creates edges to nodes defined in
    another layer) can be constructed and later merged without errors.
    Pass ``strict=True`` to enforce that every edge endpoint resolves to a node
    present in *this* graph.
    """

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
        if strict:
            if edge.source_node_id not in node_ids:
                raise ValueError(f"edge source does not resolve: {edge.source_node_id}")
            if edge.target_node_id not in node_ids:
                raise ValueError(f"edge target does not resolve: {edge.target_node_id}")

    return graph


def _infer_stub_node_type(node_id: str) -> str:
    parts = node_id.split(":")
    if parts and parts[0] == "node" and len(parts) >= 3:
        return parts[1]
    return parts[0] if parts and parts[0] else "unknown"


def resolve_dangling_edges(graph: KnowledgeGraph) -> tuple[KnowledgeGraph, int]:
    """Create minimal placeholder nodes for any edge endpoint missing from the graph.

    Individual KG builder modules in this repo use a mix of node-id
    conventions (the canonical ``make_node_id`` scheme and many hand-rolled
    ``type:id`` schemes), and are frequently authored/merged independently,
    so cross-builder or self-referential dangling edges are common (see
    ``reports/architecture_connectivity_audit_2026-07-01.md``). Rather than
    special-casing every builder's target vocabulary, this creates a minimal
    node for every dangling endpoint, inferring node_type from the id's
    leading segment (works for both ``node:type:...`` and ``type:...`` ids).

    Placeholder nodes are marked ``properties={"stub": True, ...}`` so
    consumers can distinguish them from richly-sourced nodes; they carry a
    low confidence (0.3) since they represent "this concept exists and is
    referenced" rather than validated content about it.

    Returns the graph (nodes dict extended in place is NOT mutated — a new
    KnowledgeGraph is returned) and the number of stub nodes created.
    """

    existing_ids = set(graph.nodes.keys())
    referenced_ids: set[str] = set()
    for edge in graph.edges.values():
        referenced_ids.add(edge.source_node_id)
        referenced_ids.add(edge.target_node_id)
    missing_ids = sorted(referenced_ids - existing_ids)

    if not missing_ids:
        return graph, 0

    new_nodes = dict(graph.nodes)
    for node_id in missing_ids:
        node_type = _infer_stub_node_type(node_id)
        label_part = node_id.split(":")[-1]
        new_nodes[node_id] = KnowledgeGraphNode(
            node_id=node_id,
            node_type=node_type,
            label=label_part.replace("_", " ").title(),
            properties={"stub": True, "source": "auto_generated_placeholder"},
            confidence=0.3,
        )

    resolved = KnowledgeGraph(
        nodes=new_nodes,
        edges=dict(graph.edges),
        metadata=dict(graph.metadata),
    )
    return resolved, len(missing_ids)


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


# ── Convenience aliases used by all builder modules ────────────────────────
# Builders import `GraphNode, GraphEdge, KnowledgeGraph` and pass lists of
# nodes/edges; KnowledgeGraph.coerce_lists_to_dicts handles the conversion.
GraphNode = KnowledgeGraphNode
GraphEdge = KnowledgeGraphEdge
