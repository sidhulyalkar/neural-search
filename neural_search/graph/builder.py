"""Build knowledge graphs from normalized Neural Search records."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from neural_search.awareness.taxonomy import DATA_FORMS, DataForm
from neural_search.graph.schema import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
    normalize_node_type,
    validate_graph,
)
from neural_search.normalized import NormalizedRecord
from neural_search.schemas import (
    AnalysisAffordance,
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
)
from neural_search.species import SpeciesProfile, get_species_profile

GRAPH_BUILDER_NAME = "neural_search.graph.builder"
GRAPH_BUILDER_VERSION = "v0.5.0"
TAXONOMY_EXTRACTOR_NAME = "neural_search.awareness.taxonomy"
TAXONOMY_EXTRACTOR_VERSION = "v0.7.0"

DATASET_LABEL_FIELDS: Mapping[str, tuple[str, str]] = {
    "tasks": ("task", "dataset_has_task"),
    "modalities": ("modality", "dataset_has_modality"),
    "brain_regions": ("brain_region", "dataset_records_region"),
    "species": ("species", "dataset_has_species"),
    "behavioral_events": ("behavioral_event", "dataset_has_behavioral_event"),
    "data_standards": ("data_standard", "dataset_uses_standard"),
    "file_formats": ("file_format", "dataset_has_file_format"),
}

PAPER_LABEL_FIELDS: Mapping[str, tuple[str, str]] = {
    "task": ("task", "paper_studies_task"),
    "modality": ("modality", "paper_uses_modality"),
    "brain_region": ("brain_region", "paper_mentions_region"),
    "analysis_method": ("analysis_method", "paper_uses_method"),
    "modeling_method": ("modeling_method", "paper_uses_method"),
    "finding": ("finding", "paper_reports_finding"),
}

BEHAVIORAL_REQUIREMENT_SIGNALS = {
    "events",
    "trials",
    "spike_times",
    "sweeps",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parts_from_record_id(record_id: str, expected_type: str) -> list[str]:
    parts = record_id.split(":")
    if len(parts) >= 3 and parts[0] == expected_type:
        return parts[1:]
    return [record_id]


def dataset_node_id(record: NormalizedDatasetRecord | str) -> str:
    """Return the graph node ID for a normalized dataset record or ID."""

    dataset_id = record.dataset_id if isinstance(record, NormalizedDatasetRecord) else record
    return make_node_id("dataset", *_parts_from_record_id(dataset_id, "dataset"))


def paper_node_id(record: NormalizedPaperRecord | str) -> str:
    """Return the graph node ID for a normalized paper record or ID."""

    paper_id = record.paper_id if isinstance(record, NormalizedPaperRecord) else record
    return make_node_id("paper", *_parts_from_record_id(paper_id, "paper"))


def _concept_key(label: EvidenceLabel) -> str:
    if label.id.startswith("label:"):
        return normalize_node_type(label.id.split(":")[-1])
    return normalize_node_type(label.id or label.label)


def _concept_node_id(node_type: str, label: EvidenceLabel) -> str:
    return make_node_id(node_type, _concept_key(label))


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def _record_evidence(
    *,
    source_type: str,
    source_id: str,
    source_field: str | None,
    evidence_text: str | None,
    confidence: float,
    extractor_name: str | None = None,
    extractor_version: str | None = None,
) -> GraphEvidence:
    stable_field = source_field or "record"
    return GraphEvidence(
        evidence_id=f"evidence:{source_type}:{source_id}:{stable_field}:{evidence_text or source_id}",
        source_type=source_type,
        source_id=source_id,
        source_field=source_field,
        evidence_text=evidence_text,
        confidence=confidence,
        extractor_name=extractor_name or GRAPH_BUILDER_NAME,
        extractor_version=extractor_version or GRAPH_BUILDER_VERSION,
    )


def _label_evidence(
    label: EvidenceLabel,
    *,
    source_type: str,
    source_id: str,
    source_field: str,
) -> GraphEvidence:
    return _record_evidence(
        source_type=source_type,
        source_id=source_id,
        source_field=label.source_field or source_field,
        evidence_text=label.evidence_text or label.source_value or label.label,
        confidence=label.confidence,
        extractor_name=label.extractor_name,
        extractor_version=label.extractor_version,
    )


def _analysis_evidence(
    affordance: AnalysisAffordance,
    *,
    source_id: str,
) -> GraphEvidence:
    evidence_text = "; ".join(affordance.evidence) or affordance.support_level
    return _record_evidence(
        source_type="normalized_dataset",
        source_id=source_id,
        source_field="analysis_affordances",
        evidence_text=evidence_text,
        confidence=affordance.confidence,
        extractor_name=affordance.detector_name,
        extractor_version=affordance.detector_version,
    )


def _dataset_node(record: NormalizedDatasetRecord) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=dataset_node_id(record),
        node_type="dataset",
        label=record.title,
        aliases=_dedupe([record.source_id, record.dataset_id]),
        source_ids=[record.dataset_id],
        properties={
            "source": record.source,
            "source_id": record.source_id,
            "url": record.url,
            "raw_payload_path": record.raw_payload_path,
            "missing_fields": record.missing_fields,
            "usability_flags": record.usability_flags.model_dump(mode="json"),
            "extractor_version": record.extractor_version,
        },
        evidence=[
            _record_evidence(
                source_type="normalized_dataset",
                source_id=record.dataset_id,
                source_field="title",
                evidence_text=record.title,
                confidence=1.0,
                extractor_version=record.extractor_version,
            )
        ],
        confidence=1.0,
        created_at=record.created_at,
    )


def _paper_node(record: NormalizedPaperRecord | str) -> KnowledgeGraphNode:
    if isinstance(record, str):
        return KnowledgeGraphNode(
            node_id=paper_node_id(record),
            node_type="paper",
            label=record,
            aliases=[record],
            source_ids=[record],
            properties={"placeholder": True},
            confidence=0.35,
            created_at=_now(),
        )
    return KnowledgeGraphNode(
        node_id=paper_node_id(record),
        node_type="paper",
        label=record.title,
        aliases=_dedupe([record.source_id, record.paper_id, record.doi or ""]),
        source_ids=[record.paper_id],
        properties={
            "source": record.source,
            "source_id": record.source_id,
            "doi": record.doi,
            "url": record.url,
            "year": record.year,
            "raw_payload_path": record.raw_payload_path,
            "extractor_version": record.extractor_version,
        },
        evidence=[
            _record_evidence(
                source_type="normalized_paper",
                source_id=record.paper_id,
                source_field="title",
                evidence_text=record.title,
                confidence=1.0,
                extractor_version=record.extractor_version,
            )
        ],
        confidence=1.0,
        created_at=record.created_at,
    )


def _concept_node(
    node_type: str,
    label: EvidenceLabel,
    *,
    source_id: str,
    source_type: str,
    source_field: str,
) -> KnowledgeGraphNode:
    evidence = _label_evidence(
        label,
        source_type=source_type,
        source_id=source_id,
        source_field=source_field,
    )
    return KnowledgeGraphNode(
        node_id=_concept_node_id(node_type, label),
        node_type=node_type,
        label=label.label,
        aliases=_dedupe([label.id, label.label, label.evidence_text or ""]),
        source_ids=[source_id],
        properties={"label_type": label.label_type},
        evidence=[evidence],
        confidence=label.confidence,
        created_at=_now(),
    )


def _analysis_node(
    affordance: AnalysisAffordance,
    *,
    source_id: str,
) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id("analysis_affordance", affordance.analysis_id),
        node_type="analysis_affordance",
        label=affordance.analysis_id.replace("_", " ").title(),
        aliases=[affordance.analysis_id],
        source_ids=[source_id],
        properties={
            "support_level": affordance.support_level,
            "required_fields_present": affordance.required_fields_present,
            "helpful_fields_present": affordance.helpful_fields_present,
            "missing_fields": affordance.missing_fields,
        },
        evidence=[_analysis_evidence(affordance, source_id=source_id)],
        confidence=affordance.confidence,
        created_at=_now(),
    )


def _author_node(author: str, *, source_id: str) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id("author", normalize_node_type(author)),
        node_type="author",
        label=author,
        aliases=[author],
        source_ids=[source_id],
        properties={},
        evidence=[
            _record_evidence(
                source_type="normalized_paper",
                source_id=source_id,
                source_field="authors",
                evidence_text=author,
                confidence=0.9,
            )
        ],
        confidence=0.9,
        created_at=_now(),
    )


def _taxonomy_evidence(
    *,
    data_form: DataForm,
    source_field: str,
    evidence_text: str,
) -> GraphEvidence:
    return _record_evidence(
        source_type="awareness_taxonomy",
        source_id=data_form.id,
        source_field=source_field,
        evidence_text=evidence_text,
        confidence=0.9,
        extractor_name=TAXONOMY_EXTRACTOR_NAME,
        extractor_version=TAXONOMY_EXTRACTOR_VERSION,
    )


def _taxonomy_label(
    *,
    label_type: str,
    label: str,
    data_form: DataForm,
    source_field: str,
) -> EvidenceLabel:
    return EvidenceLabel(
        id=f"label:{label_type}:{normalize_node_type(label)}",
        label=label,
        label_type=label_type,
        confidence=0.9,
        evidence_text=f"{data_form.label} requires {label}",
        source_field=source_field,
        source_value=label,
        extractor_name=TAXONOMY_EXTRACTOR_NAME,
        extractor_version=TAXONOMY_EXTRACTOR_VERSION,
    )


def _species_taxonomy_evidence(
    profile: SpeciesProfile,
    *,
    source_id: str,
    source_field: str,
    evidence_text: str,
) -> GraphEvidence:
    return _record_evidence(
        source_type="species_taxonomy",
        source_id=profile.species_id,
        source_field=source_field,
        evidence_text=evidence_text,
        confidence=0.88,
        extractor_name="neural_search.species",
        extractor_version="v0.8.0",
    )


def _species_label(label: EvidenceLabel, profile: SpeciesProfile) -> EvidenceLabel:
    return label.model_copy(
        update={
            "id": f"label:species:{profile.species_id}",
            "label": profile.label,
            "evidence_text": label.evidence_text or label.label,
        }
    )


def _simple_taxonomy_node(
    *,
    node_type: str,
    node_id_part: str,
    label: str,
    aliases: list[str],
    source_id: str,
    evidence: GraphEvidence,
) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id(node_type, node_id_part),
        node_type=node_type,
        label=label,
        aliases=_dedupe([node_id_part, label, *aliases]),
        source_ids=[source_id],
        properties={},
        evidence=[evidence],
        confidence=evidence.confidence,
        created_at=_now(),
    )


def _taxonomy_concept_node(
    *,
    node_type: str,
    label_type: str,
    label: str,
    data_form: DataForm,
    source_field: str,
) -> KnowledgeGraphNode:
    return _concept_node(
        node_type,
        _taxonomy_label(
            label_type=label_type,
            label=label,
            data_form=data_form,
            source_field=source_field,
        ),
        source_id=data_form.id,
        source_type="awareness_taxonomy",
        source_field=source_field,
    )


def _taxonomy_analysis_node(
    analysis_id: str,
    data_form: DataForm,
) -> KnowledgeGraphNode:
    evidence = _taxonomy_evidence(
        data_form=data_form,
        source_field="analysis_families",
        evidence_text=f"{data_form.label} supports {analysis_id}",
    )
    return KnowledgeGraphNode(
        node_id=make_node_id("analysis_affordance", analysis_id),
        node_type="analysis_affordance",
        label=analysis_id.replace("_", " ").title(),
        aliases=[analysis_id, data_form.label],
        source_ids=[data_form.id],
        properties={"taxonomy_data_form": data_form.id},
        evidence=[evidence],
        confidence=0.9,
        created_at=_now(),
    )


def _edge(
    source_node_id: str,
    edge_type: str,
    target_node_id: str,
    *,
    confidence: float,
    evidence: list[GraphEvidence],
    properties: dict[str, Any] | None = None,
) -> KnowledgeGraphEdge:
    return KnowledgeGraphEdge(
        edge_id=make_edge_id(source_node_id, edge_type, target_node_id),
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        directed=True,
        confidence=confidence,
        evidence=evidence,
        properties=properties or {},
        created_at=_now(),
    )


def _add_node(nodes: dict[str, KnowledgeGraphNode], node: KnowledgeGraphNode) -> None:
    existing = nodes.get(node.node_id)
    if existing is None:
        nodes[node.node_id] = node
        return
    nodes[node.node_id] = _merge_node(existing, node)


def _add_edge(edges: dict[str, KnowledgeGraphEdge], edge: KnowledgeGraphEdge) -> None:
    existing = edges.get(edge.edge_id)
    if existing is None:
        edges[edge.edge_id] = edge
        return
    edges[edge.edge_id] = _merge_edge(existing, edge)


def _merge_node(left: KnowledgeGraphNode, right: KnowledgeGraphNode) -> KnowledgeGraphNode:
    properties = {**left.properties, **right.properties}
    if left.properties.get("placeholder") and not right.properties.get("placeholder"):
        properties.pop("placeholder", None)
    return left.model_copy(
        update={
            "label": right.label if left.properties.get("placeholder") else left.label,
            "aliases": _dedupe([*left.aliases, *right.aliases]),
            "source_ids": _dedupe([*left.source_ids, *right.source_ids]),
            "properties": properties,
            "evidence": [*left.evidence, *right.evidence],
            "confidence": max(left.confidence, right.confidence),
            "updated_at": _now(),
        }
    )


def _merge_edge(left: KnowledgeGraphEdge, right: KnowledgeGraphEdge) -> KnowledgeGraphEdge:
    return left.model_copy(
        update={
            "confidence": max(left.confidence, right.confidence),
            "evidence": [*left.evidence, *right.evidence],
            "properties": {**left.properties, **right.properties},
            "updated_at": _now(),
        }
    )


def _graph_from_parts(
    nodes: dict[str, KnowledgeGraphNode],
    edges: dict[str, KnowledgeGraphEdge],
    *,
    metadata: dict[str, Any],
) -> KnowledgeGraph:
    return KnowledgeGraph(nodes=nodes, edges=edges, metadata=metadata)


def build_dataset_subgraph(
    dataset: NormalizedDatasetRecord,
    *,
    min_confidence: float = 0.5,
) -> KnowledgeGraph:
    """Build a graph slice from one normalized dataset record."""

    nodes: dict[str, KnowledgeGraphNode] = {}
    edges: dict[str, KnowledgeGraphEdge] = {}
    dataset_node = _dataset_node(dataset)
    _add_node(nodes, dataset_node)

    for field_name, (node_type, edge_type) in DATASET_LABEL_FIELDS.items():
        for label in getattr(dataset, field_name):
            if label.confidence < min_confidence:
                continue
            concept_label = label
            species_profile = get_species_profile(label.label) if field_name == "species" else None
            if species_profile is not None:
                concept_label = _species_label(label, species_profile)
            concept = _concept_node(
                node_type,
                concept_label,
                source_id=dataset.dataset_id,
                source_type="normalized_dataset",
                source_field=field_name,
            )
            evidence = _label_evidence(
                concept_label,
                source_type="normalized_dataset",
                source_id=dataset.dataset_id,
                source_field=field_name,
            )
            _add_node(nodes, concept)
            _add_edge(
                edges,
                _edge(
                    dataset_node.node_id,
                    edge_type,
                    concept.node_id,
                    confidence=label.confidence,
                    evidence=[evidence],
                    properties={"source_field": field_name},
                ),
            )
            if species_profile is not None:
                _add_species_taxonomy_edges(
                    nodes,
                    edges,
                    concept.node_id,
                    species_profile,
                    source_id=dataset.dataset_id,
                )

    for affordance in dataset.analysis_affordances:
        if affordance.confidence < min_confidence or affordance.support_level == "unsupported":
            continue
        analysis = _analysis_node(affordance, source_id=dataset.dataset_id)
        evidence = _analysis_evidence(affordance, source_id=dataset.dataset_id)
        _add_node(nodes, analysis)
        _add_edge(
            edges,
            _edge(
                dataset_node.node_id,
                "dataset_supports_analysis",
                analysis.node_id,
                confidence=affordance.confidence,
                evidence=[evidence],
                properties={"support_level": affordance.support_level},
            ),
        )

    for linked_paper_id in dataset.linked_papers:
        paper = _paper_node(linked_paper_id)
        evidence = _record_evidence(
            source_type="normalized_dataset",
            source_id=dataset.dataset_id,
            source_field="linked_papers",
            evidence_text=linked_paper_id,
            confidence=0.7,
            extractor_version=dataset.extractor_version,
        )
        _add_node(nodes, paper)
        _add_edge(
            edges,
            _edge(
                paper.node_id,
                "paper_mentions_dataset",
                dataset_node.node_id,
                confidence=0.7,
                evidence=[evidence],
                properties={"link_strength": "weak", "source_field": "linked_papers"},
            ),
        )

    return _graph_from_parts(
        nodes,
        edges,
        metadata={
            "graph_version": GRAPH_BUILDER_VERSION,
            "builder": GRAPH_BUILDER_NAME,
            "record_count": 1,
            "dataset_count": 1,
            "paper_count": 0,
        },
    )


def _add_species_taxonomy_edges(
    nodes: dict[str, KnowledgeGraphNode],
    edges: dict[str, KnowledgeGraphEdge],
    species_node_id: str,
    profile: SpeciesProfile,
    *,
    source_id: str,
) -> None:
    """Attach broader organism context to a canonical species node."""

    for group in profile.taxon_groups:
        evidence = _species_taxonomy_evidence(
            profile,
            source_id=source_id,
            source_field="taxon_groups",
            evidence_text=f"{profile.label} belongs to {group}",
        )
        group_node = _simple_taxonomy_node(
            node_type="taxon_group",
            node_id_part=group,
            label=group.replace("_", " ").title(),
            aliases=[group.replace("_", " ")],
            source_id=profile.species_id,
            evidence=evidence,
        )
        _add_node(nodes, group_node)
        _add_edge(
            edges,
            _edge(
                species_node_id,
                "species_in_taxon_group",
                group_node.node_id,
                confidence=evidence.confidence,
                evidence=[evidence],
                properties={"taxon_id": profile.taxon_id},
            ),
        )

    animal_evidence = _species_taxonomy_evidence(
        profile,
        source_id=source_id,
        source_field="animal_type",
        evidence_text=f"{profile.label} is represented as {profile.animal_type}",
    )
    animal_node = _simple_taxonomy_node(
        node_type="organism_model",
        node_id_part=profile.animal_type,
        label=profile.animal_type.replace("_", " ").title(),
        aliases=[profile.animal_type.replace("_", " ")],
        source_id=profile.species_id,
        evidence=animal_evidence,
    )
    _add_node(nodes, animal_node)
    _add_edge(
        edges,
        _edge(
            species_node_id,
            "species_has_animal_type",
            animal_node.node_id,
            confidence=animal_evidence.confidence,
            evidence=[animal_evidence],
            properties={"taxon_id": profile.taxon_id},
        ),
    )

    for role in profile.model_roles:
        evidence = _species_taxonomy_evidence(
            profile,
            source_id=source_id,
            source_field="model_roles",
            evidence_text=f"{profile.label} supports model role {role}",
        )
        role_node = _simple_taxonomy_node(
            node_type="organism_model",
            node_id_part=role,
            label=role.replace("_", " ").title(),
            aliases=[role.replace("_", " ")],
            source_id=profile.species_id,
            evidence=evidence,
        )
        _add_node(nodes, role_node)
        _add_edge(
            edges,
            _edge(
                species_node_id,
                "species_has_model_role",
                role_node.node_id,
                confidence=evidence.confidence,
                evidence=[evidence],
                properties={"taxon_id": profile.taxon_id},
            ),
        )


def build_paper_subgraph(
    paper: NormalizedPaperRecord,
    *,
    min_confidence: float = 0.5,
) -> KnowledgeGraph:
    """Build a graph slice from one normalized paper record."""

    nodes: dict[str, KnowledgeGraphNode] = {}
    edges: dict[str, KnowledgeGraphEdge] = {}
    paper_node = _paper_node(paper)
    _add_node(nodes, paper_node)

    for author in paper.authors:
        if not author.strip():
            continue
        # Skip authors whose names normalize to empty identifiers
        try:
            author_node = _author_node(author, source_id=paper.paper_id)
        except ValueError:
            continue
        evidence = author_node.evidence[0]
        _add_node(nodes, author_node)
        _add_edge(
            edges,
            _edge(
                paper_node.node_id,
                "paper_has_author",
                author_node.node_id,
                confidence=0.9,
                evidence=[evidence],
                properties={"source_field": "authors"},
            ),
        )

    for label in paper.extracted_labels:
        if label.confidence < min_confidence:
            continue
        node_type, edge_type = PAPER_LABEL_FIELDS.get(
            label.label_type,
            (label.label_type, f"paper_mentions_{label.label_type}"),
        )
        concept = _concept_node(
            node_type,
            label,
            source_id=paper.paper_id,
            source_type="normalized_paper",
            source_field="extracted_labels",
        )
        evidence = _label_evidence(
            label,
            source_type="normalized_paper",
            source_id=paper.paper_id,
            source_field="extracted_labels",
        )
        _add_node(nodes, concept)
        _add_edge(
            edges,
            _edge(
                paper_node.node_id,
                edge_type,
                concept.node_id,
                confidence=label.confidence,
                evidence=[evidence],
                properties={"source_field": "extracted_labels"},
            ),
        )

    for linked_dataset_id in paper.linked_datasets:
        dataset_node = KnowledgeGraphNode(
            node_id=dataset_node_id(linked_dataset_id),
            node_type="dataset",
            label=linked_dataset_id,
            aliases=[linked_dataset_id],
            source_ids=[linked_dataset_id],
            properties={"placeholder": True},
            confidence=0.35,
            created_at=_now(),
        )
        evidence = _record_evidence(
            source_type="normalized_paper",
            source_id=paper.paper_id,
            source_field="linked_datasets",
            evidence_text=linked_dataset_id,
            confidence=0.95,
            extractor_version=paper.extractor_version,
        )
        _add_node(nodes, dataset_node)
        _add_edge(
            edges,
            _edge(
                paper_node.node_id,
                "paper_uses_dataset",
                dataset_node.node_id,
                confidence=0.95,
                evidence=[evidence],
                properties={"link_strength": "strong", "source_field": "linked_datasets"},
            ),
        )

    return _graph_from_parts(
        nodes,
        edges,
        metadata={
            "graph_version": GRAPH_BUILDER_VERSION,
            "builder": GRAPH_BUILDER_NAME,
            "record_count": 1,
            "dataset_count": 0,
            "paper_count": 1,
        },
    )


def build_taxonomy_requirement_subgraph() -> KnowledgeGraph:
    """Build graph edges that encode data-form analysis requirements."""

    nodes: dict[str, KnowledgeGraphNode] = {}
    edges: dict[str, KnowledgeGraphEdge] = {}
    for data_form in DATA_FORMS.values():
        for analysis_id in data_form.analysis_families:
            analysis_node = _taxonomy_analysis_node(analysis_id, data_form)
            _add_node(nodes, analysis_node)

            for modality in data_form.modalities:
                modality_node = _taxonomy_concept_node(
                    node_type="modality",
                    label_type="modality",
                    label=modality,
                    data_form=data_form,
                    source_field="modalities",
                )
                evidence = _taxonomy_evidence(
                    data_form=data_form,
                    source_field="modalities",
                    evidence_text=f"{analysis_id} requires modality {modality}",
                )
                _add_node(nodes, modality_node)
                _add_edge(
                    edges,
                    _edge(
                        analysis_node.node_id,
                        "analysis_requires_modality",
                        modality_node.node_id,
                        confidence=0.9,
                        evidence=[evidence],
                        properties={
                            "data_form": data_form.id,
                            "requirement_type": "modality",
                        },
                    ),
                )

            for standard in data_form.standards:
                standard_node = _taxonomy_concept_node(
                    node_type="data_standard",
                    label_type="data_standard",
                    label=standard,
                    data_form=data_form,
                    source_field="standards",
                )
                evidence = _taxonomy_evidence(
                    data_form=data_form,
                    source_field="standards",
                    evidence_text=f"{analysis_id} benefits from standard {standard}",
                )
                _add_node(nodes, standard_node)
                _add_edge(
                    edges,
                    _edge(
                        analysis_node.node_id,
                        "analysis_requires_task_structure",
                        standard_node.node_id,
                        confidence=0.82,
                        evidence=[evidence],
                        properties={
                            "data_form": data_form.id,
                            "requirement_type": "data_standard",
                        },
                    ),
                )

            for signal in data_form.required_signals:
                signal_node = _taxonomy_concept_node(
                    node_type="required_signal",
                    label_type="required_signal",
                    label=signal,
                    data_form=data_form,
                    source_field="required_signals",
                )
                evidence = _taxonomy_evidence(
                    data_form=data_form,
                    source_field="required_signals",
                    evidence_text=f"{analysis_id} requires signal {signal}",
                )
                _add_node(nodes, signal_node)
                _add_edge(
                    edges,
                    _edge(
                        analysis_node.node_id,
                        "analysis_requires_task_structure",
                        signal_node.node_id,
                        confidence=0.86,
                        evidence=[evidence],
                        properties={
                            "data_form": data_form.id,
                            "requirement_type": "required_signal",
                        },
                    ),
                )
                if signal in BEHAVIORAL_REQUIREMENT_SIGNALS:
                    event_node = _taxonomy_concept_node(
                        node_type="behavioral_event",
                        label_type="behavioral_event",
                        label=signal,
                        data_form=data_form,
                        source_field="required_signals",
                    )
                    _add_node(nodes, event_node)
                    _add_edge(
                        edges,
                        _edge(
                            analysis_node.node_id,
                            "analysis_requires_behavioral_event",
                            event_node.node_id,
                            confidence=0.8,
                            evidence=[evidence],
                            properties={
                                "data_form": data_form.id,
                                "requirement_type": "required_signal",
                            },
                        ),
                    )

    return _graph_from_parts(
        nodes,
        edges,
        metadata={
            "graph_version": GRAPH_BUILDER_VERSION,
            "builder": GRAPH_BUILDER_NAME,
            "taxonomy_source": TAXONOMY_EXTRACTOR_NAME,
            "taxonomy_data_form_count": len(DATA_FORMS),
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    )


def merge_graphs(graphs: Iterable[KnowledgeGraph]) -> KnowledgeGraph:
    """Merge graph slices while preserving evidence and resolving placeholders."""

    nodes: dict[str, KnowledgeGraphNode] = {}
    edges: dict[str, KnowledgeGraphEdge] = {}
    graph_count = 0
    for graph in graphs:
        graph_count += 1
        for node in graph.nodes.values():
            _add_node(nodes, node)
        for edge in graph.edges.values():
            _add_edge(edges, edge)

    merged = KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        metadata={
            "graph_version": GRAPH_BUILDER_VERSION,
            "builder": GRAPH_BUILDER_NAME,
            "merged_graph_count": graph_count,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "created_at": _now(),
        },
    )
    return validate_graph(merged)


def build_graph_from_records(
    datasets: Iterable[NormalizedDatasetRecord] = (),
    papers: Iterable[NormalizedPaperRecord] = (),
    *,
    min_confidence: float = 0.5,
) -> KnowledgeGraph:
    """Build a provenance-aware graph from normalized datasets and papers."""

    dataset_records = list(datasets)
    paper_records = list(papers)
    subgraphs: list[KnowledgeGraph] = [
        build_dataset_subgraph(dataset, min_confidence=min_confidence)
        for dataset in dataset_records
    ]
    subgraphs.extend(
        build_paper_subgraph(paper, min_confidence=min_confidence)
        for paper in paper_records
    )
    subgraphs.append(build_taxonomy_requirement_subgraph())
    graph = merge_graphs(subgraphs)
    graph.metadata.update(
        {
            "dataset_count": len(dataset_records),
            "paper_count": len(paper_records),
            "record_count": len(dataset_records) + len(paper_records),
            "min_confidence": min_confidence,
            "taxonomy_requirement_edges": len(
                [
                    edge
                    for edge in graph.edges.values()
                    if edge.edge_type.startswith("analysis_requires_")
                ]
            ),
        }
    )
    return validate_graph(graph)


def split_records(
    records: Iterable[NormalizedRecord],
) -> tuple[list[NormalizedDatasetRecord], list[NormalizedPaperRecord]]:
    """Split mixed normalized records into dataset and paper lists."""

    datasets: list[NormalizedDatasetRecord] = []
    papers: list[NormalizedPaperRecord] = []
    for record in records:
        if isinstance(record, NormalizedDatasetRecord):
            datasets.append(record)
        elif isinstance(record, NormalizedPaperRecord):
            papers.append(record)
    return datasets, papers
