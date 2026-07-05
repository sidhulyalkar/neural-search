"""Add literature paper and finding artifacts to the knowledge graph."""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.graph.schema import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
    normalize_node_type,
)
from neural_search.normalized import make_paper_id
from neural_search.ontology.cognitive_atlas import get_cogat_match
from neural_search.ontology.loader import (
    get_region_atlas_refs,
    get_region_id_by_alias,
    get_task_id_by_alias,
)

KG_BUILDER_NAME = "neural_search.literature.kg_builder"
KG_BUILDER_VERSION = "v0.1.0"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    paths = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    for child in paths:
        if not child.exists():
            continue
        with child.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload


def _evidence(
    *,
    source_type: str,
    source_id: str,
    source_field: str,
    evidence_text: str | None,
    confidence: float,
    char_start: int | None = None,
    char_end: int | None = None,
    sentence_id: int | None = None,
) -> GraphEvidence:
    return GraphEvidence(
        evidence_id=(
            f"evidence:literature:{source_type}:{source_id}:"
            f"{source_field}:{evidence_text or source_id}"
        ),
        source_type=source_type,
        source_id=source_id,
        source_field=source_field,
        evidence_text=evidence_text,
        confidence=confidence,
        extractor_name=KG_BUILDER_NAME,
        extractor_version=KG_BUILDER_VERSION,
        char_start=char_start,
        char_end=char_end,
        sentence_id=sentence_id,
    )


def _add_node(graph: KnowledgeGraph, node: KnowledgeGraphNode) -> bool:
    existing = graph.nodes.get(node.node_id)
    if existing is None:
        graph.nodes[node.node_id] = node
        return True
    graph.nodes[node.node_id] = existing.model_copy(
        update={
            "label": node.label if existing.properties.get("placeholder") else existing.label,
            "aliases": _dedupe([*existing.aliases, *node.aliases]),
            "source_ids": _dedupe([*existing.source_ids, *node.source_ids]),
            "properties": {**existing.properties, **node.properties},
            "evidence": [*existing.evidence, *node.evidence],
            "confidence": max(existing.confidence, node.confidence),
            "updated_at": _now(),
        }
    )
    graph.nodes[node.node_id].properties.pop("placeholder", None)
    return False


def _add_edge(graph: KnowledgeGraph, edge: KnowledgeGraphEdge) -> bool:
    existing = graph.edges.get(edge.edge_id)
    if existing is None:
        graph.edges[edge.edge_id] = edge
        return True
    graph.edges[edge.edge_id] = existing.model_copy(
        update={
            "confidence": max(existing.confidence, edge.confidence),
            "evidence": [*existing.evidence, *edge.evidence],
            "properties": {**existing.properties, **edge.properties},
            "updated_at": _now(),
        }
    )
    return False


def _dedupe(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            ordered.append(cleaned)
    return ordered


def _edge(
    source_node_id: str,
    edge_type: str,
    target_node_id: str,
    *,
    evidence: GraphEvidence,
    confidence: float,
    properties: dict[str, Any] | None = None,
) -> KnowledgeGraphEdge:
    return KnowledgeGraphEdge(
        edge_id=make_edge_id(source_node_id, edge_type, target_node_id),
        source_node_id=source_node_id,
        target_node_id=target_node_id,
        edge_type=edge_type,
        confidence=confidence,
        evidence=[evidence],
        properties=properties or {},
        created_at=_now(),
    )


def _paper_node(record: dict[str, Any]) -> KnowledgeGraphNode:
    paper_id = str(record.get("paper_id") or make_paper_id("openalex", record["source_id"]))
    title = str(record.get("title") or paper_id)
    evidence = _evidence(
        source_type="openalex_work",
        source_id=paper_id,
        source_field="title",
        evidence_text=title,
        confidence=1.0,
    )
    return KnowledgeGraphNode(
        node_id=make_node_id("paper", *paper_id.split(":")[1:]),
        node_type="paper",
        label=title,
        aliases=_dedupe([paper_id, str(record.get("source_id", "")), str(record.get("doi", ""))]),
        source_ids=[paper_id],
        properties={
            "source": record.get("source", "openalex"),
            "source_id": record.get("source_id"),
            "abstract": record.get("abstract"),
            "doi": record.get("doi"),
            "url": record.get("url"),
            "year": record.get("year"),
            "citation_count": record.get("citation_count", 0),
            "open_access_url": record.get("open_access_url"),
            "topics": record.get("topics", []),
        },
        evidence=[evidence],
        confidence=1.0,
        created_at=str(record.get("created_at") or _now()),
    )


def _venue_node(venue: str, *, paper_id: str) -> KnowledgeGraphNode:
    evidence = _evidence(
        source_type="openalex_work",
        source_id=paper_id,
        source_field="venue",
        evidence_text=venue,
        confidence=0.95,
    )
    return KnowledgeGraphNode(
        node_id=make_node_id("venue", normalize_node_type(venue)),
        node_type="venue",
        label=venue,
        aliases=[venue],
        source_ids=[paper_id],
        properties={},
        evidence=[evidence],
        confidence=0.95,
        created_at=_now(),
    )


def _dataset_placeholder(dataset_record_id: str) -> KnowledgeGraphNode:
    parts = dataset_record_id.split(":")
    node_parts = parts[1:] if parts and parts[0] == "dataset" else parts
    return KnowledgeGraphNode(
        node_id=make_node_id("dataset", *node_parts),
        node_type="dataset",
        label=dataset_record_id,
        aliases=[dataset_record_id],
        source_ids=[dataset_record_id],
        properties={"placeholder": True},
        confidence=0.35,
        created_at=_now(),
    )


def _link_paper_id(link: dict[str, Any]) -> str:
    openalex_id = str(link.get("paper_openalex_id") or "")
    if openalex_id.startswith("paper:"):
        return openalex_id
    return make_paper_id("openalex", openalex_id)


def _load_links(links_path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if links_path is None or not links_path.exists():
        return {}
    by_paper: dict[str, list[dict[str, Any]]] = {}
    for link in _iter_jsonl(links_path):
        if not link.get("paper_openalex_id") or link.get("match_method") == "not_found":
            continue
        by_paper.setdefault(_link_paper_id(link), []).append(link)
    return by_paper


def add_papers_from_shards(
    graph: KnowledgeGraph,
    shard_dir: Path,
    links_path: Path | None = None,
) -> dict[str, int]:
    """Add paper, venue, and dataset-link nodes from OpenAlex JSONL shards."""

    links_by_paper = _load_links(links_path)
    stats = {"papers_added": 0, "venues_added": 0, "links_added": 0}

    for record in _iter_jsonl(shard_dir):
        if not record.get("paper_id") and not record.get("source_id"):
            continue
        paper = _paper_node(record)
        if _add_node(graph, paper):
            stats["papers_added"] += 1

        venue = str(record.get("venue") or "").strip()
        if venue:
            venue_node = _venue_node(venue, paper_id=str(record.get("paper_id") or ""))
            if _add_node(graph, venue_node):
                stats["venues_added"] += 1
            evidence = venue_node.evidence[0]
            _add_edge(
                graph,
                _edge(
                    paper.node_id,
                    "paper_published_in",
                    venue_node.node_id,
                    evidence=evidence,
                    confidence=0.95,
                ),
            )

        paper_record_id = paper.source_ids[0]
        for dataset_link in links_by_paper.get(paper_record_id, []):
            dataset_id = str(dataset_link["dataset_record_id"])
            dataset = _dataset_placeholder(dataset_id)
            _add_node(graph, dataset)
            evidence = _evidence(
                source_type="dataset_paper_link",
                source_id=f"{dataset_id}:{paper_record_id}",
                source_field=str(dataset_link.get("match_method") or "paper_link"),
                evidence_text=str(dataset_link.get("paper_title") or record.get("title") or ""),
                confidence=float(dataset_link.get("confidence") or 0.0),
            )
            if _add_edge(
                graph,
                _edge(
                    dataset.node_id,
                    "dataset_linked_to_paper",
                    paper.node_id,
                    evidence=evidence,
                    confidence=float(dataset_link.get("confidence") or 0.0),
                    properties={"match_method": dataset_link.get("match_method")},
                ),
            ):
                stats["links_added"] += 1

    graph.metadata["literature_paper_stats"] = stats
    return stats


def _region_crosswalk_properties(value: str) -> dict[str, Any]:
    """Resolve a free-text region string to atlas_refs via exact alias lookup.

    Cheap O(1) lookup against a cached index — safe to call per finding at
    full-corpus scale (unlike neural_search.ontology.matcher.match_brain_regions,
    which rebuilds its lookup table on every call).
    """
    canonical_id = get_region_id_by_alias(value)
    if canonical_id is None:
        return {}
    atlas_refs = get_region_atlas_refs(canonical_id)
    if not atlas_refs:
        return {}
    return {"canonical_region_id": canonical_id, "atlas_refs": atlas_refs}


def _task_crosswalk_properties(value: str) -> dict[str, Any]:
    """Resolve a free-text task string to a validated Cognitive Atlas match, if any."""
    canonical_id = get_task_id_by_alias(value)
    if canonical_id is None:
        return {}
    match = get_cogat_match(canonical_id)
    if match is None:
        return {}
    return {
        "canonical_task_id": canonical_id,
        "cogat_id": match.cogat_id,
        "cogat_label": match.cogat_label,
        "cogat_match_type": match.match_type,
    }


def _concept_node(
    node_type: str,
    value: str,
    *,
    source_id: str,
    confidence: float,
) -> KnowledgeGraphNode:
    evidence = _evidence(
        source_type="finding_extraction",
        source_id=source_id,
        source_field=node_type,
        evidence_text=value,
        confidence=confidence,
    )
    properties: dict[str, Any] = {}
    if node_type == "brain_region":
        properties.update(_region_crosswalk_properties(value))
    elif node_type == "task":
        properties.update(_task_crosswalk_properties(value))
    return KnowledgeGraphNode(
        node_id=make_node_id(node_type, normalize_node_type(value)),
        node_type=node_type,
        label=value,
        aliases=[value],
        source_ids=[source_id],
        properties=properties,
        evidence=[evidence],
        confidence=confidence,
        created_at=_now(),
    )


def _finding_node(record: dict[str, Any]) -> KnowledgeGraphNode:
    finding_id = str(record["finding_id"])
    text = str(record.get("finding_text") or finding_id)
    evidence = _evidence(
        source_type="finding_extraction",
        source_id=finding_id,
        source_field="finding_text",
        evidence_text=text,
        confidence=float(record.get("confidence") or 0.0),
        char_start=record.get("char_start"),
        char_end=record.get("char_end"),
        sentence_id=record.get("sentence_id"),
    )
    return KnowledgeGraphNode(
        node_id=make_node_id("finding", finding_id),
        node_type="finding",
        label=text[:160],
        aliases=[finding_id],
        source_ids=[finding_id],
        properties={
            "paper_id": record.get("paper_id"),
            "paper_doi": record.get("paper_doi"),
            "finding_text": text,
            "result_direction": record.get("result_direction"),
            "cell_types": record.get("cell_types", []),
            "molecules": record.get("molecules", []),
            "extraction_model": record.get("extraction_model"),
        },
        evidence=[evidence],
        confidence=float(record.get("confidence") or 0.0),
        created_at=str(record.get("extracted_at") or _now()),
    )


def add_findings_to_graph(
    graph: KnowledgeGraph,
    findings_path: Path,
) -> dict[str, int]:
    """Add finding nodes and finding-to-concept edges from extraction JSONL."""

    stats = {"findings_added": 0, "edges_added": 0}
    concept_fields = {
        "regions": ("brain_region", "finding_involves_region"),
        "tasks": ("task", "finding_involves_task"),
        "modalities": ("modality", "finding_involves_modality"),
        "species": ("species", "finding_involves_species"),
        # Typed fields from neural_search.literature.typed_finding_extractor.
        # Only these three are promoted to graph edges for now — the other 24
        # typed fields stay as finding-record properties until a dedicated
        # node/edge expansion (see docs/superpowers/plans/2026-06-22-typed-
        # field-coverage-relationship-expansion.md) is scoped and reviewed.
        "frequency_band": ("frequency_band", "finding_has_frequency_band"),
        "temporal_pattern": ("temporal_pattern", "finding_has_temporal_pattern"),
        "spatial_frame": ("spatial_frame", "finding_has_spatial_frame"),
    }

    for record in _iter_jsonl(findings_path):
        if not record.get("finding_id") or not record.get("paper_id"):
            continue
        finding = _finding_node(record)
        if _add_node(graph, finding):
            stats["findings_added"] += 1

        paper_id = str(record["paper_id"])
        paper = KnowledgeGraphNode(
            node_id=make_node_id("paper", *paper_id.split(":")[1:]),
            node_type="paper",
            label=paper_id,
            aliases=[paper_id],
            source_ids=[paper_id],
            properties={"placeholder": True},
            confidence=0.35,
            created_at=_now(),
        )
        _add_node(graph, paper)
        evidence = finding.evidence[0]
        if _add_edge(
            graph,
            _edge(
                paper.node_id,
                "paper_reports_finding",
                finding.node_id,
                evidence=evidence,
                confidence=finding.confidence,
            ),
        ):
            stats["edges_added"] += 1

        for field_name, (node_type, edge_type) in concept_fields.items():
            for value in record.get(field_name, []) or []:
                confidence = float(record.get("confidence") or 0.0)
                concept = _concept_node(
                    node_type,
                    str(value),
                    source_id=str(record["finding_id"]),
                    confidence=confidence,
                )
                _add_node(graph, concept)
                concept_evidence = concept.evidence[0]
                if _add_edge(
                    graph,
                    _edge(
                        finding.node_id,
                        edge_type,
                        concept.node_id,
                        evidence=concept_evidence,
                        confidence=confidence,
                    ),
                ):
                    stats["edges_added"] += 1

    graph.metadata["literature_finding_stats"] = stats
    return stats
