"""Memory graph builder — assembles the field-state memory graph from all sources.

Inputs consumed:
- normalized corpus (NormalizedDatasetRecord JSONL)
- neuro-judge evidence packets (optional)
- neuro-judge judgments (optional)
- feedback logs (optional)
- concept memory artifacts (optional)

Outputs:
- FieldStateGraphStore (in-memory, exportable to JSONL + manifest)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neural_search.field_state.graph_store import FieldStateGraphStore
from neural_search.field_state.normalizers import (
    extract_modalities_from_text,
    extract_raw_data_evidence,
    extract_regions_from_text,
    extract_species_from_text,
)
from neural_search.graph.schema import (
    GraphEvidence,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)

log = logging.getLogger(__name__)

BUILDER_NAME = "neural_search.field_state.memory_graph"
BUILDER_VERSION = "v0.9.0"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _content_hash(obj: Any) -> str:
    serialized = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def _builder_evidence(source_id: str, source_field: str, text: str | None = None, confidence: float = 1.0) -> GraphEvidence:
    eid = f"ev:{BUILDER_NAME}:{source_id}:{source_field}"
    return GraphEvidence(
        evidence_id=eid[:200],
        source_type="corpus_record",
        source_id=source_id,
        source_field=source_field,
        evidence_text=text,
        confidence=confidence,
        extractor_name=BUILDER_NAME,
        extractor_version=BUILDER_VERSION,
    )


class MemoryGraphBuilder:
    """Builds a FieldStateGraphStore from corpus and ancillary sources."""

    def __init__(self) -> None:
        self.store = FieldStateGraphStore()
        self._dataset_nodes: dict[str, str] = {}  # dataset_id -> node_id

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        corpus_path: Path | None = None,
        corpus_records: list[dict[str, Any]] | None = None,
        evidence_packets_path: Path | None = None,
        judgments_path: Path | None = None,
        feedback_path: Path | None = None,
        concept_memory_path: Path | None = None,
    ) -> FieldStateGraphStore:
        """Build the graph from all available inputs and return the store."""
        records = corpus_records or []
        if corpus_path and corpus_path.exists():
            records = list(_load_jsonl(corpus_path))

        log.info("Building memory graph from %d corpus records", len(records))
        for rec in records:
            self._add_dataset_record(rec)

        if evidence_packets_path and evidence_packets_path.exists():
            for pkt in _load_jsonl(evidence_packets_path):
                self._add_evidence_packet(pkt)

        if judgments_path and judgments_path.exists():
            for jmt in _load_jsonl(judgments_path):
                self._add_judgment(jmt)

        if feedback_path and feedback_path.exists():
            for fb in _load_jsonl(feedback_path):
                self._add_feedback(fb)

        if concept_memory_path and concept_memory_path.exists():
            for concept in _load_jsonl(concept_memory_path):
                self._add_concept(concept)

        log.info("Graph built: %s", self.store)
        return self.store

    # ------------------------------------------------------------------
    # Dataset record → nodes + edges
    # ------------------------------------------------------------------

    def _add_dataset_record(self, rec: dict[str, Any]) -> None:
        dataset_id: str = rec.get("dataset_id", "")
        if not dataset_id:
            return

        source: str = rec.get("source", "unknown")
        title: str = rec.get("title", dataset_id)
        description: str = rec.get("description") or ""

        # Source archive node
        archive_node_id = self._upsert_source_archive(source)

        # Dataset node
        content_hash = _content_hash({k: rec.get(k) for k in ("title", "description", "source_id")})
        ds_node = KnowledgeGraphNode(
            node_id=make_node_id("dataset", *dataset_id.split(":")[1:]) if ":" in dataset_id else make_node_id("dataset", dataset_id),
            node_type="dataset",
            label=title,
            source_ids=[dataset_id],
            properties={
                "dataset_id": dataset_id,
                "source": source,
                "source_id": rec.get("source_id", ""),
                "url": rec.get("url"),
                "description": description[:500] if description else None,
                "extractor_version": rec.get("extractor_version"),
                "content_hash": content_hash,
                "created_at": rec.get("created_at", _utc_now()),
            },
            evidence=[_builder_evidence(dataset_id, "corpus_record", title)],
            confidence=1.0,
        )
        self.store.upsert_node(ds_node)
        ds_node_id = ds_node.node_id
        self._dataset_nodes[dataset_id] = ds_node_id

        # dataset → source_archive
        self._upsert_edge(ds_node_id, "dataset_from_source", archive_node_id, confidence=1.0)

        # Linked papers
        for paper_id in rec.get("linked_papers", []):
            paper_node_id = self._upsert_paper_stub(paper_id)
            self._upsert_edge(ds_node_id, "dataset_linked_to_paper", paper_node_id, confidence=0.9)

        # Label-type nodes from structured metadata
        self._add_label_list_nodes(ds_node_id, rec, "modalities", "modality", "dataset_has_modality", dataset_id)
        self._add_label_list_nodes(ds_node_id, rec, "species", "species", "dataset_has_species", dataset_id)
        self._add_label_list_nodes(ds_node_id, rec, "brain_regions", "brain_region", "dataset_records_region", dataset_id)
        self._add_label_list_nodes(ds_node_id, rec, "tasks", "task", "dataset_has_task", dataset_id)
        self._add_label_list_nodes(ds_node_id, rec, "behavioral_events", "behavioral_event", "dataset_has_behavioral_event", dataset_id)
        self._add_label_list_nodes(ds_node_id, rec, "data_standards", "data_standard", "dataset_uses_standard", dataset_id)
        self._add_label_list_nodes(ds_node_id, rec, "file_formats", "file_format", "dataset_has_file_format", dataset_id)

        # Analysis affordances
        for affordance in rec.get("analysis_affordances", []):
            aff_id = affordance.get("analysis_id", "")
            if not aff_id:
                continue
            aff_node_id = self._upsert_concept_node("analysis_affordance", aff_id, aff_id)
            support = affordance.get("support_level", "unknown")
            confidence = affordance.get("confidence", 0.5)
            if support in ("high", "medium"):
                self._upsert_edge(ds_node_id, "dataset_supports_analysis", aff_node_id, confidence=confidence)
            elif support in ("low", "unsupported"):
                self._upsert_edge(ds_node_id, "dataset_lacks_required_evidence", aff_node_id, confidence=confidence)

        # Usability flags → raw/processed signal nodes
        flags: dict[str, Any] = rec.get("usability_flags", {})
        if flags.get("has_raw_data") is True:
            raw_node_id = self._upsert_concept_node("raw_data_signal", f"{dataset_id}_raw", "raw_data")
            self._upsert_edge(ds_node_id, "dataset_has_raw_signal", raw_node_id, confidence=0.8)
        elif flags.get("has_raw_data") is False:
            proc_node_id = self._upsert_concept_node("processed_data_signal", f"{dataset_id}_processed", "processed_only")
            self._upsert_edge(ds_node_id, "dataset_has_processed_signal", proc_node_id, confidence=0.8)

        # Fallback: text-based raw/processed inference (marked as inferred)
        if description and flags.get("has_raw_data") is None:
            raw_avail, raw_evidence = extract_raw_data_evidence(description)
            if raw_avail is True:
                raw_node_id = self._upsert_concept_node("raw_data_signal", f"{dataset_id}_raw_inferred", "raw_data_inferred")
                self._upsert_edge(
                    ds_node_id, "dataset_has_raw_signal", raw_node_id,
                    confidence=0.6,
                    props={"inferred": True, "evidence": raw_evidence[:3]},
                )

        # Text-based species/modality/region augmentation (inferred, lower confidence)
        if description:
            self._add_text_inferred_labels(ds_node_id, dataset_id, description)

    def _add_label_list_nodes(
        self,
        ds_node_id: str,
        rec: dict[str, Any],
        field: str,
        node_type: str,
        edge_type: str,
        dataset_id: str,
    ) -> None:
        for label_obj in rec.get(field, []):
            if isinstance(label_obj, dict):
                label_text = label_obj.get("label", "")
                confidence = label_obj.get("confidence", 0.8)
                evidence_text = label_obj.get("evidence_text")
            else:
                label_text = str(label_obj)
                confidence = 0.7
                evidence_text = None
            if not label_text:
                continue
            concept_id = label_text.lower().replace(" ", "_").replace("-", "_")
            concept_node_id = self._upsert_concept_node(node_type, concept_id, label_text)
            self._upsert_edge(
                ds_node_id, edge_type, concept_node_id,
                confidence=confidence,
                props={"evidence_text": evidence_text, "source_field": field},
            )

    def _add_text_inferred_labels(self, ds_node_id: str, dataset_id: str, text: str) -> None:
        for canonical, phrase in extract_species_from_text(text):
            sp_node_id = self._upsert_concept_node("species", canonical, canonical)
            self._upsert_edge(
                ds_node_id, "dataset_has_species", sp_node_id,
                confidence=0.55,
                props={"inferred": True, "matched_phrase": phrase},
            )
        for canonical, phrase in extract_modalities_from_text(text):
            mod_node_id = self._upsert_concept_node("modality", canonical, canonical)
            self._upsert_edge(
                ds_node_id, "dataset_has_modality", mod_node_id,
                confidence=0.55,
                props={"inferred": True, "matched_phrase": phrase},
            )
        for canonical, phrase in extract_regions_from_text(text):
            reg_node_id = self._upsert_concept_node("brain_region", canonical, canonical)
            self._upsert_edge(
                ds_node_id, "dataset_records_region", reg_node_id,
                confidence=0.55,
                props={"inferred": True, "matched_phrase": phrase},
            )

    # ------------------------------------------------------------------
    # Neuro-judge evidence packet → node
    # ------------------------------------------------------------------

    def _add_evidence_packet(self, pkt: dict[str, Any]) -> None:
        query_id = pkt.get("query_id", "")
        dataset_id = pkt.get("dataset_id", "")
        if not query_id or not dataset_id:
            return

        pkt_node_id = make_node_id("neuro_judge_evidence_packet", query_id, dataset_id)
        pkt_node = KnowledgeGraphNode(
            node_id=pkt_node_id,
            node_type="neuro_judge_evidence_packet",
            label=f"EvidencePacket({query_id}, {dataset_id})",
            source_ids=[query_id, dataset_id],
            properties={
                "query_id": query_id,
                "dataset_id": dataset_id,
                "schema_version": pkt.get("schema_version", ""),
                "has_raw_data": pkt.get("has_raw_data"),
                "has_processed_data": pkt.get("has_processed_data"),
                "provenance": "neuro_judge_evidence_packet",
            },
            confidence=0.9,
        )
        self.store.upsert_node(pkt_node)

        # Link to dataset node if present
        ds_node = self.store.query_by_dataset_id(dataset_id)
        if ds_node:
            self._upsert_edge(pkt_node_id, "judgment_labels_query_dataset", ds_node.node_id, confidence=0.9)

    # ------------------------------------------------------------------
    # Neuro-judge judgment → node
    # ------------------------------------------------------------------

    def _add_judgment(self, jmt: dict[str, Any]) -> None:
        query_id = jmt.get("query_id", "")
        dataset_id = jmt.get("dataset_id", "")
        if not query_id or not dataset_id:
            return

        # Hard guardrail: never store as human_gold
        provenance = jmt.get("label_provenance", "neuro_judge_silver")
        if provenance == "human_gold":
            log.warning("Judgment %s/%s has human_gold provenance — skipping", query_id, dataset_id)
            return

        jmt_node_id = make_node_id("neuro_judge_judgment", query_id, dataset_id)
        jmt_node = KnowledgeGraphNode(
            node_id=jmt_node_id,
            node_type="neuro_judge_judgment",
            label=f"Judgment({query_id}, {dataset_id})",
            source_ids=[query_id, dataset_id],
            properties={
                "query_id": query_id,
                "dataset_id": dataset_id,
                "label": jmt.get("label"),
                "confidence": jmt.get("confidence"),
                "label_provenance": provenance,
                "abstain_recommended": jmt.get("abstain_recommended"),
                "evidence_completeness": jmt.get("evidence_completeness"),
                "provenance_note": "neuro_judge_silver_not_human_gold",
            },
            confidence=float(jmt.get("confidence", 0.5)),
        )
        self.store.upsert_node(jmt_node)

        ds_node = self.store.query_by_dataset_id(dataset_id)
        if ds_node:
            self._upsert_edge(jmt_node_id, "judgment_labels_query_dataset", ds_node.node_id, confidence=0.8)

    # ------------------------------------------------------------------
    # Feedback signal → node
    # ------------------------------------------------------------------

    def _add_feedback(self, fb: dict[str, Any]) -> None:
        feedback_id = fb.get("feedback_id", "")
        dataset_id = fb.get("dataset_id", "")
        if not feedback_id or not dataset_id:
            return

        # Feedback is downstream signal, never gold
        fb_node_id = make_node_id("feedback_signal", feedback_id)
        fb_node = KnowledgeGraphNode(
            node_id=fb_node_id,
            node_type="feedback_signal",
            label=f"Feedback({feedback_id})",
            source_ids=[feedback_id, dataset_id],
            properties={
                "feedback_id": feedback_id,
                "dataset_id": dataset_id,
                "usefulness": fb.get("usefulness"),
                "would_use_for_analysis": fb.get("would_use_for_analysis"),
                "reason_tags": fb.get("reason_tags", []),
                "provenance": "user_feedback_downstream_signal",
                "timestamp": fb.get("timestamp"),
            },
            confidence=0.7,
        )
        self.store.upsert_node(fb_node)

        ds_node = self.store.query_by_dataset_id(dataset_id)
        if ds_node:
            self._upsert_edge(fb_node_id, "feedback_marks_result", ds_node.node_id, confidence=0.7)

    # ------------------------------------------------------------------
    # Concept memory → concept nodes + edges
    # ------------------------------------------------------------------

    def _add_concept(self, concept: dict[str, Any]) -> None:
        concept_id = concept.get("concept_id", "")
        canonical_name = concept.get("canonical_name", "")
        concept_type = concept.get("concept_type", "concept")
        if not concept_id or not canonical_name:
            return

        node_type = concept_type if concept_type in {
            "modality", "species", "brain_region", "task", "analysis_affordance"
        } else "concept"

        c_node_id = make_node_id("concept", concept_id)
        c_node = KnowledgeGraphNode(
            node_id=c_node_id,
            node_type=node_type,
            label=canonical_name,
            aliases=concept.get("aliases", []),
            source_ids=concept.get("source_ids", []),
            properties={
                "concept_id": concept_id,
                "concept_type": concept_type,
                "description": concept.get("description"),
                "confidence": concept.get("confidence", 0.5),
                "review_status": concept.get("review_status", "unreviewed"),
                "tags": concept.get("tags", []),
            },
            confidence=float(concept.get("confidence", 0.5)),
        )
        self.store.upsert_node(c_node)

    # ------------------------------------------------------------------
    # Helper: source archive node
    # ------------------------------------------------------------------

    def _upsert_source_archive(self, source: str) -> str:
        node_id = make_node_id("source_archive", source)
        node = KnowledgeGraphNode(
            node_id=node_id,
            node_type="source_archive",
            label=source,
            properties={"source": source},
            confidence=1.0,
        )
        self.store.upsert_node(node)
        return node_id

    def _upsert_paper_stub(self, paper_id: str) -> str:
        parts = paper_id.split(":")[1:] if ":" in paper_id else [paper_id]
        node_id = make_node_id("paper", *parts)
        node = KnowledgeGraphNode(
            node_id=node_id,
            node_type="paper",
            label=paper_id,
            source_ids=[paper_id],
            properties={"paper_id": paper_id},
            confidence=0.8,
        )
        self.store.upsert_node(node)
        return node_id

    def _upsert_concept_node(self, node_type: str, concept_id: str, label: str) -> str:
        node_id = make_node_id(node_type, concept_id)
        node = KnowledgeGraphNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            properties={"concept_id": concept_id},
            confidence=0.9,
        )
        self.store.upsert_node(node)
        return node_id

    def _upsert_edge(
        self,
        source_id: str,
        edge_type: str,
        target_id: str,
        confidence: float = 1.0,
        props: dict[str, Any] | None = None,
    ) -> str:
        edge_id = make_edge_id(source_id, edge_type, target_id)
        edge = KnowledgeGraphEdge(
            edge_id=edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            edge_type=edge_type,
            confidence=confidence,
            properties=props or {},
        )
        self.store.upsert_edge(edge)
        return edge_id


# ---------------------------------------------------------------------------
# JSONL loading helper
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed JSONL line in %s: %s", path, exc)
    return records
