"""Tests for the extended graph schema node/edge type sets."""

from __future__ import annotations

import pytest

from neural_search.graph.schema import (
    SUPPORTED_EDGE_TYPES,
    SUPPORTED_NODE_TYPES,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)


class TestNewNodeTypes:
    """The field-state memory graph node types must be present."""

    required_node_types = [
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
    ]

    @pytest.mark.parametrize("node_type", required_node_types)
    def test_node_type_in_supported_set(self, node_type: str) -> None:
        assert node_type in SUPPORTED_NODE_TYPES, (
            f"Node type '{node_type}' missing from SUPPORTED_NODE_TYPES"
        )


class TestNewEdgeTypes:
    """The field-state memory graph edge types must be present."""

    required_edge_types = [
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
    ]

    @pytest.mark.parametrize("edge_type", required_edge_types)
    def test_edge_type_in_supported_set(self, edge_type: str) -> None:
        assert edge_type in SUPPORTED_EDGE_TYPES, (
            f"Edge type '{edge_type}' missing from SUPPORTED_EDGE_TYPES"
        )


class TestMakeNodeId:
    def test_source_archive_id(self) -> None:
        nid = make_node_id("source_archive", "dandi")
        assert nid == "node:source_archive:dandi"

    def test_feedback_signal_id(self) -> None:
        nid = make_node_id("feedback_signal", "fb_abc123")
        assert "feedback_signal" in nid

    def test_neuro_judge_judgment_id(self) -> None:
        nid = make_node_id("neuro_judge_judgment", "q1", "dataset:dandi:000026")
        assert "neuro_judge_judgment" in nid


class TestNodeConstruction:
    def test_source_archive_node(self) -> None:
        node = KnowledgeGraphNode(
            node_id=make_node_id("source_archive", "dandi"),
            node_type="source_archive",
            label="DANDI Archive",
        )
        assert node.node_type == "source_archive"

    def test_feedback_signal_node(self) -> None:
        node = KnowledgeGraphNode(
            node_id=make_node_id("feedback_signal", "fb1"),
            node_type="feedback_signal",
            label="Feedback fb1",
            properties={"provenance": "user_feedback_downstream_signal"},
        )
        assert node.properties["provenance"] == "user_feedback_downstream_signal"

    def test_neuro_judge_judgment_node(self) -> None:
        node = KnowledgeGraphNode(
            node_id=make_node_id("neuro_judge_judgment", "q1", "ds1"),
            node_type="neuro_judge_judgment",
            label="Judgment(q1, ds1)",
            properties={"label_provenance": "neuro_judge_silver"},
            confidence=0.7,
        )
        assert node.properties["label_provenance"] != "human_gold"

    def test_snapshot_manifest_node(self) -> None:
        node = KnowledgeGraphNode(
            node_id=make_node_id("snapshot_manifest", "20260612T000000Z"),
            node_type="snapshot_manifest",
            label="Snapshot 20260612T000000Z",
        )
        assert node.node_type == "snapshot_manifest"


class TestEdgeConstruction:
    def test_dataset_from_source_edge(self) -> None:
        src = make_node_id("dataset", "dandi", "000026")
        tgt = make_node_id("source_archive", "dandi")
        eid = make_edge_id(src, "dataset_from_source", tgt)
        edge = KnowledgeGraphEdge(
            edge_id=eid,
            source_node_id=src,
            target_node_id=tgt,
            edge_type="dataset_from_source",
        )
        assert edge.edge_type == "dataset_from_source"

    def test_judgment_labels_edge(self) -> None:
        jmt = make_node_id("neuro_judge_judgment", "q1", "ds1")
        ds = make_node_id("dataset", "dandi", "000026")
        eid = make_edge_id(jmt, "judgment_labels_query_dataset", ds)
        edge = KnowledgeGraphEdge(
            edge_id=eid,
            source_node_id=jmt,
            target_node_id=ds,
            edge_type="judgment_labels_query_dataset",
            confidence=0.8,
        )
        assert edge.confidence == 0.8

    def test_feedback_marks_result_edge(self) -> None:
        fb = make_node_id("feedback_signal", "fb1")
        ds = make_node_id("dataset", "dandi", "000026")
        eid = make_edge_id(fb, "feedback_marks_result", ds)
        edge = KnowledgeGraphEdge(
            edge_id=eid,
            source_node_id=fb,
            target_node_id=ds,
            edge_type="feedback_marks_result",
        )
        assert edge.edge_type == "feedback_marks_result"
