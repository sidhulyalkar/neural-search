"""Tests for FieldStateGraphStore invariant validation."""

from __future__ import annotations

import pytest

from neural_search.field_state.graph_store import FieldStateGraphStore
from neural_search.field_state.memory_graph import MemoryGraphBuilder
from neural_search.graph.schema import (
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)


def _make_dataset_store() -> FieldStateGraphStore:
    """Build a minimal valid store with one dataset."""
    builder = MemoryGraphBuilder()
    store = builder.build(corpus_records=[{
        "dataset_id": "dataset:dandi:000001",
        "source": "dandi",
        "source_id": "000001",
        "title": "Test Dataset",
        "description": "Mouse ephys",
        "modalities": [{"id": "m1", "label": "neuropixels", "label_type": "modality", "confidence": 0.9}],
        "species": [{"id": "s1", "label": "mouse", "label_type": "species", "confidence": 0.9}],
        "brain_regions": [],
        "tasks": [],
        "data_standards": [],
        "linked_papers": [],
        "usability_flags": {"has_raw_data": True},
        "analysis_affordances": [],
        "behavioral_events": [],
        "file_formats": [],
    }])
    return store


class TestGraphStoreInvariants:
    def test_valid_graph_passes(self) -> None:
        store = _make_dataset_store()
        errors = store.validate_invariants()
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_edge_with_missing_target_fails(self) -> None:
        store = FieldStateGraphStore()
        ds_id = make_node_id("dataset", "dandi", "000001")
        ds_node = KnowledgeGraphNode(node_id=ds_id, node_type="dataset", label="Test", properties={"source": "dandi"})
        store.upsert_node(ds_node)

        # Add edge pointing to non-existent target
        edge = KnowledgeGraphEdge(
            edge_id="edge:dataset:dandi:000001:from_source:missing",
            source_node_id=ds_id,
            target_node_id="node:source_archive:missing",
            edge_type="dataset_from_source",
        )
        store._edges[edge.edge_id] = edge  # bypass upsert to force bad state
        errors = store.validate_invariants()
        assert any("does not resolve" in e or "not in nodes" in e for e in errors)

    def test_dataset_without_source_fails(self) -> None:
        store = FieldStateGraphStore()
        ds_id = make_node_id("dataset", "dandi", "999")
        ds_node = KnowledgeGraphNode(
            node_id=ds_id,
            node_type="dataset",
            label="Orphan Dataset",
            # No source property and no dataset_from_source edge
        )
        store.upsert_node(ds_node)
        errors = store.validate_invariants()
        assert any("source" in e for e in errors)


class TestGraphStoreUpsert:
    def test_upsert_node_merges_aliases(self) -> None:
        store = FieldStateGraphStore()
        nid = make_node_id("source_archive", "dandi")
        n1 = KnowledgeGraphNode(node_id=nid, node_type="source_archive", label="DANDI", aliases=["dandi_archive"])
        n2 = KnowledgeGraphNode(node_id=nid, node_type="source_archive", label="DANDI v2", aliases=["dandi_v2"])
        store.upsert_node(n1)
        store.upsert_node(n2)
        merged = store.get_node(nid)
        assert "dandi_archive" in merged.aliases
        assert "dandi_v2" in merged.aliases

    def test_upsert_edge_merges_confidence(self) -> None:
        store = FieldStateGraphStore()
        src = make_node_id("dataset", "dandi", "1")
        tgt = make_node_id("source_archive", "dandi")
        src_node = KnowledgeGraphNode(node_id=src, node_type="dataset", label="DS", properties={"source": "dandi"})
        tgt_node = KnowledgeGraphNode(node_id=tgt, node_type="source_archive", label="DANDI")
        store.upsert_node(src_node)
        store.upsert_node(tgt_node)

        eid = make_edge_id(src, "dataset_from_source", tgt)
        e1 = KnowledgeGraphEdge(edge_id=eid, source_node_id=src, target_node_id=tgt, edge_type="dataset_from_source", confidence=0.5)
        e2 = KnowledgeGraphEdge(edge_id=eid, source_node_id=src, target_node_id=tgt, edge_type="dataset_from_source", confidence=0.9)
        store.upsert_edge(e1)
        store.upsert_edge(e2)
        merged = store.get_edge(eid)
        assert merged.confidence == 0.9  # max of 0.5 and 0.9


class TestGraphStoreQueries:
    def test_query_datasets_returns_all_datasets(self) -> None:
        store = _make_dataset_store()
        datasets = store.query_datasets()
        assert len(datasets) == 1
        assert all(d.node_type == "dataset" for d in datasets)

    def test_query_by_type_source_archive(self) -> None:
        store = _make_dataset_store()
        archives = store.query_by_type("source_archive")
        assert len(archives) == 1

    def test_query_by_dataset_id(self) -> None:
        store = _make_dataset_store()
        node = store.query_by_dataset_id("dataset:dandi:000001")
        assert node is not None
        assert node.node_type == "dataset"

    def test_query_by_dataset_id_missing(self) -> None:
        store = _make_dataset_store()
        node = store.query_by_dataset_id("dataset:nonexistent:999")
        assert node is None

    def test_query_datasets_missing_evidence(self) -> None:
        store = FieldStateGraphStore()
        ds_id = make_node_id("dataset", "dandi", "orphan")
        archive_id = make_node_id("source_archive", "dandi")
        aff_id = make_node_id("analysis_affordance", "spike_sorting")
        ds = KnowledgeGraphNode(node_id=ds_id, node_type="dataset", label="Orphan", properties={"source": "dandi"})
        arch = KnowledgeGraphNode(node_id=archive_id, node_type="source_archive", label="DANDI")
        aff = KnowledgeGraphNode(node_id=aff_id, node_type="analysis_affordance", label="spike_sorting")
        store.upsert_node(ds)
        store.upsert_node(arch)
        store.upsert_node(aff)

        edge = KnowledgeGraphEdge(
            edge_id=make_edge_id(ds_id, "dataset_lacks_required_evidence", aff_id),
            source_node_id=ds_id,
            target_node_id=aff_id,
            edge_type="dataset_lacks_required_evidence",
        )
        store.upsert_edge(edge)
        missing = store.query_datasets_missing_evidence()
        assert ds_id in [n.node_id for n in missing]


class TestGraphStoreExportImport:
    def test_round_trip_jsonl(self, tmp_path) -> None:
        store = _make_dataset_store()
        nodes_path = tmp_path / "nodes.jsonl"
        edges_path = tmp_path / "edges.jsonl"
        store.export_jsonl(nodes_path, edges_path)

        loaded = FieldStateGraphStore.from_jsonl(nodes_path, edges_path)
        assert loaded.node_count == store.node_count
        assert loaded.edge_count == store.edge_count

    def test_manifest_written(self, tmp_path) -> None:
        store = _make_dataset_store()
        manifest_path = tmp_path / "manifest.json"
        manifest = store.write_manifest(manifest_path, build_id="test_build")
        assert manifest_path.exists()
        assert manifest["total_nodes"] == store.node_count
        assert manifest["build_id"] == "test_build"
