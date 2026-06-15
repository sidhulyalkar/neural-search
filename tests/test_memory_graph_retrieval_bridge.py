"""Tests for neural_search.field_state.retrieval_bridge."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from neural_search.field_state.graph_store import FieldStateGraphStore
from neural_search.field_state.retrieval_bridge import (
    compute_memory_graph_evidence,
    compute_memory_graph_score,
    load_memory_graph_store,
)
from neural_search.graph.schema import (
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    make_edge_id,
    make_node_id,
)

# ---------------------------------------------------------------------------
# Helpers to build minimal mock graph stores
# ---------------------------------------------------------------------------

def _make_node(node_type: str, label: str, *id_parts: str, aliases: list[str] | None = None) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id(node_type, *id_parts),
        node_type=node_type,
        label=label,
        aliases=aliases or [],
    )


def _make_edge(source_id: str, edge_type: str, target_id: str, confidence: float = 1.0) -> KnowledgeGraphEdge:
    return KnowledgeGraphEdge(
        edge_id=make_edge_id(source_id, edge_type, target_id),
        source_node_id=source_id,
        target_node_id=target_id,
        edge_type=edge_type,
        confidence=confidence,
    )


def _build_store_with_dataset(
    dataset_id: str = "dandi:000001",
    *,
    modalities: list[str] | None = None,
    species: list[str] | None = None,
    regions: list[str] | None = None,
    affordances: list[str] | None = None,
    has_raw_signal: bool = False,
    lacks_evidence_count: int = 0,
    contraindicated_labels: list[str] | None = None,
) -> tuple[FieldStateGraphStore, str]:
    """Build a minimal store with one dataset node and optional neighbors."""
    store = FieldStateGraphStore()

    ds_node_id = make_node_id("dataset", dataset_id)
    ds_node = KnowledgeGraphNode(
        node_id=ds_node_id,
        node_type="dataset",
        label=dataset_id,
        properties={"dataset_id": dataset_id, "source": "dandi"},
    )
    store.upsert_node(ds_node)

    # Modality nodes + edges
    for m in (modalities or []):
        m_node = _make_node("modality", m, m.lower().replace(" ", "_"))
        store.upsert_node(m_node)
        store.upsert_edge(_make_edge(ds_node_id, "dataset_has_modality", m_node.node_id))

    # Species nodes + edges
    for sp in (species or []):
        sp_node = _make_node("species", sp, sp.lower().replace(" ", "_"))
        store.upsert_node(sp_node)
        store.upsert_edge(_make_edge(ds_node_id, "dataset_has_species", sp_node.node_id))

    # Region nodes + edges
    for r in (regions or []):
        r_node = _make_node("brain_region", r, r.lower().replace(" ", "_"))
        store.upsert_node(r_node)
        store.upsert_edge(_make_edge(ds_node_id, "dataset_records_region", r_node.node_id))

    # Affordance nodes + edges
    for a in (affordances or []):
        a_node = _make_node("analysis_affordance", a, a.lower().replace(" ", "_"))
        store.upsert_node(a_node)
        store.upsert_edge(_make_edge(ds_node_id, "dataset_supports_analysis", a_node.node_id))

    # Raw signal edge
    if has_raw_signal:
        raw_node = _make_node("raw_data_signal", "raw_ephys", "raw_ephys")
        store.upsert_node(raw_node)
        store.upsert_edge(_make_edge(ds_node_id, "dataset_has_raw_signal", raw_node.node_id))

    # Lacks-evidence edges
    for i in range(lacks_evidence_count):
        gap_node = _make_node("concept", f"gap_{i}", f"gap_{i}")
        store.upsert_node(gap_node)
        store.upsert_edge(_make_edge(ds_node_id, "dataset_lacks_required_evidence", gap_node.node_id))

    # Contraindicated nodes
    for c in (contraindicated_labels or []):
        c_node = _make_node("concept", c, c.lower().replace(" ", "_"))
        store.upsert_node(c_node)
        store.upsert_edge(_make_edge(ds_node_id, "dataset_contraindicated_for", c_node.node_id))

    return store, dataset_id


def _result(dataset_id: str) -> Any:
    r = MagicMock()
    r.dataset_id = dataset_id
    r.score_breakdown = {}
    return r


# ---------------------------------------------------------------------------
# compute_memory_graph_score: positive signals
# ---------------------------------------------------------------------------

class TestPositiveSignals:
    def test_modality_match_adds_score(self) -> None:
        store, did = _build_store_with_dataset(modalities=["neuropixels"])
        result = _result(did)
        parsed = {"modalities": ["neuropixels"]}
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(0.10)

    def test_multiple_modality_matches_accumulate(self) -> None:
        store, did = _build_store_with_dataset(modalities=["neuropixels", "calcium_imaging"])
        result = _result(did)
        parsed = {"modalities": ["neuropixels", "calcium_imaging"]}
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(0.20)

    def test_species_match_adds_score(self) -> None:
        store, did = _build_store_with_dataset(species=["mouse"])
        result = _result(did)
        parsed = {"species": ["mouse"]}
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(0.10)

    def test_brain_region_match_adds_score(self) -> None:
        store, did = _build_store_with_dataset(regions=["hippocampus"])
        result = _result(did)
        parsed = {"brain_regions": ["hippocampus"]}
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(0.05)

    def test_affordance_match_adds_score(self) -> None:
        store, did = _build_store_with_dataset(affordances=["spike_sorting"])
        result = _result(did)
        parsed = {"affordances": ["spike_sorting"]}
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(0.08)

    def test_raw_signal_bonus_when_query_contains_raw(self) -> None:
        store, did = _build_store_with_dataset(has_raw_signal=True)
        result = _result(did)
        parsed = {"modalities": ["raw"], "brain_regions": []}
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(0.06)

    def test_no_raw_signal_bonus_when_no_raw_edge(self) -> None:
        store, did = _build_store_with_dataset(has_raw_signal=False)
        result = _result(did)
        parsed = {"modalities": ["raw"]}
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(0.0)

    def test_combined_positive_signals(self) -> None:
        store, did = _build_store_with_dataset(
            modalities=["neuropixels"],
            species=["mouse"],
            regions=["v1"],
            affordances=["spike_sorting"],
        )
        result = _result(did)
        parsed = {
            "modalities": ["neuropixels"],
            "species": ["mouse"],
            "brain_regions": ["v1"],
            "affordances": ["spike_sorting"],
        }
        score = compute_memory_graph_score(store, result, parsed)
        # 0.10 + 0.10 + 0.05 + 0.08 = 0.33
        assert score == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# compute_memory_graph_score: penalty signals
# ---------------------------------------------------------------------------

class TestPenaltySignals:
    def test_single_lacks_evidence_penalty(self) -> None:
        store, did = _build_store_with_dataset(lacks_evidence_count=1)
        result = _result(did)
        parsed: dict = {}
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(-0.04)

    def test_lacks_evidence_capped_at_three(self) -> None:
        store, did = _build_store_with_dataset(lacks_evidence_count=5)
        result = _result(did)
        parsed: dict = {}
        score = compute_memory_graph_score(store, result, parsed)
        # max penalty is 0.12, clamped from 0.20 to [-0.20, 0.40]
        assert score == pytest.approx(-0.12)

    def test_contraindicated_penalty_when_relevant(self) -> None:
        store, did = _build_store_with_dataset(contraindicated_labels=["spike_sorting"])
        result = _result(did)
        parsed = {"affordances": ["spike_sorting"]}
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(-0.10)

    def test_contraindicated_no_penalty_when_not_relevant(self) -> None:
        store, did = _build_store_with_dataset(contraindicated_labels=["spike_sorting"])
        result = _result(did)
        # query asks for something unrelated
        parsed = {"affordances": ["calcium_imaging"]}
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_memory_graph_score: clamping and edge cases
# ---------------------------------------------------------------------------

class TestClampingAndEdgeCases:
    def test_score_clamped_to_max(self) -> None:
        # Build a dataset that should produce a very high score
        store, did = _build_store_with_dataset(
            modalities=["m1", "m2", "m3", "m4"],
            species=["s1", "s2", "s3", "s4"],
            has_raw_signal=True,
        )
        result = _result(did)
        parsed = {
            "modalities": ["m1", "m2", "m3", "m4", "raw"],
            "species": ["s1", "s2", "s3", "s4"],
        }
        score = compute_memory_graph_score(store, result, parsed)
        assert score <= 0.40

    def test_score_clamped_to_min(self) -> None:
        store, did = _build_store_with_dataset(
            lacks_evidence_count=10,
            contraindicated_labels=["x1", "x2"],
        )
        result = _result(did)
        parsed = {"affordances": ["x1", "x2"]}
        score = compute_memory_graph_score(store, result, parsed)
        assert score >= -0.20

    def test_dataset_not_in_store_returns_zero(self) -> None:
        store, _ = _build_store_with_dataset(dataset_id="dandi:000001")
        result = _result("dandi:999999")  # not present
        parsed = {"modalities": ["neuropixels"]}
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(0.0)

    def test_empty_parsed_query_returns_zero_for_clean_dataset(self) -> None:
        store, did = _build_store_with_dataset()
        result = _result(did)
        score = compute_memory_graph_score(store, result, {})
        assert score == pytest.approx(0.0)

    def test_no_partial_query_match_returns_zero(self) -> None:
        store, did = _build_store_with_dataset(modalities=["neuropixels"])
        result = _result(did)
        parsed = {"modalities": ["calcium_imaging"]}  # present in store but not in query
        score = compute_memory_graph_score(store, result, parsed)
        assert score == pytest.approx(0.0)

    def test_missing_dataset_id_returns_zero(self) -> None:
        store, _ = _build_store_with_dataset()
        result = _result("")
        score = compute_memory_graph_score(store, result, {"modalities": ["neuropixels"]})
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# load_memory_graph_store
# ---------------------------------------------------------------------------

class TestLoadMemoryGraphStore:
    def test_returns_none_when_files_missing(self, tmp_path: Path) -> None:
        store = load_memory_graph_store(
            str(tmp_path / "nodes.jsonl"),
            str(tmp_path / "edges.jsonl"),
        )
        # Cache key will be unique per tmp_path, safe to call directly
        assert store is None

    def test_loads_store_from_valid_jsonl(self, tmp_path: Path) -> None:
        # Write a minimal valid nodes/edges JSONL pair
        src_store, did = _build_store_with_dataset(modalities=["neuropixels"])
        nodes_path = tmp_path / "nodes.jsonl"
        edges_path = tmp_path / "edges.jsonl"
        src_store.export_jsonl(nodes_path, edges_path)

        # Clear the lru_cache so this tmp_path is a fresh lookup
        load_memory_graph_store.cache_clear()
        loaded = load_memory_graph_store(str(nodes_path), str(edges_path))
        assert loaded is not None
        assert loaded.node_count == src_store.node_count
        assert loaded.edge_count == src_store.edge_count
        load_memory_graph_store.cache_clear()


# ---------------------------------------------------------------------------
# compute_memory_graph_evidence
# ---------------------------------------------------------------------------

class TestMemoryGraphEvidence:
    def test_returns_empty_for_unknown_dataset(self) -> None:
        store, _ = _build_store_with_dataset(dataset_id="dandi:000001")
        result = _result("dandi:999999")
        ev = compute_memory_graph_evidence(store, result, {"modalities": ["neuropixels"]})
        assert ev["modality_matches"] == []
        assert ev["species_matches"] == []
        assert ev["has_raw_signal"] is False
        assert ev["lacks_evidence_count"] == 0
        assert ev["contraindicated"] == []

    def test_modality_match_returned_in_evidence(self) -> None:
        store, did = _build_store_with_dataset(modalities=["neuropixels"])
        result = _result(did)
        ev = compute_memory_graph_evidence(store, result, {"modalities": ["neuropixels"]})
        assert "neuropixels" in ev["modality_matches"]

    def test_species_match_returned_in_evidence(self) -> None:
        store, did = _build_store_with_dataset(species=["mouse"])
        result = _result(did)
        ev = compute_memory_graph_evidence(store, result, {"species": ["mouse"]})
        assert "mouse" in ev["species_matches"]

    def test_region_match_returned_in_evidence(self) -> None:
        store, did = _build_store_with_dataset(regions=["hippocampus"])
        result = _result(did)
        ev = compute_memory_graph_evidence(store, result, {"brain_regions": ["hippocampus"]})
        assert "hippocampus" in ev["region_matches"]

    def test_affordance_match_returned_in_evidence(self) -> None:
        store, did = _build_store_with_dataset(affordances=["q_learning_modeling"])
        result = _result(did)
        ev = compute_memory_graph_evidence(store, result, {"affordances": ["q_learning_modeling"]})
        assert "q_learning_modeling" in ev["affordance_matches"]

    def test_raw_signal_flag_set(self) -> None:
        store, did = _build_store_with_dataset(has_raw_signal=True)
        result = _result(did)
        ev = compute_memory_graph_evidence(store, result, {"_raw_query": "I need raw spike data"})
        assert ev["has_raw_signal"] is True

    def test_lacks_evidence_count_returned(self) -> None:
        store, did = _build_store_with_dataset(lacks_evidence_count=3)
        result = _result(did)
        ev = compute_memory_graph_evidence(store, result, {})
        assert ev["lacks_evidence_count"] == 3

    def test_contraindicated_label_returned(self) -> None:
        store, did = _build_store_with_dataset(contraindicated_labels=["spike_sorting"])
        result = _result(did)
        ev = compute_memory_graph_evidence(store, result, {"affordances": ["spike_sorting"]})
        assert "spike_sorting" in ev["contraindicated"]

    def test_no_contraindicated_when_not_in_query(self) -> None:
        store, did = _build_store_with_dataset(contraindicated_labels=["spike_sorting"])
        result = _result(did)
        ev = compute_memory_graph_evidence(store, result, {"affordances": ["calcium_imaging"]})
        assert ev["contraindicated"] == []

    def test_empty_query_returns_all_defaults(self) -> None:
        store, did = _build_store_with_dataset(modalities=["fmri"], species=["human"])
        result = _result(did)
        ev = compute_memory_graph_evidence(store, result, {})
        assert ev["modality_matches"] == []  # no query modalities
        assert ev["species_matches"] == []   # no query species
        assert ev["has_raw_signal"] is False
        assert ev["lacks_evidence_count"] == 0
