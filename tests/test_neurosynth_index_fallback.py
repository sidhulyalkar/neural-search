"""Regression test for the 2026-07-02 fragility fix: `_load_neurosynth_index()`
must not silently return {} when `artifacts/kg/composed_kg.jsonl` is absent.
That file is gitignored and has no automated regeneration pipeline, so its
absence previously caused the neurosynth score to silently drop to zero. See
`reports/architecture_connectivity_audit_2026-07-01.md` for the audit that
found this.
"""

from __future__ import annotations

from neural_search.graph import search_features
from neural_search.graph.schema import KnowledgeGraph, KnowledgeGraphEdge, KnowledgeGraphNode


def test_falls_back_to_building_from_source_when_artifact_missing(monkeypatch, tmp_path):
    search_features._load_neurosynth_index.cache_clear()
    missing_path = tmp_path / "does_not_exist.jsonl"
    monkeypatch.setattr(search_features, "_COMPOSED_KG_PATH", missing_path)

    fake_kg = KnowledgeGraph(
        nodes={
            "concept:001": KnowledgeGraphNode(node_id="concept:001", node_type="concept", label="001"),
            "ontology_region:visual_cortex": KnowledgeGraphNode(
                node_id="ontology_region:visual_cortex", node_type="ontology_region", label="visual_cortex"
            ),
        },
        edges={
            "edge:1": KnowledgeGraphEdge(
                edge_id="edge:1",
                edge_type="topic_activates_region",
                source_node_id="concept:001",
                target_node_id="ontology_region:visual_cortex",
                directed=True,
                confidence=0.8,
            )
        },
    )
    monkeypatch.setattr(
        "neural_search.ingestion.neurosynth_builder.build_neurosynth_kg", lambda: fake_kg
    )

    try:
        index = search_features._load_neurosynth_index()
    finally:
        search_features._load_neurosynth_index.cache_clear()

    assert index == {"visual_cortex": [0.8]}


def test_returns_empty_when_artifact_missing_and_source_build_fails(monkeypatch, tmp_path):
    search_features._load_neurosynth_index.cache_clear()
    missing_path = tmp_path / "does_not_exist.jsonl"
    monkeypatch.setattr(search_features, "_COMPOSED_KG_PATH", missing_path)

    def _raise():
        raise FileNotFoundError("no neurosynth source data")

    monkeypatch.setattr(
        "neural_search.ingestion.neurosynth_builder.build_neurosynth_kg", _raise
    )

    try:
        index = search_features._load_neurosynth_index()
    finally:
        search_features._load_neurosynth_index.cache_clear()

    assert index == {}
