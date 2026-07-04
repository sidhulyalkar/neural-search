"""Tests for scripts/build_artifact_manifest.py.

This script replaces the previously hand-maintained
reports/eval/current_artifact_manifest.json (which drifted stale — see the
module docstring). Tests use small fixture files so they don't depend on the
real multi-hundred-MB production graph.
"""

from __future__ import annotations

import json

import scripts.build_artifact_manifest as bam


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def test_corpus_section_counts_rows_and_unique_ids(tmp_path, monkeypatch):
    corpus_path = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus_path,
        [
            {"source": "dandi", "source_id": "000001", "dataset_id": "dataset:dandi:000001"},
            {"source": "dandi", "source_id": "000002", "dataset_id": "dataset:dandi:000002"},
            {"source": "dandi", "source_id": "000001"},  # duplicate source_id, no dataset_id
        ],
    )
    monkeypatch.setattr(bam, "CORPUS_PATH", corpus_path)
    section = bam._corpus_section()
    assert section["available"] is True
    assert section["row_count"] == 3
    assert section["unique_source_ids"] == 2
    assert section["unique_dataset_ids"] == 2


def test_corpus_section_reports_unavailable_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bam, "CORPUS_PATH", tmp_path / "missing.jsonl")
    section = bam._corpus_section()
    assert section["available"] is False


def test_graph_section_computes_type_counts_and_stub_nodes(tmp_path, monkeypatch):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": {
                    "node:dataset:a": {"node_type": "dataset", "properties": {}},
                    "node:dataset:b": {"node_type": "dataset", "properties": {}},
                    "node:method:x": {"node_type": "method", "properties": {"stub": True}},
                },
                "edges": {
                    "e1": {"edge_type": "dataset_has_modality"},
                    "e2": {"edge_type": "dataset_reanalysis_bridge_dataset"},
                    "e3": {
                        "edge_type": "dataset_similar_to_dataset",
                        "properties": {"cross_type": "same_region_cross_modality"},
                    },
                    "e4": {
                        "edge_type": "dataset_similar_to_dataset",
                        "properties": {"cross_type": "same_task_cross_species"},
                    },
                },
                "metadata": {"builder": "test"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bam, "GRAPH_PATH", graph_path)
    section = bam._graph_section()
    assert section["total_nodes"] == 3
    assert section["total_edges"] == 4
    assert section["node_type_counts"]["dataset"] == 2
    assert section["edge_type_counts"]["dataset_reanalysis_bridge_dataset"] == 1
    assert section["edge_type_counts"]["dataset_similar_to_dataset"] == 2
    assert section["stub_node_count"] == 1


def test_graph_section_display_edge_counts_use_cross_type_property():
    """Regression test: same_region_cross_modality/same_task_cross_species are
    NOT top-level edge_type values, they're a `cross_type` property under the
    single `dataset_similar_to_dataset` edge_type — caught as a real bug while
    building this script (first pass silently reported 0 for both)."""

    graph = bam._graph_section()
    assert graph["available"] is True
    display_counts = graph["display_edge_counts"]
    assert display_counts.get("same_region_cross_modality", 0) > 0
    assert display_counts.get("same_task_cross_species", 0) > 0
    assert "dataset_similar_to_dataset" not in display_counts


def test_paper_links_section_computes_real_match_rate(tmp_path, monkeypatch):
    path = tmp_path / "paper_dataset_links.jsonl"
    _write_jsonl(
        path,
        [
            {"dataset_record_id": "a", "match_method": "doi_exact"},
            {"dataset_record_id": "b", "match_method": "title_fuzzy_local"},
            {"dataset_record_id": "c", "match_method": "not_found"},
            {"dataset_record_id": "d", "match_method": "not_found"},
        ],
    )
    monkeypatch.setattr(bam, "PAPER_LINKS_PATH", path)
    monkeypatch.setattr(bam, "ADDITIONAL_PAPER_LINKS_PATHS", {})
    section = bam._paper_links_section()
    assert section["total_rows"] == 4
    assert section["real_matches"] == 2
    assert section["real_match_rate"] == 0.5
    assert section["combined_datasets_with_real_link"] == 2
    assert set(section["by_source"]) == {"openalex"}


def test_paper_links_section_combines_additional_sources(tmp_path, monkeypatch):
    openalex_path = tmp_path / "paper_dataset_links.jsonl"
    _write_jsonl(
        openalex_path,
        [
            {"dataset_record_id": "a", "match_method": "doi_exact"},
            {"dataset_record_id": "b", "match_method": "not_found"},
        ],
    )
    datacite_path = tmp_path / "paper_dataset_links.datacite.jsonl"
    _write_jsonl(
        datacite_path,
        [
            # Overlaps dataset "a" -- should not double-count in the union.
            {"dataset_record_id": "a", "match_method": "datacite_related_identifier"},
            {"dataset_record_id": "c", "match_method": "datacite_related_identifier"},
            {"dataset_record_id": "d", "match_method": "not_applicable_no_dataset_doi"},
        ],
    )
    monkeypatch.setattr(bam, "PAPER_LINKS_PATH", openalex_path)
    monkeypatch.setattr(bam, "ADDITIONAL_PAPER_LINKS_PATHS", {"datacite": datacite_path})

    section = bam._paper_links_section()

    assert section["real_matches"] == 1  # OpenAlex-only backward-compat field
    assert section["by_source"]["datacite"]["real_matches"] == 2
    # openalex real dataset_ids = {a}; datacite real dataset_ids = {a, c};
    # union = {a, c} -- "a" must not be double-counted despite appearing in
    # both sources.
    assert section["combined_datasets_with_real_link"] == 2


def test_ablation_section_flags_partial_run(tmp_path, monkeypatch):
    path = tmp_path / "ablation.json"
    path.write_text(
        json.dumps(
            {
                "rungs": [
                    {"rung": "bm25", "status": "skipped", "metrics": {}},
                    {
                        "rung": "hybrid_graph",
                        "status": "ok",
                        "metrics": {"ndcg@10": 0.85},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bam, "ABLATION_REPORT_PATH", path)
    section = bam._ablation_section()
    assert section["is_partial_run"] is True
    assert section["skipped_rungs"] == ["bm25"]
    assert section["rungs"]["hybrid_graph"]["metrics"]["ndcg@10"] == 0.85


def test_qrels_section_reports_zero_rows_for_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bam,
        "QRELS_FILES",
        {"gold": tmp_path / "missing_gold.jsonl"},
    )
    section = bam._qrels_section()
    assert section["gold"]["available"] is False
    assert section["gold"]["rows"] == 0


def test_build_manifest_on_real_repo_files_is_internally_consistent():
    """Integration check against the real repo (not a fixture): confirms the
    script runs end-to-end and produces cross-referenced-consistent output."""

    manifest = bam.build_manifest()
    assert manifest["corpus"]["available"] is True
    assert manifest["corpus"]["row_count"] > 0
    assert manifest["knowledge_graph"]["available"] is True
    assert manifest["knowledge_graph"]["total_nodes"] > 0
    # reanalysis_edges must be sourced from the same graph's edge_type_counts
    graph_edge_counts = manifest["knowledge_graph"]["edge_type_counts"]
    for edge_type, count in manifest["reanalysis_edges"].items():
        assert count == graph_edge_counts.get(edge_type, 0)
