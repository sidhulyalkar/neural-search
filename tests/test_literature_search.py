"""Tests for literature paper and finding search."""

from __future__ import annotations

import json

from neural_search.literature.search import search_findings, search_papers


def _write_jsonl(path, records) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_search_papers_returns_ranked_results(tmp_path):
    shard = tmp_path / "tier1_batch_0000.jsonl"
    _write_jsonl(
        shard,
        [
            {
                "paper_id": "paper:openalex:W1",
                "title": "Hippocampal theta supports spatial memory",
                "abstract": "Theta rhythms in CA1 predict navigation choices.",
                "year": 2020,
                "citation_count": 250,
                "venue": "Nature Neuroscience",
                "doi": "10.123/theta",
                "linked_datasets": ["dandi:000001"],
            },
            {
                "paper_id": "paper:openalex:W2",
                "title": "Retinal development in zebrafish",
                "abstract": "Visual system development.",
                "year": 2018,
                "citation_count": 120,
                "venue": "Neuron",
            },
        ],
    )

    results = search_papers("hippocampal theta memory", shard_dir=tmp_path, limit=2)

    assert [r.paper_id for r in results] == ["paper:openalex:W1"]
    assert results[0].result_type == "paper"
    assert results[0].linked_datasets == ["dandi:000001"]
    assert results[0].abstract_snippet
    assert results[0].why_matched


def test_search_papers_empty_shards_returns_empty(tmp_path):
    assert search_papers("hippocampus", shard_dir=tmp_path / "missing", limit=5) == []


def test_search_papers_filters_by_min_citations(tmp_path):
    shard = tmp_path / "tier1_batch_0000.jsonl"
    _write_jsonl(
        shard,
        [
            {"paper_id": "paper:openalex:W1", "title": "Hippocampus memory", "citation_count": 10},
            {"paper_id": "paper:openalex:W2", "title": "Hippocampus memory", "citation_count": 200},
        ],
    )

    results = search_papers(
        "hippocampus memory",
        shard_dir=tmp_path,
        filters={"min_citations": 100},
    )

    assert [r.paper_id for r in results] == ["paper:openalex:W2"]


def test_search_findings_filters_by_region(tmp_path):
    findings = tmp_path / "findings.jsonl"
    _write_jsonl(
        findings,
        [
            {
                "finding_id": "paper:openalex:W1:f0",
                "finding_text": "CA1 replay increases after spatial learning.",
                "result_direction": "increase",
                "regions": ["CA1"],
                "species": ["mouse"],
                "modalities": ["extracellular_ephys"],
                "tasks": ["spatial_learning"],
                "paper_id": "paper:openalex:W1",
                "paper_title": "Replay and learning",
                "paper_year": 2021,
            },
            {
                "finding_id": "paper:openalex:W2:f0",
                "finding_text": "V1 responses are modulated by attention.",
                "result_direction": "correlation",
                "regions": ["V1"],
                "species": ["macaque"],
                "modalities": ["ephys"],
                "tasks": ["attention"],
                "paper_id": "paper:openalex:W2",
            },
        ],
    )

    results = search_findings(
        "spatial learning replay",
        findings_path=findings,
        filters={"region": "CA1"},
    )

    assert [r.finding_id for r in results] == ["paper:openalex:W1:f0"]
    assert results[0].result_type == "finding"
    assert results[0].regions == ["CA1"]
    assert results[0].why_matched


def test_search_findings_empty_file_returns_empty(tmp_path):
    assert search_findings("memory", findings_path=tmp_path / "missing.jsonl") == []
