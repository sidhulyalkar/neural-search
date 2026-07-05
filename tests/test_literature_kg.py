"""Tests for literature artifact integration into the knowledge graph."""

from __future__ import annotations

import json

from neural_search.graph.schema import (
    KnowledgeGraph,
    make_edge_id,
    make_node_id,
    validate_graph,
)
from neural_search.literature.kg_builder import (
    add_findings_to_graph,
    add_papers_from_shards,
)


def _write_jsonl(path, records) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_add_papers_creates_paper_and_venue_nodes(tmp_path):
    shard = tmp_path / "tier1_batch_0000.jsonl"
    _write_jsonl(
        shard,
        [
            {
                "paper_id": "paper:openalex:W123",
                "source": "openalex",
                "source_id": "W123",
                "title": "Hippocampal replay supports memory",
                "abstract": "Replay in hippocampus during memory consolidation.",
                "doi": "https://doi.org/10.123/example",
                "year": 2020,
                "citation_count": 150,
                "venue": "Nature Neuroscience",
            }
        ],
    )
    graph = KnowledgeGraph()

    stats = add_papers_from_shards(graph, tmp_path)

    paper_id = make_node_id("paper", "openalex", "W123")
    venue_id = make_node_id("venue", "nature_neuroscience")
    assert stats == {"papers_added": 1, "venues_added": 1, "links_added": 0}
    assert paper_id in graph.nodes
    assert venue_id in graph.nodes
    assert make_edge_id(paper_id, "paper_published_in", venue_id) in graph.edges
    assert graph.nodes[paper_id].properties["citation_count"] == 150
    validate_graph(graph)


def test_add_papers_dataset_link_edges(tmp_path):
    shard = tmp_path / "tier1_batch_0000.jsonl"
    links = tmp_path / "links.jsonl"
    _write_jsonl(
        shard,
        [
            {
                "paper_id": "paper:openalex:W123",
                "source": "openalex",
                "source_id": "W123",
                "title": "A dataset paper",
            }
        ],
    )
    _write_jsonl(
        links,
        [
            {
                "dataset_record_id": "dandi:000026",
                "paper_openalex_id": "W123",
                "paper_doi": "10.123/example",
                "paper_title": "A dataset paper",
                "paper_year": 2020,
                "match_method": "doi_exact",
                "confidence": 1.0,
            }
        ],
    )
    graph = KnowledgeGraph()

    stats = add_papers_from_shards(graph, tmp_path, links)

    dataset_id = make_node_id("dataset", "dandi", "000026")
    paper_id = make_node_id("paper", "openalex", "W123")
    assert stats["links_added"] == 1
    assert dataset_id in graph.nodes
    assert make_edge_id(dataset_id, "dataset_linked_to_paper", paper_id) in graph.edges
    validate_graph(graph)


def test_add_findings_creates_finding_and_concept_edges(tmp_path):
    findings = tmp_path / "findings.jsonl"
    _write_jsonl(
        findings,
        [
            {
                "paper_id": "paper:openalex:W123",
                "paper_doi": "10.123/example",
                "finding_id": "paper:openalex:W123:f0",
                "finding_text": "CA1 replay increases after spatial learning.",
                "result_direction": "increase",
                "regions": ["CA1"],
                "species": ["mouse"],
                "modalities": ["extracellular_ephys"],
                "tasks": ["spatial_learning"],
                "cell_types": [],
                "molecules": [],
                "confidence": 0.86,
                "extraction_model": "claude-haiku",
                "extracted_at": "2026-06-17T00:00:00+00:00",
            }
        ],
    )
    graph = KnowledgeGraph()

    stats = add_findings_to_graph(graph, findings)

    paper_id = make_node_id("paper", "openalex", "W123")
    finding_id = make_node_id("finding", "paper:openalex:W123:f0")
    region_id = make_node_id("brain_region", "ca1")
    task_id = make_node_id("task", "spatial_learning")
    assert stats["findings_added"] == 1
    assert make_edge_id(paper_id, "paper_reports_finding", finding_id) in graph.edges
    assert make_edge_id(finding_id, "finding_involves_region", region_id) in graph.edges
    assert make_edge_id(finding_id, "finding_involves_task", task_id) in graph.edges
    validate_graph(graph)


def test_add_findings_attaches_region_atlas_crosswalk(tmp_path):
    findings = tmp_path / "findings.jsonl"
    _write_jsonl(
        findings,
        [
            {
                "paper_id": "paper:openalex:W123",
                "finding_id": "paper:openalex:W123:f0",
                "finding_text": "Replay increases in hippocampus after learning.",
                "result_direction": "increase",
                "regions": ["hippocampus"],
                "species": [],
                "modalities": [],
                "tasks": [],
                "confidence": 0.86,
            }
        ],
    )
    graph = KnowledgeGraph()

    add_findings_to_graph(graph, findings)

    region_id = make_node_id("brain_region", "hippocampus")
    region_node = graph.nodes[region_id]
    assert region_node.properties["canonical_region_id"] == "hippocampus"
    assert region_node.properties["atlas_refs"]["uberon"] == "UBERON:0002421"
    validate_graph(graph)


def test_add_findings_unmatched_region_has_no_atlas_refs(tmp_path):
    findings = tmp_path / "findings.jsonl"
    _write_jsonl(
        findings,
        [
            {
                "paper_id": "paper:openalex:W124",
                "finding_id": "paper:openalex:W124:f0",
                "finding_text": "Activity changed in a made-up brain area.",
                "result_direction": "increase",
                "regions": ["not a real brain area"],
                "species": [],
                "modalities": [],
                "tasks": [],
                "confidence": 0.7,
            }
        ],
    )
    graph = KnowledgeGraph()

    add_findings_to_graph(graph, findings)

    region_id = make_node_id("brain_region", "not_a_real_brain_area")
    region_node = graph.nodes[region_id]
    assert "atlas_refs" not in region_node.properties
    validate_graph(graph)


def test_add_findings_attaches_task_cogat_crosswalk(tmp_path):
    findings = tmp_path / "findings.jsonl"
    _write_jsonl(
        findings,
        [
            {
                "paper_id": "paper:openalex:W123",
                "finding_id": "paper:openalex:W123:f0",
                "finding_text": "Performance improved on the stroop task after training.",
                "result_direction": "increase",
                "regions": [],
                "species": [],
                "modalities": [],
                "tasks": ["stroop_task"],
                "confidence": 0.8,
            }
        ],
    )
    graph = KnowledgeGraph()

    add_findings_to_graph(graph, findings)

    task_id = make_node_id("task", "stroop_task")
    task_node = graph.nodes[task_id]
    assert task_node.properties["canonical_task_id"] == "stroop_task"
    assert task_node.properties["cogat_label"] == "Stroop task"
    validate_graph(graph)


def test_add_findings_propagates_evidence_span(tmp_path):
    findings = tmp_path / "findings.jsonl"
    _write_jsonl(
        findings,
        [
            {
                "paper_id": "paper:openalex:W123",
                "finding_id": "paper:openalex:W123:f0",
                "finding_text": "CA1 replay increases after spatial learning.",
                "result_direction": "increase",
                "regions": ["CA1"],
                "species": ["mouse"],
                "modalities": [],
                "tasks": [],
                "confidence": 0.86,
                "char_start": 42,
                "char_end": 88,
                "sentence_id": 2,
            }
        ],
    )
    graph = KnowledgeGraph()

    add_findings_to_graph(graph, findings)

    finding_id = make_node_id("finding", "paper:openalex:W123:f0")
    evidence = graph.nodes[finding_id].evidence[0]
    assert evidence.char_start == 42
    assert evidence.char_end == 88
    assert evidence.sentence_id == 2
    validate_graph(graph)


def test_add_findings_creates_typed_field_edges(tmp_path):
    findings = tmp_path / "findings.jsonl"
    _write_jsonl(
        findings,
        [
            {
                "paper_id": "paper:openalex:W123",
                "finding_id": "paper:openalex:W123:f0",
                "finding_text": "Theta oscillations increased transiently in the hippocampus.",
                "result_direction": "increase",
                "regions": ["hippocampus"],
                "species": [],
                "modalities": [],
                "tasks": [],
                "confidence": 0.8,
                "frequency_band": ["theta"],
                "temporal_pattern": ["transient"],
                "spatial_frame": ["local"],
            }
        ],
    )
    graph = KnowledgeGraph()

    add_findings_to_graph(graph, findings)

    finding_id = make_node_id("finding", "paper:openalex:W123:f0")
    band_id = make_node_id("frequency_band", "theta")
    pattern_id = make_node_id("temporal_pattern", "transient")
    frame_id = make_node_id("spatial_frame", "local")
    assert make_edge_id(finding_id, "finding_has_frequency_band", band_id) in graph.edges
    assert make_edge_id(finding_id, "finding_has_temporal_pattern", pattern_id) in graph.edges
    assert make_edge_id(finding_id, "finding_has_spatial_frame", frame_id) in graph.edges
    validate_graph(graph)


def test_add_findings_without_typed_fields_skips_typed_edges(tmp_path):
    """Backward compatibility: raw (un-enriched) finding records produce no typed edges."""
    findings = tmp_path / "findings.jsonl"
    _write_jsonl(
        findings,
        [
            {
                "paper_id": "paper:openalex:W999",
                "finding_id": "paper:openalex:W999:f0",
                "finding_text": "Some finding text.",
                "result_direction": "increase",
                "regions": [],
                "species": [],
                "modalities": [],
                "tasks": [],
                "confidence": 0.8,
            }
        ],
    )
    graph = KnowledgeGraph()

    add_findings_to_graph(graph, findings)

    typed_edges = [
        e
        for e in graph.edges.values()
        if e.edge_type
        in {"finding_has_frequency_band", "finding_has_temporal_pattern", "finding_has_spatial_frame"}
    ]
    assert typed_edges == []
    validate_graph(graph)


def test_duplicate_papers_are_merged_not_readded(tmp_path):
    shard = tmp_path / "tier1_batch_0000.jsonl"
    paper = {
        "paper_id": "paper:openalex:W123",
        "source": "openalex",
        "source_id": "W123",
        "title": "Duplicate paper",
    }
    _write_jsonl(shard, [paper, paper])
    graph = KnowledgeGraph()

    stats = add_papers_from_shards(graph, tmp_path)

    assert stats["papers_added"] == 1
    assert list(graph.nodes).count(make_node_id("paper", "openalex", "W123")) == 1
    validate_graph(graph)
