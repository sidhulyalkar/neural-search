"""Tests for neural_search.graph.paper_node_builder -- the first production
paper-node builder, wired into scripts/build_real_corpus_graph.py's
orphaned_layers list."""

from __future__ import annotations

import json

from neural_search.graph.paper_node_builder import (
    attach_retraction_status,
    build_doi_to_paper_node_id_index,
    build_paper_nodes_and_links,
    get_paper_trust_signals,
)
from neural_search.graph.schema import (
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
)
from neural_search.kg.schemas.evidence_tier import EvidenceTier


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _graph_with_dataset_nodes(*dataset_node_ids: str) -> KnowledgeGraph:
    """Minimal graph containing only the given dataset nodes -- mirrors how
    build_graph() already has all dataset nodes present before the
    orphaned_layers loop runs."""

    return KnowledgeGraph(
        nodes={
            node_id: KnowledgeGraphNode(node_id=node_id, node_type="dataset", label=node_id)
            for node_id in dataset_node_ids
        },
        edges={},
    )


def test_openalex_row_creates_paper_mentions_dataset_edge(tmp_path) -> None:
    openalex_path = tmp_path / "paper_dataset_links.jsonl"
    _write_jsonl(
        openalex_path,
        [
            {
                "dataset_record_id": "dandi:000785",
                "paper_openalex_id": "W123",
                "paper_source": "openalex",
                "paper_source_id": "W123",
                "paper_doi": "10.1/x",
                "paper_title": "A paper",
                "paper_year": 2020,
                "match_method": "doi_exact",
                "confidence": 1.0,
            }
        ],
    )
    datacite_path = tmp_path / "paper_dataset_links.datacite.jsonl"
    _write_jsonl(datacite_path, [])
    base_graph = _graph_with_dataset_nodes("node:dataset:dandi:000785")

    graph = build_paper_nodes_and_links(base_graph, link_paths=(openalex_path, datacite_path))

    paper_node_id = "node:paper:openalex:W123"
    dataset_node_id = "node:dataset:dandi:000785"
    assert paper_node_id in graph.nodes
    assert graph.nodes[paper_node_id].node_type == "paper"

    edges = list(graph.edges.values())
    assert len(edges) == 1
    edge = edges[0]
    assert edge.edge_type == "paper_mentions_dataset"
    assert edge.source_node_id == paper_node_id
    assert edge.target_node_id == dataset_node_id
    assert edge.properties["evidence_tier"] is None


def test_datacite_row_creates_paper_uses_dataset_edge_with_source_declared_tier(tmp_path) -> None:
    openalex_path = tmp_path / "paper_dataset_links.jsonl"
    _write_jsonl(openalex_path, [])
    datacite_path = tmp_path / "paper_dataset_links.datacite.jsonl"
    _write_jsonl(
        datacite_path,
        [
            {
                "dataset_record_id": "zenodo:11236154",
                "paper_openalex_id": "",
                "paper_source": "datacite",
                "paper_source_id": "10.1038/s41467-024-49226-9",
                "paper_doi": "10.1038/s41467-024-49226-9",
                "paper_title": None,
                "paper_year": None,
                "match_method": "datacite_related_identifier",
                "confidence": 1.0,
            }
        ],
    )
    base_graph = _graph_with_dataset_nodes("node:dataset:zenodo:11236154")

    graph = build_paper_nodes_and_links(base_graph, link_paths=(openalex_path, datacite_path))

    # make_node_id sanitizes non-token characters (e.g. "/", ".") in ID parts,
    # so a DOI-based source_id becomes an underscored token, not a literal DOI.
    (paper_node_id,) = graph.nodes.keys()
    assert paper_node_id.startswith("node:paper:datacite:")

    edges = list(graph.edges.values())
    assert len(edges) == 1
    edge = edges[0]
    assert edge.edge_type == "paper_uses_dataset"
    assert edge.source_node_id == paper_node_id
    assert edge.target_node_id == "node:dataset:zenodo:11236154"
    assert edge.properties["evidence_tier"] == EvidenceTier.SOURCE_DECLARED.value


def test_not_found_and_not_applicable_rows_produce_no_nodes_or_edges(tmp_path) -> None:
    openalex_path = tmp_path / "paper_dataset_links.jsonl"
    _write_jsonl(openalex_path, [{"dataset_record_id": "dandi:1", "match_method": "not_found"}])
    datacite_path = tmp_path / "paper_dataset_links.datacite.jsonl"
    _write_jsonl(
        datacite_path,
        [{"dataset_record_id": "dandi:2", "match_method": "not_applicable_no_dataset_doi"}],
    )
    base_graph = _graph_with_dataset_nodes("node:dataset:dandi:1", "node:dataset:dandi:2")

    graph = build_paper_nodes_and_links(base_graph, link_paths=(openalex_path, datacite_path))

    assert graph.nodes == {}
    assert graph.edges == {}


def test_missing_link_file_is_skipped_gracefully(tmp_path) -> None:
    missing_path = tmp_path / "does_not_exist.jsonl"
    base_graph = _graph_with_dataset_nodes()

    graph = build_paper_nodes_and_links(base_graph, link_paths=(missing_path,))

    assert graph.nodes == {}
    assert graph.edges == {}


def test_row_for_dataset_not_in_graph_is_skipped(tmp_path) -> None:
    """Root-cause regression test for a real fixture-pollution bug found
    2026-07-02: this builder must scope itself to datasets already present
    in the graph being built (like reanalysis_bridge_builder.py does),
    otherwise it silently injects edges for every real-world dataset in
    paper_dataset_links.jsonl regardless of which corpus is being
    processed -- caught by a fixture-scale test unexpectedly gaining 403
    stub dataset nodes from the real production artifact."""

    openalex_path = tmp_path / "paper_dataset_links.jsonl"
    _write_jsonl(
        openalex_path,
        [
            {
                "dataset_record_id": "dandi:999999",
                "paper_openalex_id": "W999",
                "paper_source": "openalex",
                "paper_source_id": "W999",
                "match_method": "doi_exact",
                "confidence": 1.0,
            }
        ],
    )
    datacite_path = tmp_path / "paper_dataset_links.datacite.jsonl"
    _write_jsonl(datacite_path, [])
    # Graph contains a *different* dataset than the one in the link file.
    base_graph = _graph_with_dataset_nodes("node:dataset:dandi:000785")

    graph = build_paper_nodes_and_links(base_graph, link_paths=(openalex_path, datacite_path))

    assert graph.nodes == {}
    assert graph.edges == {}


def test_same_paper_from_two_sources_creates_two_distinct_paper_nodes(tmp_path) -> None:
    """A dataset can have both a real OpenAlex row and a real DataCite row --
    these are different paper_source values and should not collide, even if
    they happen to describe the same underlying paper (no cross-source
    identity resolution in this phase)."""

    openalex_path = tmp_path / "paper_dataset_links.jsonl"
    _write_jsonl(
        openalex_path,
        [
            {
                "dataset_record_id": "zenodo:1",
                "paper_openalex_id": "W1",
                "paper_source": "openalex",
                "paper_source_id": "W1",
                "paper_doi": "10.1038/s41467-024-49226-9",
                "match_method": "doi_exact",
                "confidence": 1.0,
            }
        ],
    )
    datacite_path = tmp_path / "paper_dataset_links.datacite.jsonl"
    _write_jsonl(
        datacite_path,
        [
            {
                "dataset_record_id": "zenodo:1",
                "paper_source": "datacite",
                "paper_source_id": "10.1038/s41467-024-49226-9",
                "paper_doi": "10.1038/s41467-024-49226-9",
                "match_method": "datacite_related_identifier",
                "confidence": 1.0,
            }
        ],
    )
    base_graph = _graph_with_dataset_nodes("node:dataset:zenodo:1")

    graph = build_paper_nodes_and_links(base_graph, link_paths=(openalex_path, datacite_path))

    assert len(graph.nodes) == 2
    assert len(graph.edges) == 2


def _paper_graph_with_doi(doi: str) -> KnowledgeGraph:
    node_id = "node:paper:crossref:x"
    return KnowledgeGraph(
        nodes={
            node_id: KnowledgeGraphNode(
                node_id=node_id, node_type="paper", label="A paper", properties={"doi": doi}
            )
        },
        edges={},
    )


def test_attach_retraction_status_sets_property_on_matching_paper(tmp_path) -> None:
    retraction_path = tmp_path / "retractions.jsonl"
    _write_jsonl(
        retraction_path,
        [
            {
                "doi": "10.1016/j.micpro.2020.103768",
                "status": "retracted",
                "related_dois": ["10.1016/j.micpro.2020.103768"],
                "source": "crossref",
                "checked_at": "2026-07-02T00:00:00+00:00",
            }
        ],
    )
    graph = _paper_graph_with_doi("10.1016/j.micpro.2020.103768")

    updated = attach_retraction_status(graph, retraction_path=retraction_path)

    node = updated.nodes["node:paper:crossref:x"]
    assert node.properties["retraction_status"]["status"] == "retracted"


def test_attach_retraction_status_leaves_unmatched_papers_alone(tmp_path) -> None:
    retraction_path = tmp_path / "retractions.jsonl"
    _write_jsonl(retraction_path, [{"doi": "10.9999/other", "status": "retracted"}])
    graph = _paper_graph_with_doi("10.1038/s41597-020-0415-9")

    updated = attach_retraction_status(graph, retraction_path=retraction_path)

    node = updated.nodes["node:paper:crossref:x"]
    assert "retraction_status" not in node.properties


def test_attach_retraction_status_is_noop_when_artifact_missing(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    graph = _paper_graph_with_doi("10.1038/s41597-020-0415-9")

    updated = attach_retraction_status(graph, retraction_path=missing)

    assert updated.nodes["node:paper:crossref:x"].properties == {"doi": "10.1038/s41597-020-0415-9"}


def test_attach_retraction_status_does_not_mutate_input_graph(tmp_path) -> None:
    retraction_path = tmp_path / "retractions.jsonl"
    _write_jsonl(retraction_path, [{"doi": "10.1/x", "status": "retracted"}])
    graph = _paper_graph_with_doi("10.1/x")

    attach_retraction_status(graph, retraction_path=retraction_path)

    assert "retraction_status" not in graph.nodes["node:paper:crossref:x"].properties


def _graph_with_paper_and_dataset_edge(
    *, doi: str = "10.1/x", retraction_status: dict | None = None, evidence_tier: str | None = None
) -> KnowledgeGraph:
    paper_id = "node:paper:crossref:x"
    dataset_id = "node:dataset:dandi:000785"
    properties = {"doi": doi}
    if retraction_status is not None:
        properties["retraction_status"] = retraction_status
    edge_properties = {}
    if evidence_tier is not None:
        edge_properties["evidence_tier"] = evidence_tier
    return KnowledgeGraph(
        nodes={
            paper_id: KnowledgeGraphNode(
                node_id=paper_id, node_type="paper", label="A paper", properties=properties
            ),
            dataset_id: KnowledgeGraphNode(node_id=dataset_id, node_type="dataset", label="A dataset"),
        },
        edges={
            "edge:1": KnowledgeGraphEdge(
                edge_id="edge:1",
                source_node_id=paper_id,
                target_node_id=dataset_id,
                edge_type="paper_mentions_dataset",
                properties=edge_properties,
            )
        },
    )


def test_build_doi_to_paper_node_id_index_indexes_only_paper_nodes() -> None:
    graph = _graph_with_paper_and_dataset_edge()

    index = build_doi_to_paper_node_id_index(graph)

    assert index == {"10.1/x": "node:paper:crossref:x"}


def test_get_paper_trust_signals_returns_empty_dict_without_doi() -> None:
    graph = _graph_with_paper_and_dataset_edge()

    assert get_paper_trust_signals(graph, doi=None) == {}
    assert get_paper_trust_signals(graph, doi="10.1/unmatched") == {}


def test_get_paper_trust_signals_surfaces_retraction_status() -> None:
    status = {"status": "retracted", "related_dois": [], "source": "crossref"}
    graph = _graph_with_paper_and_dataset_edge(retraction_status=status)

    signals = get_paper_trust_signals(graph, doi="10.1/x")

    assert signals == {"retraction_status": status}


def test_get_paper_trust_signals_surfaces_link_evidence_tier_for_dataset() -> None:
    graph = _graph_with_paper_and_dataset_edge(evidence_tier="source_declared")

    signals = get_paper_trust_signals(
        graph, doi="10.1/x", dataset_node_id="node:dataset:dandi:000785"
    )

    assert signals == {"evidence_tier": "source_declared"}


def test_get_paper_trust_signals_omits_evidence_tier_for_wrong_dataset() -> None:
    graph = _graph_with_paper_and_dataset_edge(evidence_tier="source_declared")

    signals = get_paper_trust_signals(
        graph, doi="10.1/x", dataset_node_id="node:dataset:openneuro:ds999999"
    )

    assert signals == {}
