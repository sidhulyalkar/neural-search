import json

import pytest
from pydantic import ValidationError

from neural_search.graph import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    graph_from_dict,
    graph_to_dict,
    make_edge_id,
    make_node_id,
    normalize_edge_type,
    normalize_node_type,
    read_graph_json,
    read_graph_jsonl,
    validate_graph,
    write_graph_json,
    write_graph_jsonl,
)


def _evidence(**overrides) -> GraphEvidence:
    payload = {
        "evidence_id": "evidence:test:1",
        "source_type": "Normalized Dataset",
        "source_id": "dataset:dandi:000026",
        "source_field": "tasks",
        "evidence_text": "Reversal learning",
        "confidence": 0.91,
        "extractor_name": "test_extractor",
        "extractor_version": "v0.5.0",
    }
    payload.update(overrides)
    return GraphEvidence(**payload)


def _node(node_type: str, *parts: str, label: str | None = None) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id(node_type, *parts),
        node_type=node_type,
        label=label or parts[-1].replace("_", " ").title(),
        aliases=[parts[-1]],
        source_ids=[":".join(parts)],
        properties={"source": "test"},
        evidence=[_evidence()],
        confidence=0.9,
        created_at="2026-05-24T00:00:00+00:00",
    )


def _graph() -> KnowledgeGraph:
    dataset = _node("dataset", "dandi", "000026", label="DANDI 000026")
    task = _node("task", "reversal_learning", label="Reversal learning")
    edge = KnowledgeGraphEdge(
        edge_id=make_edge_id(dataset.node_id, "dataset_has_task", task.node_id),
        source_node_id=dataset.node_id,
        target_node_id=task.node_id,
        edge_type="Dataset Has Task",
        directed=True,
        confidence=0.88,
        evidence=[_evidence(source_field="title")],
        properties={"match_type": "label"},
        created_at="2026-05-24T00:00:00+00:00",
    )
    return KnowledgeGraph(
        nodes={dataset.node_id: dataset, task.node_id: task},
        edges={edge.edge_id: edge},
        metadata={"graph_version": "v0.5.0"},
    )


def test_stable_node_and_edge_ids_match_expected_formats():
    dataset_id = make_node_id("dataset", "dandi", "000026")
    paper_id = make_node_id("paper", "openalex", "W123456789")
    task_id = make_node_id("task", "reversal_learning")
    edge_id = make_edge_id(dataset_id, "dataset_has_task", task_id)
    paper_edge_id = make_edge_id(
        make_node_id("paper", "openalex", "W123"),
        "paper_uses_dataset",
        dataset_id,
    )

    assert dataset_id == "node:dataset:dandi:000026"
    assert paper_id == "node:paper:openalex:W123456789"
    assert edge_id == "edge:dataset:dandi:000026:has_task:reversal_learning"
    assert paper_edge_id == "edge:paper:openalex:W123:uses_dataset:dandi:000026"


def test_node_edge_and_evidence_types_are_normalized():
    evidence = _evidence()
    node = KnowledgeGraphNode(
        node_id=make_node_id("Brain Region", "orbitofrontal_cortex"),
        node_type="Brain Region",
        label="Orbitofrontal cortex",
        confidence=0.8,
    )
    edge = KnowledgeGraphEdge(
        edge_id="edge:example",
        source_node_id="node:a",
        target_node_id="node:b",
        edge_type="Paper Reports Finding",
        confidence=0.7,
    )

    assert normalize_node_type("Brain Region") == "brain_region"
    assert normalize_edge_type("Paper Reports Finding") == "paper_reports_finding"
    assert evidence.source_type == "normalized_dataset"
    assert node.node_type == "brain_region"
    assert edge.edge_type == "paper_reports_finding"


def test_valid_graph_roundtrips_through_dict_json_and_jsonl(tmp_path):
    graph = _graph()

    as_dict = graph_to_dict(graph)
    from_dict = graph_from_dict(as_dict)
    json_path = write_graph_json(graph, tmp_path / "graph.json")
    jsonl_path = write_graph_jsonl(graph, tmp_path / "graph.jsonl")

    assert validate_graph(graph) == graph
    assert from_dict == graph
    assert read_graph_json(json_path) == graph
    assert read_graph_jsonl(jsonl_path) == graph


def test_invalid_empty_node_and_edge_ids_fail_validation():
    with pytest.raises(ValidationError):
        KnowledgeGraphNode(node_id=" ", node_type="task", label="Task")

    with pytest.raises(ValidationError):
        KnowledgeGraphEdge(
            edge_id=" ",
            source_node_id="node:dataset:dandi:000026",
            target_node_id="node:task:go_nogo",
            edge_type="dataset_has_task",
        )


def test_confidence_bounds_are_enforced():
    with pytest.raises(ValidationError):
        _evidence(confidence=1.1)

    with pytest.raises(ValidationError):
        KnowledgeGraphNode(
            node_id=make_node_id("task", "go_nogo"),
            node_type="task",
            label="Go NoGo",
            confidence=-0.1,
        )


def test_graph_rejects_edges_that_reference_missing_nodes():
    dataset = _node("dataset", "dandi", "000026")
    missing_task_id = make_node_id("task", "go_nogo")
    edge = KnowledgeGraphEdge(
        edge_id=make_edge_id(dataset.node_id, "dataset_has_task", missing_task_id),
        source_node_id=dataset.node_id,
        target_node_id=missing_task_id,
        edge_type="dataset_has_task",
    )

    graph = KnowledgeGraph(nodes={dataset.node_id: dataset}, edges={edge.edge_id: edge})
    with pytest.raises(ValueError, match="edge target does not resolve"):
        validate_graph(graph, strict=True)


def test_graph_rejects_mismatched_identity_map_keys():
    graph = _graph()
    node = next(iter(graph.nodes.values()))

    with pytest.raises(ValidationError, match="node map key does not match"):
        KnowledgeGraph(nodes={"node:wrong": node}, edges={})


def test_graph_jsonl_rejects_duplicate_node_ids(tmp_path):
    graph = _graph()
    node = next(iter(graph.nodes.values()))
    path = tmp_path / "duplicate_graph.jsonl"

    with path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": "metadata", "metadata": {}}))
        handle.write("\n")
        payload = {"record_type": "node", "node": node.model_dump(mode="json")}
        handle.write(json.dumps(payload))
        handle.write("\n")
        handle.write(json.dumps(payload))
        handle.write("\n")

    with pytest.raises(ValueError, match="duplicate node ID"):
        read_graph_jsonl(path)
