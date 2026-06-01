from neural_search.graph import (
    GraphEvidence,
    KnowledgeGraph,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    audit_graph_quality,
    make_edge_id,
    make_node_id,
)


def _evidence() -> GraphEvidence:
    return GraphEvidence(
        evidence_id="evidence:test:1",
        source_type="test",
        source_id="1",
        confidence=0.9,
        extractor_name="test",
        extractor_version="v0.9.0",
    )


def _node(node_type: str, label: str, **properties) -> KnowledgeGraphNode:
    return KnowledgeGraphNode(
        node_id=make_node_id(node_type, label),
        node_type=node_type,
        label=label,
        properties=properties,
        confidence=0.9,
        evidence=[_evidence()],
    )


def _edge(
    source: KnowledgeGraphNode,
    edge_type: str,
    target: KnowledgeGraphNode,
    confidence: float = 0.9,
) -> KnowledgeGraphEdge:
    return KnowledgeGraphEdge(
        edge_id=make_edge_id(source.node_id, edge_type, target.node_id),
        source_node_id=source.node_id,
        target_node_id=target.node_id,
        edge_type=edge_type,
        confidence=confidence,
        evidence=[_evidence()],
    )


def test_graph_quality_passes_complete_required_graph():
    dataset = _node("dataset", "D1")
    task = _node("task", "go_nogo")
    edge = _edge(dataset, "dataset_has_task", task)
    graph = KnowledgeGraph(
        nodes={dataset.node_id: dataset, task.node_id: task},
        edges={edge.edge_id: edge},
    )

    report = audit_graph_quality(
        graph,
        required_node_types=["dataset", "task"],
        required_edge_types=["dataset_has_task"],
    )

    assert report.passed
    assert report.issue_count == 0
    assert report.node_type_counts == {"dataset": 1, "task": 1}
    assert report.edge_type_counts == {"dataset_has_task": 1}


def test_graph_quality_detects_orphans_placeholders_and_weak_edges():
    dataset = _node("dataset", "D1")
    task = _node("task", "go_nogo")
    orphan = _node("paper", "P1", placeholder=True)
    edge = _edge(dataset, "dataset_has_task", task, confidence=0.2)
    graph = KnowledgeGraph(
        nodes={
            dataset.node_id: dataset,
            task.node_id: task,
            orphan.node_id: orphan,
        },
        edges={edge.edge_id: edge},
    )

    report = audit_graph_quality(graph, weak_confidence_threshold=0.5)
    codes = [issue.code for issue in report.issues]

    assert report.passed
    assert codes == ["orphan_node", "unresolved_placeholder", "weak_edge"]
    assert report.warning_count == 3


def test_graph_quality_detects_missing_required_types():
    dataset = _node("dataset", "D1")
    graph = KnowledgeGraph(nodes={dataset.node_id: dataset}, edges={})

    report = audit_graph_quality(
        graph,
        required_node_types=["dataset", "paper"],
        required_edge_types=["paper_uses_dataset"],
    )

    assert not report.passed
    assert [issue.code for issue in report.issues] == [
        "missing_required_edge_type",
        "missing_required_node_type",
    ]


def test_graph_quality_detects_raw_invalid_confidence_and_dangling_edges():
    raw_graph = {
        "nodes": {
            "node:dataset:D1": {
                "node_id": "node:dataset:D1",
                "node_type": "dataset",
                "label": "D1",
                "confidence": 1.4,
            }
        },
        "edges": {
            "edge:dataset:D1:has_task:T1": {
                "edge_id": "edge:dataset:D1:has_task:T1",
                "source_node_id": "node:dataset:D1",
                "target_node_id": "node:task:T1",
                "edge_type": "dataset_has_task",
                "confidence": -0.1,
            }
        },
    }

    report = audit_graph_quality(raw_graph)
    codes = [issue.code for issue in report.issues]

    assert not report.passed
    assert codes == [
        "dangling_edge_reference",
        "invalid_edge_confidence",
        "invalid_node_confidence",
    ]
    assert report.error_count == 3
