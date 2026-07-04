from neural_search.graph import build_graph_from_records, write_graph_json
from neural_search.graph.schema import KnowledgeGraphEdge, make_edge_id
from neural_search.graph.search_features import (
    compute_graph_features_for_result,
    graph_context_score,
    load_graph_if_exists,
)
from neural_search.normalized import (
    make_dataset_id,
    make_evidence_label_id,
    make_paper_id,
)
from neural_search.schemas import (
    AnalysisAffordance,
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
)


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label),
        label=label,
        label_type=label_type,
        confidence=0.9,
        evidence_text=label,
        extractor_name="test",
        extractor_version="v0.5.0",
    )


def _graph():
    dataset = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("dandi", "000026"),
        source="dandi",
        source_id="000026",
        title="Mouse OFC reversal learning",
        tasks=[_label("task", "reversal_learning")],
        modalities=[_label("modality", "Neuropixels")],
        brain_regions=[_label("brain_region", "orbitofrontal_cortex")],
        behavioral_events=[_label("behavioral_event", "spike_times")],
        data_standards=[_label("data_standard", "NWB")],
        analysis_affordances=[
            AnalysisAffordance(
                analysis_id="decoding",
                support_level="high",
                confidence=0.9,
                required_fields_present=["spike_times"],
                evidence=["spike_times"],
                detector_name="test",
                detector_version="v0.5.0",
            )
        ],
        linked_papers=[make_paper_id("openalex", "W123")],
    )
    paper = NormalizedPaperRecord(
        paper_id=make_paper_id("openalex", "W123"),
        source="openalex",
        source_id="W123",
        title="OFC reversal paper",
        linked_datasets=[dataset.dataset_id],
    )
    return build_graph_from_records([dataset], [paper])


def test_graph_features_are_empty_and_zero_when_graph_is_absent():
    features = compute_graph_features_for_result(None, "dataset:dandi:000026")

    assert features["graph_available"] is False
    assert features["graph_degree"] == 0
    assert features["requirement_matches"] == {
        "modality": [],
        "behavioral_event": [],
        "data_standard": [],
        "required_signal": [],
    }
    assert graph_context_score(None, "dataset:dandi:000026") == 0.0


def test_graph_features_and_context_score_are_bounded():
    graph = _graph()

    features = compute_graph_features_for_result(
        graph,
        "dataset:dandi:000026",
        {
            "tasks": ["reversal_learning"],
            "modalities": ["Neuropixels"],
            "analysis": ["Decoding"],
        },
    )
    score = graph_context_score(
        graph,
        "dataset:dandi:000026",
        {"tasks": ["reversal_learning"], "modalities": ["Neuropixels"]},
    )

    assert features["graph_available"] is True
    assert features["linked_papers"]
    assert features["analysis_affordances"]
    assert features["requirement_matches"]["modality"]
    assert features["requirement_matches"]["data_standard"]
    assert features["requirement_matches"]["required_signal"]
    assert 0 < score <= 0.25


def test_graph_features_include_relationship_and_reanalysis_edges():
    graph = _graph()
    source = "node:dataset:dandi:000026"
    target = "node:dataset:dandi:000027"
    graph.nodes[target] = graph.nodes[source].model_copy(
        update={"node_id": target, "label": "Related OFC dataset"}
    )
    edge = KnowledgeGraphEdge(
        edge_id=make_edge_id(source, "dataset_reanalysis_bridge_dataset", target),
        source_node_id=source,
        target_node_id=target,
        edge_type="dataset_reanalysis_bridge_dataset",
        confidence=0.8,
        properties={
            "relationship_type": "multimodal_reanalysis_bridge",
            "explanation": "shared OFC region across modalities",
        },
    )
    graph.edges[edge.edge_id] = edge

    features = compute_graph_features_for_result(graph, "dataset:dandi:000026")
    score = graph_context_score(graph, "dataset:dandi:000026")

    assert features["relationship_edges"][0]["relationship_type"] == "multimodal_reanalysis_bridge"
    assert score > 0.0


def test_reanalysis_candidate_edges_excluded_from_graph_degree():
    """Regression test for a 2026-07-01 NDCG@10 regression (0.8594 -> 0.8494 on
    the 317-query canonical benchmark): once dataset_old_dataset_new_method_candidate
    edges were first populated at scale (59,126 edges across ~80% of datasets),
    an unbounded degree count dominated by these edges flattened previously
    useful connectivity differentiation. They must be excluded from
    graph_degree the same way dataset_similar_to_dataset already is."""

    graph = _graph()
    dataset_node_id = "node:dataset:dandi:000026"
    baseline_degree = compute_graph_features_for_result(graph, "dataset:dandi:000026")[
        "graph_degree"
    ]

    for i in range(20):
        method_node_id = f"method:fake_method_{i}"
        graph.nodes[method_node_id] = graph.nodes[dataset_node_id].model_copy(
            update={"node_id": method_node_id, "node_type": "method", "label": f"Method {i}"}
        )
        edge = KnowledgeGraphEdge(
            edge_id=f"edge:{dataset_node_id}:new_method_candidate:{method_node_id}",
            source_node_id=dataset_node_id,
            target_node_id=method_node_id,
            edge_type="dataset_old_dataset_new_method_candidate",
            confidence=0.7,
            properties={"requires_human_review": True},
        )
        graph.edges[edge.edge_id] = edge

    degree_after_candidates = compute_graph_features_for_result(graph, "dataset:dandi:000026")[
        "graph_degree"
    ]
    assert degree_after_candidates == baseline_degree


def test_reanalysis_bridge_edges_excluded_from_graph_degree():
    """Regression test for a second, smaller 2026-07-01 NDCG@10 regression
    (0.8594 -> 0.8545): populating dataset_reanalysis_bridge_dataset edges
    (2,517 edges, 818 datasets) also inflated graph_degree enough to move the
    canonical benchmark. Must be excluded the same way the candidate edges
    already are."""

    graph = _graph()
    dataset_node_id = "node:dataset:dandi:000026"
    baseline_degree = compute_graph_features_for_result(graph, "dataset:dandi:000026")[
        "graph_degree"
    ]

    for i in range(20):
        other_dataset_id = f"node:dataset:dandi:fake_{i}"
        graph.nodes[other_dataset_id] = graph.nodes[dataset_node_id].model_copy(
            update={"node_id": other_dataset_id, "label": f"Fake dataset {i}"}
        )
        edge = KnowledgeGraphEdge(
            edge_id=f"edge:{dataset_node_id}:reanalysis_bridge:{other_dataset_id}",
            source_node_id=dataset_node_id,
            target_node_id=other_dataset_id,
            edge_type="dataset_reanalysis_bridge_dataset",
            confidence=0.34,
            properties={"requires_human_review": True},
        )
        graph.edges[edge.edge_id] = edge

    degree_after_bridge_edges = compute_graph_features_for_result(graph, "dataset:dandi:000026")[
        "graph_degree"
    ]
    assert degree_after_bridge_edges == baseline_degree


def test_reanalysis_edge_weight_defaults_to_zero():
    """Locks in the 2026-07-01 fix: this weight was set speculatively before
    dataset_old_dataset_new_method_candidate edges existed, and a nonzero
    value measurably regressed NDCG@10 once they were populated. Keep at 0.0
    until the reanalysis-candidate signal is validated against gold qrels."""

    from neural_search.graph.search_features import DEFAULT_GRAPH_SEARCH_WEIGHTS

    assert DEFAULT_GRAPH_SEARCH_WEIGHTS["reanalysis_edge"] == 0.0


def test_linked_paper_weight_defaults_to_zero():
    """Locks in the 2026-07-02 fix: this weight (previously 0.04) was
    permanently inert before any paper nodes existed in production, and a
    nonzero value measurably regressed NDCG@10 (0.8594 -> 0.8583) once
    neural_search.graph.paper_node_builder populated real paper_mentions_dataset/
    paper_uses_dataset edges. Keep at 0.0 until validated against gold qrels."""

    from neural_search.graph.search_features import DEFAULT_GRAPH_SEARCH_WEIGHTS

    assert DEFAULT_GRAPH_SEARCH_WEIGHTS["linked_paper"] == 0.0


def test_graph_features_return_empty_requirement_matches_without_analysis_edges():
    graph = _graph()

    features = compute_graph_features_for_result(graph, "missing")

    assert features["graph_available"] is True
    assert features["requirement_matches"] == {
        "modality": [],
        "behavioral_event": [],
        "data_standard": [],
        "required_signal": [],
    }


def test_load_graph_if_exists_is_optional(tmp_path):
    graph = _graph()
    graph_path = write_graph_json(graph, tmp_path / "graph.json")

    assert load_graph_if_exists(tmp_path / "missing.json") is None
    assert load_graph_if_exists(graph_path) == graph
