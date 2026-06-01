from neural_search.graph import build_graph_from_records, write_graph_json
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
