from neural_search.graph import (
    build_graph_from_records,
    find_datasets_for_experimental_design,
    list_experimental_designs,
    load_experimental_design_seeds,
)
from neural_search.normalized import make_dataset_id, make_evidence_label_id
from neural_search.schemas import (
    AnalysisAffordance,
    EvidenceLabel,
    NormalizedDatasetRecord,
    UsabilityFlags,
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


def test_load_experimental_design_seeds_from_default_file():
    designs = load_experimental_design_seeds()

    assert len(designs) >= 8
    assert "reversal_learning_ephys_experiment" in {design.id for design in designs}
    assert list_experimental_designs()


def test_experimental_design_matching_uses_requirements_and_minimum_flags():
    dataset = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("dandi", "000026"),
        source="dandi",
        source_id="000026",
        title="Mouse reversal learning ephys",
        tasks=[_label("task", "reversal_learning")],
        modalities=[_label("modality", "neuropixels")],
        behavioral_events=[
            _label("behavioral_event", "choice"),
            _label("behavioral_event", "reward"),
            _label("behavioral_event", "trial_outcome"),
        ],
        analysis_affordances=[
            AnalysisAffordance(
                analysis_id="q_learning_modeling",
                support_level="high",
                confidence=0.9,
                evidence=["choice", "reward", "trial_outcome"],
                detector_name="test",
                detector_version="v0.5.0",
            )
        ],
        usability_flags=UsabilityFlags(
            has_trials=True,
            has_behavior=True,
            has_neural_data=True,
        ),
    )
    graph = build_graph_from_records([dataset], [])

    matches = find_datasets_for_experimental_design(
        graph,
        "q_learning_behavior_neural_experiment",
        min_score=0.5,
    )

    assert matches
    assert matches[0].dataset_id.endswith("000026")
    assert "minimum:has_trials" in matches[0].satisfied_requirements
