from neural_search.analysis_affordances import detect_analysis_affordances
from neural_search.normalized import make_dataset_id, make_evidence_label_id
from neural_search.reports.corpus_report import summarize_corpus
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord, UsabilityFlags


def _label(label_type: str, label_id: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label_id),
        label=label_id.replace("_", " "),
        label_type=label_type,
        confidence=0.9,
        evidence_text=label_id,
        source_field="test",
        source_value=label_id,
        extractor_name="test",
        extractor_version="v0.3.0",
    )


def _record(**overrides) -> NormalizedDatasetRecord:
    payload = {
        "dataset_id": make_dataset_id("dandi", "000001"),
        "source": "dandi",
        "source_id": "000001",
        "title": "Affordance test",
    }
    payload.update(overrides)
    return NormalizedDatasetRecord(**payload)


def _affordance(record: NormalizedDatasetRecord, analysis_id: str):
    return next(item for item in detect_analysis_affordances(record) if item.analysis_id == analysis_id)


def test_event_aligned_activity_high_with_neural_data_and_event_timestamps():
    record = _record(
        modalities=[_label("modality", "neuropixels")],
        usability_flags=UsabilityFlags(has_neural_data=True, has_event_timestamps=True),
    )

    result = _affordance(record, "event_aligned_activity")

    assert result.support_level == "high"
    assert result.confidence >= 0.85
    assert "event_timestamps" in result.required_fields_present


def test_q_learning_high_only_with_choice_reward_and_trial_outcome():
    high = _record(
        behavioral_events=[
            _label("behavioral_event", "choice"),
            _label("behavioral_event", "reward"),
            _label("behavioral_event", "trial_outcome"),
        ]
    )
    reward_only = _record(behavioral_events=[_label("behavioral_event", "reward")])

    assert _affordance(high, "q_learning_modeling").support_level == "high"
    low = _affordance(reward_only, "q_learning_modeling")
    assert low.support_level == "low"
    assert low.confidence < 0.35
    assert "choice" in low.missing_fields


def test_motor_decoding_not_high_from_motor_cortex_alone():
    record = _record(brain_regions=[_label("brain_region", "motor_cortex")])

    result = _affordance(record, "motor_decoding")

    assert result.support_level == "low"
    assert result.confidence < 0.35


def test_speech_decoding_not_high_from_auditory_cortex_alone():
    record = _record(brain_regions=[_label("brain_region", "auditory_cortex")])

    result = _affordance(record, "speech_decoding")

    assert result.support_level == "low"
    assert result.confidence < 0.35


def test_sleep_stage_classification_not_high_from_resting_state_alone():
    record = _record(analysis_goals=[_label("subject_state", "resting_state")])

    result = _affordance(record, "sleep_stage_classification")

    assert result.support_level == "low"
    assert result.confidence < 0.35


def test_brain_behavior_alignment_high_with_neural_and_behavior_flags():
    record = _record(
        modalities=[_label("modality", "calcium_imaging")],
        usability_flags=UsabilityFlags(has_neural_data=True, has_behavior=True),
    )

    result = _affordance(record, "brain_behavior_alignment")

    assert result.support_level == "high"
    assert "behavior" in result.required_fields_present


def test_bci_decoding_not_inferred_from_eeg_alone():
    record = _record(modalities=[_label("modality", "eeg")])

    result = _affordance(record, "bci_decoding")

    assert result.support_level == "low"
    assert result.confidence < 0.35
    assert "bci_context" in result.missing_fields


def test_missing_fields_are_reported_for_event_alignment_without_events():
    record = _record(
        modalities=[_label("modality", "lfp")],
        usability_flags=UsabilityFlags(has_neural_data=True),
    )

    result = _affordance(record, "event_aligned_activity")

    assert result.support_level == "low"
    assert "event_timestamps" in result.missing_fields


def test_support_level_and_confidence_are_consistent():
    record = _record(
        modalities=[_label("modality", "ecog")],
        tasks=[_label("task", "speech_production")],
        behavioral_events=[_label("behavioral_event", "speech_onset")],
        usability_flags=UsabilityFlags(has_neural_data=True),
    )

    result = _affordance(record, "speech_decoding")

    assert result.support_level == "high"
    assert result.confidence >= 0.8


def test_corpus_summary_counts_analysis_affordances():
    record = _record(
        modalities=[_label("modality", "neuropixels")],
        usability_flags=UsabilityFlags(has_neural_data=True, has_event_timestamps=True),
    )
    enriched = record.model_copy(
        update={"analysis_affordances": detect_analysis_affordances(record)},
        deep=True,
    )

    summary = summarize_corpus([enriched])

    assert summary["label_counts"]["analysis_affordances"]["event_aligned_activity"] == 1
