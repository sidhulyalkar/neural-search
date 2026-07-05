from neural_search.normalized import make_dataset_id, make_evidence_label_id
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord, UsabilityFlags
from neural_search.spectral.eligibility import detect_aperiodic_eligibility


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
        "title": "Aperiodic eligibility test",
    }
    payload.update(overrides)
    return NormalizedDatasetRecord(**payload)


def test_high_support_requires_modality_raw_data_sampling_rate_and_channel_metadata():
    record = _record(
        modalities=[_label("modality", "electrophysiology")],
        data_standards=[_label("data_standard", "nwb")],
        description="Includes a full electrode/channel table.",
        usability_flags=UsabilityFlags(has_raw_data=True),
    )

    result = detect_aperiodic_eligibility(record)

    assert result.support_level == "high"
    assert result.confidence >= 0.8
    assert result.compatible_modality is True
    assert result.sampling_rate_likely_available is True
    assert result.channel_or_probe_metadata_present is True


def test_medium_support_with_processed_data_only():
    record = _record(
        modalities=[_label("modality", "eeg")],
        usability_flags=UsabilityFlags(has_processed_data=True),
    )

    result = detect_aperiodic_eligibility(record)

    assert result.support_level == "medium"
    assert 0.3 < result.confidence < 0.8
    assert result.compatible_modality is True


def test_medium_support_with_compatible_modality_and_missing_metadata():
    record = _record(
        modalities=[_label("modality", "lfp")],
        missing_fields=["channel_metadata"],
    )

    result = detect_aperiodic_eligibility(record)

    assert result.support_level == "medium"


def test_low_support_with_compatible_modality_but_unclear_data_access():
    record = _record(modalities=[_label("modality", "ecog")])

    result = detect_aperiodic_eligibility(record)

    assert result.support_level == "low"
    assert result.confidence < 0.5


def test_unsupported_for_fmri_only():
    record = _record(modalities=[_label("modality", "fmri")])

    result = detect_aperiodic_eligibility(record)

    assert result.support_level == "unsupported"
    assert result.confidence > 0.8


def test_unsupported_for_behavior_tracking_only():
    record = _record(
        modalities=[_label("modality", "behavior_tracking")],
        usability_flags=UsabilityFlags(has_behavior=True),
    )

    result = detect_aperiodic_eligibility(record)

    assert result.support_level == "unsupported"


def test_unsupported_for_anatomical_only():
    record = _record(modalities=[_label("modality", "structural_mri")])

    result = detect_aperiodic_eligibility(record)

    assert result.support_level == "unsupported"


def test_unknown_when_no_modality_evidence_at_all():
    record = _record()

    result = detect_aperiodic_eligibility(record)

    assert result.support_level == "unknown"


def test_eligibility_dataset_id_matches_record():
    record = _record(modalities=[_label("modality", "eeg")])

    result = detect_aperiodic_eligibility(record)

    assert result.dataset_id == record.dataset_id
