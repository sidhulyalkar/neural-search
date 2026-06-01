from neural_search.normalized import (
    make_dataset_id,
    make_evidence_label_id,
    make_paper_id,
)
from neural_search.schemas import (
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
)
from neural_search.scientific_labels import (
    enrich_record_with_scientific_labels,
    extract_scientific_labels,
)


def _label(label_type: str, label_id: str, confidence: float = 0.95) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label_id),
        label=label_id.replace("_", " "),
        label_type=label_type,
        confidence=confidence,
        evidence_text=label_id,
        source_field="metadata",
        source_value=label_id,
        extractor_name="test",
        extractor_version="v0.3.0",
    )


def _dataset(**overrides) -> NormalizedDatasetRecord:
    payload = {
        "dataset_id": make_dataset_id("dandi", "000001"),
        "source": "dandi",
        "source_id": "000001",
        "title": "Untitled dataset",
    }
    payload.update(overrides)
    return NormalizedDatasetRecord(**payload)


def _ids(labels: list[EvidenceLabel], label_type: str) -> set[str]:
    return {
        label.id.removeprefix(f"label:{label_type}:")
        for label in labels
        if label.label_type == label_type
    }


def test_exact_metadata_match_keeps_high_confidence():
    record = _dataset(modalities=[_label("modality", "neuropixels", 0.96)])

    labels = extract_scientific_labels(record)

    match = next(label for label in labels if label.id == "label:modality:neuropixels")
    assert match.confidence == 0.96
    assert match.source_field == "metadata"


def test_synonym_match_produces_medium_high_confidence():
    record = _dataset(title="Two-photon imaging during go/no-go behavior")

    labels = extract_scientific_labels(record, include_existing=False)

    match = next(label for label in labels if label.id == "label:modality:calcium_imaging")
    assert 0.75 <= match.confidence <= 0.9
    assert match.evidence_text
    assert match.source_field == "title"


def test_free_text_description_match_produces_moderate_confidence():
    record = _dataset(description="This dataset includes fiber photometry during reward delivery.")

    labels = extract_scientific_labels(record, include_existing=False)

    match = next(label for label in labels if label.id == "label:modality:fiber_photometry")
    assert 0.6 <= match.confidence <= 0.8
    assert match.source_field == "description"


def test_duplicate_labels_are_merged_with_highest_confidence():
    record = _dataset(
        title="Neuropixels recordings",
        modalities=[_label("modality", "neuropixels", 0.97)],
    )

    labels = [
        label for label in extract_scientific_labels(record) if label.id == "label:modality:neuropixels"
    ]

    assert len(labels) == 1
    assert labels[0].confidence == 0.97


def test_false_positive_traps_are_rejected():
    labels = extract_scientific_labels(
        _dataset(title="OFC recordings with reward", description="EEG resting state extracellular electrophysiology"),
        include_existing=False,
    )

    assert "reversal_learning" not in _ids(labels, "task")
    assert "q_learning_modeling" not in _ids(labels, "analysis_goal")
    assert "bci_decoding" not in _ids(labels, "analysis_goal")
    assert "sleep_stage_classification" not in _ids(labels, "analysis_goal")
    assert "neuropixels" not in _ids(labels, "modality")
    assert "electrophysiology" in _ids(labels, "modality")


def test_extraction_works_for_paper_records():
    paper = NormalizedPaperRecord(
        paper_id=make_paper_id("openalex", "W123"),
        source="openalex",
        source_id="W123",
        title="Q-learning models for reversal learning",
        abstract="Neuropixels recordings during reward prediction error tasks.",
    )

    enriched = enrich_record_with_scientific_labels(paper)

    assert isinstance(enriched, NormalizedPaperRecord)
    ids = {label.id for label in enriched.extracted_labels}
    assert "label:analysis_goal:q_learning_modeling" in ids
    assert "label:task:reversal_learning" in ids


def test_missing_fields_do_not_crash_extraction():
    labels = extract_scientific_labels(_dataset(description=None), include_existing=False)

    assert any(label.id == "label:data_standard:dandi" for label in labels)
