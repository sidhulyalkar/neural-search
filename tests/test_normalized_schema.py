from pathlib import Path

import pytest
from pydantic import ValidationError

from neural_search.normalized import (
    make_dataset_id,
    make_evidence_label_id,
    make_paper_id,
    read_json,
    read_jsonl,
    record_from_dict,
    record_to_dict,
    write_json,
    write_jsonl,
)
from neural_search.schemas import (
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
    UsabilityFlags,
)

FIXTURES = Path(__file__).parent / "fixtures" / "normalized"


def _label(label_type: str = "modality", label: str = "Neuropixels") -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label),
        label=label,
        label_type=label_type,
        confidence=0.9,
        evidence_text=label,
        source_field="title",
        source_value=f"{label} recordings",
        extractor_name="test",
        extractor_version="v0.3.0",
    )


def _dataset(**overrides) -> NormalizedDatasetRecord:
    payload = {
        "dataset_id": make_dataset_id("dandi", "000026"),
        "source": "dandi",
        "source_id": "000026",
        "title": "Mouse Neuropixels decision task",
        "modalities": [_label()],
        "usability_flags": UsabilityFlags(has_neural_data=True),
        "created_at": "2026-05-23T00:00:00+00:00",
        "extractor_version": "v0.3.0",
    }
    payload.update(overrides)
    return NormalizedDatasetRecord(**payload)


def _paper(**overrides) -> NormalizedPaperRecord:
    payload = {
        "paper_id": make_paper_id("openalex", "W123456789"),
        "source": "openalex",
        "source_id": "W123456789",
        "title": "Neuropixels recordings during decision making",
        "authors": ["Demo Author"],
        "extracted_labels": [_label()],
        "created_at": "2026-05-23T00:00:00+00:00",
        "extractor_version": "v0.3.0",
    }
    payload.update(overrides)
    return NormalizedPaperRecord(**payload)


def test_evidence_label_accepts_valid_confidence():
    label = _label()

    assert label.confidence == 0.9
    assert label.label_type == "modality"


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_evidence_label_rejects_out_of_range_confidence(confidence):
    with pytest.raises(ValidationError):
        EvidenceLabel(
            id="label:modality:bad",
            label="bad",
            label_type="modality",
            confidence=confidence,
            extractor_name="test",
            extractor_version="v0.3.0",
        )


def test_label_type_is_normalized_to_predictable_value():
    label = _label("Data Standard", "NWB")

    assert label.label_type == "data_standard"


def test_normalized_dataset_record_serializes_and_deserializes(tmp_path):
    record = _dataset(description=None, url=None, raw_payload_path=None)
    path = write_json(record, tmp_path / "dataset.json")

    loaded = read_json(path)

    assert loaded == record
    assert record_from_dict(record_to_dict(record)) == record


def test_normalized_paper_record_serializes_and_deserializes(tmp_path):
    record = _paper(abstract=None, year=None, raw_payload_path=None)
    path = write_json(record, tmp_path / "paper.json")

    loaded = read_json(path)

    assert loaded == record
    assert record_from_dict(record_to_dict(record)) == record


def test_jsonl_roundtrip_mixed_normalized_records(tmp_path):
    records = [_dataset(), _paper()]
    path = write_jsonl(records, tmp_path / "records.jsonl")

    assert read_jsonl(path) == records


def test_stable_id_helpers_are_deterministic_and_safe():
    assert make_dataset_id("DANDI", "000026") == make_dataset_id("dandi", "000026")
    assert make_dataset_id("dandi", "000026") == "dataset:dandi:000026"
    assert make_dataset_id("openneuro", "ds004148") == "dataset:openneuro:ds004148"
    assert make_paper_id("openalex", "W123456789") == "paper:openalex:W123456789"
    assert make_evidence_label_id("modality", "Neuropixels") == "label:modality:neuropixels"
    assert make_paper_id("openalex", "https://openalex.org/W123") == (
        "paper:openalex:https_openalex.org_W123"
    )


@pytest.mark.parametrize(
    "field",
    ["dataset_id", "source", "source_id", "title"],
)
def test_empty_dataset_required_fields_fail_validation(field):
    payload = record_to_dict(_dataset())
    payload[field] = "  "

    with pytest.raises(ValidationError):
        NormalizedDatasetRecord.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["paper_id", "source", "source_id", "title"],
)
def test_empty_paper_required_fields_fail_validation(field):
    payload = record_to_dict(_paper())
    payload[field] = "  "

    with pytest.raises(ValidationError):
        NormalizedPaperRecord.model_validate(payload)


def test_missing_optional_metadata_is_allowed():
    dataset = _dataset(description=None, url=None, raw_payload_path=None)
    paper = _paper(abstract=None, doi=None, url=None, year=None, raw_payload_path=None)

    assert dataset.description is None
    assert paper.abstract is None


def test_fixture_records_load_successfully():
    records = [
        read_json(FIXTURES / "dandi_dataset_minimal.json"),
        read_json(FIXTURES / "openneuro_dataset_minimal.json"),
        read_json(FIXTURES / "openalex_paper_minimal.json"),
    ]

    assert isinstance(records[0], NormalizedDatasetRecord)
    assert isinstance(records[1], NormalizedDatasetRecord)
    assert isinstance(records[2], NormalizedPaperRecord)
