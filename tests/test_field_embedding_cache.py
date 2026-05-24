import pytest

from neural_search.embeddings import (
    HashingEmbeddingProvider,
    build_field_embedding_records,
    field_texts_for_record,
    read_field_embedding_cache,
    write_field_embedding_cache,
)
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


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label),
        label=label,
        label_type=label_type,
        confidence=0.9,
        evidence_text=label,
        extractor_name="test",
        extractor_version="v0.4.0",
    )


def _dataset(**overrides) -> NormalizedDatasetRecord:
    payload = {
        "dataset_id": make_dataset_id("dandi", "000026"),
        "source": "dandi",
        "source_id": "000026",
        "title": "Mouse Go NoGo calcium imaging",
        "description": "Lick events and reward omission trials in mPFC.",
        "tasks": [_label("task", "Go NoGo")],
        "behavioral_events": [_label("behavioral_event", "lick")],
        "modalities": [_label("modality", "calcium imaging")],
        "brain_regions": [_label("brain_region", "mPFC")],
        "data_standards": [_label("data_standard", "NWB")],
    }
    payload.update(overrides)
    return NormalizedDatasetRecord(**payload)


def _paper(**overrides) -> NormalizedPaperRecord:
    payload = {
        "paper_id": make_paper_id("openalex", "W123"),
        "source": "openalex",
        "source_id": "W123",
        "title": "Go NoGo calcium imaging",
        "abstract": "Reward omission and licking in mouse calcium imaging.",
        "extracted_labels": [_label("modality", "calcium imaging")],
    }
    payload.update(overrides)
    return NormalizedPaperRecord(**payload)


def test_field_text_selection_includes_expected_dataset_and_paper_fields():
    dataset_fields = field_texts_for_record(_dataset(description=None))
    paper_fields = field_texts_for_record(_paper(abstract=None))

    assert "description" not in dataset_fields
    assert dataset_fields["tasks"] == "Go NoGo label:task:go_nogo"
    assert "calcium imaging" in dataset_fields["combined_scientific_summary"]
    assert "abstract" not in paper_fields
    assert "extracted_labels" in paper_fields


def test_field_embedding_cache_roundtrip_and_metadata_validation(tmp_path):
    provider = HashingEmbeddingProvider(dimensions=12)
    records = build_field_embedding_records(
        [_dataset(), _paper()],
        provider,
        created_at="2026-05-24T00:00:00+00:00",
    )
    path = write_field_embedding_cache(records, tmp_path / "embeddings.jsonl")

    loaded = read_field_embedding_cache(
        path,
        expected_provider_name="hashing",
        expected_model_name="signed-token-hashing-12",
        expected_dimension=12,
        expected_normalize=True,
    )

    assert loaded == records
    assert {record.record_type for record in loaded} == {"dataset", "paper"}
    assert all(len(record.embedding) == 12 for record in loaded)

    with pytest.raises(ValueError, match="metadata mismatch"):
        read_field_embedding_cache(path, expected_dimension=8)
