import pytest
from pydantic import ValidationError

from neural_search.latent import (
    EventAlignedLatentSignature,
    LatentQCMetadata,
    LatentSignatureVector,
    LatentSourceFile,
)


def _signature(**overrides) -> EventAlignedLatentSignature:
    payload = {
        "dataset_id": "dataset:dandi:000026",
        "session_id": "sub-01_ses-01",
        "alignment_event": "choice",
        "pre_event_ms": 500.0,
        "post_event_ms": 1000.0,
        "vectors": [
            LatentSignatureVector(
                name="event_aligned_rate",
                dimensions=3,
                values=[0.1, 0.2, 0.3],
                units="zscore",
            )
        ],
        "source_files": [
            LatentSourceFile(
                path="sub-01/sub-01_ses-01.nwb",
                file_format="NWB",
                modality="extracellular_ephys",
            )
        ],
        "qc": LatentQCMetadata(quality_score=0.92),
        "extractor_name": "test_event_aligned_extractor",
        "extractor_version": "v0.9.0",
    }
    payload.update(overrides)
    return EventAlignedLatentSignature(**payload)


def test_event_aligned_latent_signature_exposes_compact_metadata():
    signature = _signature()

    assert signature.schema_version == "v0.9.0"
    assert signature.total_dimensions == 3
    assert signature.has_valid_qc
    assert signature.compact_metadata() == {
        "schema_version": "v0.9.0",
        "dataset_id": "dataset:dandi:000026",
        "session_id": "sub-01_ses-01",
        "alignment_event": "choice",
        "pre_event_ms": 500.0,
        "post_event_ms": 1000.0,
        "total_dimensions": 3,
        "vector_names": ["event_aligned_rate"],
        "source_files": ["sub-01/sub-01_ses-01.nwb"],
        "extractor_name": "test_event_aligned_extractor",
        "extractor_version": "v0.9.0",
        "qc_valid": True,
        "warnings": [],
    }


def test_latent_signature_vector_dimensions_must_match_values():
    with pytest.raises(ValidationError, match="values length must match dimensions"):
        LatentSignatureVector(
            name="bad_vector",
            dimensions=2,
            values=[0.1, 0.2, 0.3],
        )


def test_event_aligned_signature_requires_vector_source_and_nonzero_window():
    with pytest.raises(ValidationError, match="at least one latent vector"):
        _signature(vectors=[])

    with pytest.raises(ValidationError, match="at least one source file"):
        _signature(source_files=[])

    with pytest.raises(ValidationError, match="non-zero duration"):
        _signature(pre_event_ms=0.0, post_event_ms=0.0)


def test_qc_missing_signals_marks_signature_not_valid_for_search():
    signature = _signature(
        qc=LatentQCMetadata(
            valid=True,
            quality_score=0.5,
            missing_signals=["spike_times"],
            warnings=["spike times unavailable"],
        )
    )

    assert not signature.has_valid_qc
    assert signature.compact_metadata()["qc_valid"] is False
    assert signature.compact_metadata()["warnings"] == ["spike times unavailable"]
