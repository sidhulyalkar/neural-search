from neural_search.normalized import make_dataset_id
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord
from neural_search.source_quality import (
    assess_source_quality,
    summarize_source_quality,
)


def _standard(label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=f"label:data_standard:{label.lower()}",
        label=label,
        label_type="data_standard",
        confidence=0.9,
        evidence_text=label,
    )


def test_assess_source_quality_rewards_expected_standard_without_ranking_relevance():
    record = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("dandi", "000026"),
        source="dandi",
        source_id="000026",
        title="Mouse OFC ephys",
        url="https://dandiarchive.org/dandiset/000026",
        data_standards=[_standard("NWB")],
    )

    assessment = assess_source_quality(record)

    assert assessment.record_id == "dataset:dandi:000026"
    assert assessment.source == "dandi"
    assert assessment.trust_level == "high"
    assert assessment.quality_score == 0.97
    assert assessment.matched_standards == ("nwb",)
    assert assessment.warnings == ()


def test_assess_source_quality_warns_for_unknown_source_and_missing_url():
    record = {
        "dataset_id": "dataset:unknown:X1",
        "source": "mystery_archive",
        "source_id": "X1",
        "title": "Mystery dataset",
    }

    assessment = assess_source_quality(record)

    assert assessment.trust_level == "unknown"
    assert assessment.quality_score == 0.36
    assert assessment.warnings == (
        "record lacks a source URL",
        "source profile is not registered",
    )


def test_summarize_source_quality_counts_trust_levels_and_warnings():
    summary = summarize_source_quality(
        [
            {
                "dataset_id": "dataset:dandi:000001",
                "source": "dandi",
                "source_id": "000001",
                "url": "https://dandiarchive.org/dandiset/000001",
                "data_standards": ["NWB"],
            },
            {
                "dataset_id": "dataset:demo:DEMO",
                "source": "demo",
                "source_id": "DEMO",
            },
            {
                "dataset_id": "dataset:mystery:X1",
                "source": "mystery",
                "source_id": "X1",
            },
        ]
    )

    assert summary["record_count"] == 3
    assert summary["trust_level_counts"] == {"high": 1, "low": 1, "unknown": 1}
    assert summary["source_counts"] == {"dandi": 1, "demo": 1, "mystery": 1}
    assert summary["warning_count"] == 2
