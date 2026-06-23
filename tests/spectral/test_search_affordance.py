import pytest

from neural_search.analysis_affordances import detect_analysis_affordances
from neural_search.normalized import make_dataset_id, make_evidence_label_id
from neural_search.ontology import match_affordances
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord, UsabilityFlags
from neural_search.spectral.search_features import (
    APERIODIC_AFFORDANCE_ID,
    SPECTRAL_TRIGGER_TERMS,
    explain_spectral_search_match,
    query_matches_spectral_affordance,
)


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
        "dataset_id": make_dataset_id("dandi", "000002"),
        "source": "dandi",
        "source_id": "000002",
        "title": "Spectral search affordance test",
    }
    payload.update(overrides)
    return NormalizedDatasetRecord(**payload)


@pytest.mark.parametrize(
    "query",
    [
        "datasets with aperiodic activity",
        "looking for 1/f spectral data",
        "spectral slope analysis",
        "power law neural data",
        "FOOOF analysis ready datasets",
        "specparam reanalysis",
        "IRASA decomposition",
        "E/I balance estimation",
        "intrinsic timescale of neural activity",
    ],
)
def test_spectral_trigger_terms_match_affordance(query: str):
    assert query_matches_spectral_affordance(query)
    matches = match_affordances(query)
    assert any(match.id == APERIODIC_AFFORDANCE_ID for match in matches)


def test_unrelated_query_does_not_match_spectral_affordance():
    assert not query_matches_spectral_affordance("trial-averaged calcium imaging responses to reward")


def test_all_documented_trigger_terms_are_individually_matchable():
    for term in SPECTRAL_TRIGGER_TERMS:
        assert query_matches_spectral_affordance(f"datasets about {term}"), f"trigger term failed: {term}"


def test_detect_analysis_affordances_includes_aperiodic_id():
    record = _record(modalities=[_label("modality", "electrophysiology")])

    affordances = detect_analysis_affordances(record)

    ids = {affordance.analysis_id for affordance in affordances}
    assert APERIODIC_AFFORDANCE_ID in ids


def test_explain_spectral_search_match_boosts_eligible_dataset():
    record = _record(
        modalities=[_label("modality", "electrophysiology")],
        data_standards=[_label("data_standard", "nwb")],
        description="Channel and electrode metadata included.",
        usability_flags=UsabilityFlags(has_raw_data=True),
    )

    explanation = explain_spectral_search_match("aperiodic 1/f spectral slope", record)

    assert explanation["boost_applicable"] is True
    assert explanation["matched_query_affordance"] == APERIODIC_AFFORDANCE_ID
    assert explanation["eligibility_support_level"] in ("high", "medium")
    assert explanation["explanation"]


def test_explain_spectral_search_match_does_not_boost_ineligible_dataset():
    record = _record(modalities=[_label("modality", "fmri")])

    explanation = explain_spectral_search_match("aperiodic 1/f spectral slope", record)

    assert explanation["boost_applicable"] is False
    assert explanation["eligibility_support_level"] == "unsupported"
