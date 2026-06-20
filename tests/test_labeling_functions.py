"""Tests for all 13 deterministic labeling functions."""
from __future__ import annotations

from neural_search.eval.evidence import DatasetEvidence, LFVote, PairEvidence, QuerySpec
from neural_search.eval.labeling_functions import (
    lf_hard_negative,
    lf_license_reusable,
    lf_metadata_completeness,
    lf_raw_data_available,
    lf_required_modality,
    lf_species_constraint,
    run_all_lfs,
)


def _make_pair(
    required_modalities=None,
    preferred_modalities=None,
    required_species=None,
    hard_negatives=None,
    intent="META_ANALYSIS",
    analysis_affordances=None,
    task_constraints=None,
    data_level_requirements=None,
    brain_regions=None,
    d_species=None,
    d_modalities=None,
    d_tasks=None,
    d_regions=None,
    d_license=None,
    d_raw=False,
    d_completeness=0.5,
    d_data_levels=None,
    d_data_standards=None,
    d_has_behavior=False,
    d_has_trials=False,
    d_title="A dataset",
    d_description="",
) -> PairEvidence:
    q = QuerySpec(
        query_id="q1", query_text="test", intent=intent, scientific_goal="x",
        required_modalities=required_modalities or [],
        preferred_modalities=preferred_modalities or [],
        required_species=required_species or [],
        hard_negatives=hard_negatives or [],
        analysis_affordances=analysis_affordances or [],
        task_constraints=task_constraints or [],
        data_level_requirements=data_level_requirements or [],
        brain_regions=brain_regions or [],
    )
    d = DatasetEvidence(
        record_id="dandi:1", source="dandi", title=d_title,
        description=d_description,
        species=d_species or [], modalities=d_modalities or [],
        tasks=d_tasks or [], regions=d_regions or [],
        license=d_license, raw_data_available=d_raw,
        metadata_completeness=d_completeness,
        data_levels=d_data_levels or [],
        data_standards=d_data_standards or [],
        has_behavior=d_has_behavior,
        has_trials=d_has_trials,
        doi=None, url=None,
    )
    return PairEvidence(query_id="q1", record_id="dandi:1", query=q, dataset=d)


class TestLfHardNegative:
    def test_no_hard_negatives_abstains(self):
        pair = _make_pair(hard_negatives=[])
        vote = lf_hard_negative(pair)
        assert vote.abstain is True

    def test_matching_hard_negative_votes_zero(self):
        pair = _make_pair(
            hard_negatives=["resting-state fMRI with reward words in description"],
            d_title="Resting state fMRI study",
            d_modalities=["fmri"],
        )
        vote = lf_hard_negative(pair)
        assert vote.abstain is False
        assert vote.label == 0
        assert vote.confidence >= 0.90

    def test_non_matching_hard_negative_abstains(self):
        pair = _make_pair(
            hard_negatives=["resting-state fMRI with reward words"],
            d_title="Mouse neuropixels visual cortex",
            d_modalities=["neuropixels"],
            d_species=["mouse"],
        )
        vote = lf_hard_negative(pair)
        assert vote.abstain is True


class TestLfRequiredModality:
    def test_full_match_votes_3(self):
        pair = _make_pair(required_modalities=["fmri"], d_modalities=["fmri"])
        vote = lf_required_modality(pair)
        assert vote.label == 3
        assert vote.confidence >= 0.85

    def test_no_required_abstains(self):
        pair = _make_pair(required_modalities=[])
        vote = lf_required_modality(pair)
        assert vote.abstain is True

    def test_modality_mismatch_votes_zero(self):
        pair = _make_pair(
            required_modalities=["fmri"],
            d_modalities=["extracellular_ephys"],
        )
        vote = lf_required_modality(pair)
        assert vote.label == 0
        assert vote.confidence >= 0.80

    def test_partial_match_votes_2(self):
        pair = _make_pair(
            required_modalities=["fmri", "meg"],
            d_modalities=["fmri"],
        )
        vote = lf_required_modality(pair)
        assert vote.label == 2


class TestLfSpeciesConstraint:
    def test_species_match_votes_3(self):
        pair = _make_pair(required_species=["mouse"], d_species=["mouse"])
        vote = lf_species_constraint(pair)
        assert vote.label == 3

    def test_species_mismatch_votes_zero(self):
        pair = _make_pair(required_species=["human"], d_species=["mouse"])
        vote = lf_species_constraint(pair)
        assert vote.label == 0

    def test_no_constraint_abstains(self):
        pair = _make_pair(required_species=[])
        vote = lf_species_constraint(pair)
        assert vote.abstain is True


class TestLfLicenseReusable:
    def test_cc_by_votes_high(self):
        pair = _make_pair(d_license="CC-BY-4.0")
        vote = lf_license_reusable(pair)
        assert vote.label >= 2
        assert vote.abstain is False

    def test_no_license_abstains(self):
        pair = _make_pair(d_license=None)
        vote = lf_license_reusable(pair)
        assert vote.abstain is True

    def test_restrictive_license_votes_low(self):
        pair = _make_pair(d_license="All rights reserved")
        vote = lf_license_reusable(pair)
        assert vote.label <= 1


class TestLfRawDataAvailable:
    def test_raw_available_votes_positive(self):
        pair = _make_pair(d_raw=True)
        vote = lf_raw_data_available(pair)
        assert vote.label >= 2

    def test_no_raw_votes_lower(self):
        pair = _make_pair(d_raw=False)
        vote = lf_raw_data_available(pair)
        assert vote.label <= 2


class TestLfMetadataCompleteness:
    def test_high_completeness_votes_high(self):
        pair = _make_pair(d_completeness=0.9)
        vote = lf_metadata_completeness(pair)
        assert vote.label >= 2

    def test_low_completeness_abstains_or_votes_low(self):
        pair = _make_pair(d_completeness=0.1)
        vote = lf_metadata_completeness(pair)
        assert vote.abstain or vote.label <= 1


class TestRunAllLfs:
    def test_returns_13_votes(self):
        pair = _make_pair(
            required_modalities=["fmri"],
            required_species=["human"],
            d_modalities=["fmri"],
            d_species=["human"],
            d_license="CC-BY-4.0",
            d_raw=True,
            d_completeness=0.8,
        )
        votes = run_all_lfs(pair)
        assert len(votes) == 13

    def test_all_votes_are_lf_vote(self):
        pair = _make_pair()
        votes = run_all_lfs(pair)
        assert all(isinstance(v, LFVote) for v in votes)
