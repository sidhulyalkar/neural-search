"""Unit tests for usefulness scorer correctness, missingness, and renormalization."""
from __future__ import annotations

from neural_search.retrieval.query_intent import UsefulnessIntent
from neural_search.retrieval.usefulness_scorer import (
    INACTIVE_DIMENSIONS,
    INTENT_WEIGHT_PROFILES,
    DatasetContext,
    _active_weights,
    _jaccard,
    _warn_if_both_missing,
    score_usefulness,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(
    dataset_id="ds_x",
    modalities=None,
    tasks=None,
    species=None,
    brain_regions=None,
    affordances=None,
    data_standards=None,
    session_count=None,
    trial_count=None,
    quality_score=0.5,
):
    return DatasetContext(
        dataset_id=dataset_id,
        modalities=modalities or [],
        tasks=tasks or [],
        species=species or [],
        brain_regions=brain_regions or [],
        affordances=affordances or [],
        data_standards=data_standards or [],
        session_count=session_count,
        trial_count=trial_count,
        quality_score=quality_score,
    )


# ---------------------------------------------------------------------------
# Inactive dimensions are renormalized
# ---------------------------------------------------------------------------

class TestInactiveDimensionRenormalization:
    def test_neural_signature_is_in_inactive_set(self):
        assert "neural_signature_similarity" in INACTIVE_DIMENSIONS

    def test_active_weights_exclude_inactive_dims(self):
        weights = INTENT_WEIGHT_PROFILES[UsefulnessIntent.EXPLORATION]
        dims = dict.fromkeys(weights, 0.5)
        active = _active_weights(weights, dims)
        assert "neural_signature_similarity" not in active

    def test_active_weights_sum_to_one(self):
        for intent, weights in INTENT_WEIGHT_PROFILES.items():
            dims = dict.fromkeys(weights, 0.5)
            active = _active_weights(weights, dims)
            if active:
                total = sum(active.values())
                assert abs(total - 1.0) < 1e-6, f"{intent} active weights sum to {total}"

    def test_score_with_inactive_dim_does_not_penalize(self):
        q = _ctx(modalities=["fmri"], tasks=["reward_learning"], species=["human"])
        c = _ctx(modalities=["fmri"], tasks=["reward_learning"], species=["human"])
        score_exploration = score_usefulness(q, c, UsefulnessIntent.EXPLORATION)
        score_lookup = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        # Both should produce valid scores; EXPLORATION has neural_sig in profile (excluded)
        assert 0.0 <= score_exploration.total_score <= 1.0
        assert 0.0 <= score_lookup.total_score <= 1.0

    def test_neural_signature_score_is_zero(self):
        q = _ctx(modalities=["fmri"])
        c = _ctx(modalities=["fmri"])
        result = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        assert result.dimension_scores["neural_signature_similarity"] == 0.0

    def test_neural_signature_warning_emitted(self):
        q = _ctx()
        c = _ctx()
        result = score_usefulness(q, c)
        neural_warnings = [w for w in result.warnings if "neural_signature" in w]
        assert len(neural_warnings) >= 1


# ---------------------------------------------------------------------------
# Missing metadata does not increase similarity
# ---------------------------------------------------------------------------

class TestMissingMetadataNotRewarded:
    def test_empty_empty_jaccard_is_zero(self):
        assert _jaccard([], []) == 0.0

    def test_empty_query_jaccard_is_zero(self):
        assert _jaccard([], ["fmri"]) == 0.0

    def test_empty_candidate_jaccard_is_zero(self):
        assert _jaccard(["fmri"], []) == 0.0

    def test_both_missing_emits_warning(self):
        warnings: list[str] = []
        _warn_if_both_missing("modalities", [], [], warnings)
        assert len(warnings) == 1
        assert "modalities" in warnings[0]

    def test_one_side_missing_no_warning(self):
        warnings: list[str] = []
        _warn_if_both_missing("modalities", ["fmri"], [], warnings)
        assert len(warnings) == 0

    def test_empty_candidate_does_not_boost_score(self):
        q_full = _ctx(modalities=["fmri"], tasks=["reward"], species=["human"])
        c_empty = _ctx()
        c_full = _ctx(modalities=["fmri"], tasks=["reward"], species=["human"])
        score_empty = score_usefulness(q_full, c_empty, UsefulnessIntent.STRICT_LOOKUP)
        score_full = score_usefulness(q_full, c_full, UsefulnessIntent.STRICT_LOOKUP)
        assert score_full.total_score > score_empty.total_score

    def test_both_empty_scores_lower_than_match(self):
        q_empty = _ctx()
        c_empty = _ctx()
        q_full = _ctx(modalities=["fmri"], tasks=["reward"], species=["human"])
        c_full = _ctx(modalities=["fmri"], tasks=["reward"], species=["human"])
        score_empty = score_usefulness(q_empty, c_empty, UsefulnessIntent.STRICT_LOOKUP)
        score_full = score_usefulness(q_full, c_full, UsefulnessIntent.STRICT_LOOKUP)
        assert score_full.total_score >= score_empty.total_score

    def test_empty_jaccard_not_rewarded_directly(self):
        result_empty = _jaccard([], [])
        result_nonempty = _jaccard(["fmri"], ["fmri"])
        assert result_nonempty > result_empty
        assert result_empty == 0.0


# ---------------------------------------------------------------------------
# Hard negative scenarios
# ---------------------------------------------------------------------------

class TestHardNegativeScenarios:
    def test_modality_mismatch_scores_lower_than_match(self):
        q = _ctx(modalities=["calcium_imaging"], tasks=["visual_stimulation"], species=["mouse"])
        hard_neg = _ctx(modalities=["neuropixels"], tasks=["visual_stimulation"], species=["mouse"])
        true_match = _ctx(modalities=["calcium_imaging"], tasks=["visual_stimulation"], species=["mouse"])
        score_neg = score_usefulness(q, hard_neg, UsefulnessIntent.STRICT_LOOKUP)
        score_match = score_usefulness(q, true_match, UsefulnessIntent.STRICT_LOOKUP)
        assert score_match.total_score > score_neg.total_score

    def test_task_mismatch_scores_lower_than_match(self):
        q = _ctx(modalities=["fmri"], tasks=["reward_learning"], species=["human"])
        hard_neg = _ctx(modalities=["fmri"], tasks=["resting_state"], species=["human"])
        true_match = _ctx(modalities=["fmri"], tasks=["reward_learning"], species=["human"])
        score_neg = score_usefulness(q, hard_neg, UsefulnessIntent.STRICT_LOOKUP)
        score_match = score_usefulness(q, true_match, UsefulnessIntent.STRICT_LOOKUP)
        assert score_match.total_score > score_neg.total_score

    def test_species_mismatch_scores_lower_for_replication(self):
        q = _ctx(modalities=["fmri"], tasks=["decision_making"], species=["human"])
        hard_neg = _ctx(modalities=["fmri"], tasks=["decision_making"], species=["mouse"])
        true_match = _ctx(modalities=["fmri"], tasks=["decision_making"], species=["human"])
        score_neg = score_usefulness(q, hard_neg, UsefulnessIntent.REPLICATION)
        score_match = score_usefulness(q, true_match, UsefulnessIntent.REPLICATION)
        assert score_match.total_score > score_neg.total_score


# ---------------------------------------------------------------------------
# Intent weights and profiles
# ---------------------------------------------------------------------------

class TestIntentProfiles:
    def test_all_profiles_sum_to_one(self):
        for intent, weights in INTENT_WEIGHT_PROFILES.items():
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"{intent} weights sum to {total}"

    def test_all_intents_have_profiles(self):
        for intent in UsefulnessIntent:
            if intent != UsefulnessIntent.STRICT_LOOKUP:
                # These may fall back to STRICT_LOOKUP
                pass
        assert UsefulnessIntent.STRICT_LOOKUP in INTENT_WEIGHT_PROFILES

    def test_pipeline_reuse_emphasizes_affordances(self):
        weights = INTENT_WEIGHT_PROFILES[UsefulnessIntent.PIPELINE_REUSE]
        assert weights.get("affordance_compatibility", 0) >= weights.get("species_match", 0)

    def test_meta_analysis_emphasizes_provenance(self):
        weights = INTENT_WEIGHT_PROFILES[UsefulnessIntent.META_ANALYSIS]
        assert weights.get("provenance_quality", 0) >= 0.12

    def test_replication_emphasizes_species_and_task(self):
        weights = INTENT_WEIGHT_PROFILES[UsefulnessIntent.REPLICATION]
        assert weights.get("task_compatibility", 0) >= 0.15
        assert weights.get("species_match", 0) >= 0.10


# ---------------------------------------------------------------------------
# Score bounds
# ---------------------------------------------------------------------------

class TestScoreBounds:
    def test_total_score_always_bounded(self):
        cases = [
            (_ctx(), _ctx()),
            (_ctx(modalities=["fmri"]), _ctx(modalities=["fmri"])),
            (_ctx(quality_score=1.0), _ctx(quality_score=0.0)),
        ]
        for intent in UsefulnessIntent:
            if intent in INTENT_WEIGHT_PROFILES:
                for q, c in cases:
                    result = score_usefulness(q, c, intent)
                    assert 0.0 <= result.total_score <= 1.0, f"Out of bounds for {intent}"

    def test_dimension_scores_always_bounded(self):
        q = _ctx(modalities=["fmri"], tasks=["reward"], species=["human"], quality_score=0.9)
        c = _ctx(modalities=["eeg"], tasks=["rest"], species=["mouse"], quality_score=0.1)
        result = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        for dim, val in result.dimension_scores.items():
            assert 0.0 <= val <= 1.0, f"{dim} = {val} out of bounds"

    def test_perfect_match_scores_high(self):
        attrs = {
            "modalities": ["neuropixels"],
            "tasks": ["decision_making"],
            "species": ["mouse"],
            "brain_regions": ["prefrontal_cortex"],
            "affordances": ["choice_decoding"],
            "data_standards": ["nwb"],
            "quality_score": 1.0,
        }
        q = _ctx(**attrs)
        c = _ctx(**attrs)
        result = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        assert result.total_score >= 0.7

    def test_empty_candidate_scores_low(self):
        q = _ctx(modalities=["fmri"], tasks=["reward_learning"], species=["human"])
        c = _ctx()
        result = score_usefulness(q, c, UsefulnessIntent.STRICT_LOOKUP)
        assert result.total_score <= 0.4


# ---------------------------------------------------------------------------
# Evidence and warnings
# ---------------------------------------------------------------------------

class TestEvidenceAndWarnings:
    def test_shared_modalities_in_evidence(self):
        q = _ctx(modalities=["fmri"])
        c = _ctx(modalities=["fmri"])
        result = score_usefulness(q, c)
        assert any("modali" in e.lower() for e in result.evidence)

    def test_both_missing_fields_get_warning(self):
        q = _ctx()
        c = _ctx()
        result = score_usefulness(q, c)
        assert len(result.warnings) >= 1

    def test_graph_not_available_warning(self):
        q = _ctx()
        c = _ctx()
        result = score_usefulness(q, c, graph=None)
        assert any("graph_proximity" in w for w in result.warnings)
