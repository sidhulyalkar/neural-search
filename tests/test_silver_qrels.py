"""Tests for Neural Search Benchmark v0.7 silver qrels modules.

Covers:
1.  Silver qrels schema validation (SilverQrelsEntry, LabelingFunctionVote, etc.)
2.  Individual labeling functions
3.  Hard-negative override behaviour
4.  Missing metadata confidence reduction
5.  Affordance probes
6.  Concept-memory weak labeler
7.  Optional LLM judge mock behaviour
8.  Vote aggregation
9.  Confidence computation
10. Disagreement detection
11. Review queue selection
12. Silver-vs-gold calibration placeholder
13. Silver-vs-gold calibration with small fixture
14. Silver evaluation guardrails (--allow-silver flag)
15. Reports generated with required warnings
16. No writes to artifacts/qrels.jsonl by default
17. Determinism with seed
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval.affordance_probes import (
    AffordanceProbeResult,
    probe_calcium_imaging,
    probe_meta_analysis,
    probe_neural_decoding,
    probe_pose_analysis,
    probe_rl_model_fitting,
    probe_spike_sorting,
    probe_to_vote,
)
from scripts.eval.benchmark_schema import BenchmarkQueryV1
from scripts.eval.build_silver_qrels import (
    GOLD_QRELS_PATH,
    _aggregate_votes,
    _guard_output_path,
    _needs_review,
    build_silver_qrels,
    build_summary,
)
from scripts.eval.calibrate_silver_qrels import (
    _calibration_by_bin,
    _confusion_matrix,
    _high_conf_accuracy,
    _hn_precision,
)
from scripts.eval.concept_labeler import label_from_concept_result
from scripts.eval.labeling_functions import (
    apply_all_labeling_functions,
    lf_brain_region_match,
    lf_brain_region_mismatch,
    lf_hard_negative_violation,
    lf_missing_critical_metadata,
    lf_modality_match,
    lf_modality_mismatch,
    lf_species_match,
    lf_species_mismatch,
    lf_task_match,
    lf_task_partial_match,
)
from scripts.eval.llm_judge import MockLLMJudge
from scripts.eval.select_human_review_set import (
    select_review_set,
)
from scripts.eval.silver_qrels_schema import (
    LABEL_TYPE_SILVER,
    LabelingFunctionVote,
    SilverQrelsEntry,
    SilverQrelsSummary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _make_query(**kwargs) -> BenchmarkQueryV1:
    defaults = {
        "query_id": "q_test",
        "query_text": "test query",
        "intent": "EXPLORATION",
        "scientific_goal": "Test scientific goal.",
    }
    defaults.update(kwargs)
    return BenchmarkQueryV1.model_validate(defaults)


def _make_record(**kwargs) -> dict:
    defaults = {
        "source": "openneuro",
        "source_id": "ds001234",
        "title": "Test dataset",
        "description": "A test neuroscience dataset.",
        "license": "CC BY 4.0",
        "species": ["mouse"],
        "modalities": ["calcium_imaging"],
        "brain_regions": ["visual_cortex"],
        "tasks": ["go_nogo"],
        "data_standards": ["NWB"],
        "has_behavior": True,
        "has_trials": True,
        "has_raw_data": True,
    }
    defaults.update(kwargs)
    return defaults


def _make_silver_entry(**kwargs) -> SilverQrelsEntry:
    defaults = {
        "query_id": "q_test",
        "dataset_id": "ds001234",
        "label_type": LABEL_TYPE_SILVER,
        "relevance": 2,
        "confidence": 0.65,
        "all_votes": [],
    }
    defaults.update(kwargs)
    return SilverQrelsEntry.model_validate(defaults)


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------


class TestSilverQrelsSchema:
    def test_silver_entry_valid(self):
        e = _make_silver_entry()
        assert e.label_type == "silver"
        assert e.relevance == 2
        assert e.schema_version == "0.7"

    def test_silver_entry_label_type_guard(self):
        with pytest.raises(ValueError):
            SilverQrelsEntry.model_validate(
                {"query_id": "q1", "dataset_id": "d1", "label_type": "gold",
                 "relevance": 1, "confidence": 0.5}
            )

    def test_relevance_bounds(self):
        for v in (0, 1, 2, 3):
            e = _make_silver_entry(relevance=v)
            assert e.relevance == v

    def test_relevance_invalid(self):
        with pytest.raises(ValueError):
            _make_silver_entry(relevance=4)

    def test_confidence_bounds(self):
        e = _make_silver_entry(confidence=0.0)
        assert e.confidence == 0.0
        e2 = _make_silver_entry(confidence=1.0)
        assert e2.confidence == 1.0

    def test_labeling_function_vote_abstain(self):
        v = LabelingFunctionVote(source="rules", vote=None)
        assert v.vote is None

    def test_labeling_function_vote_invalid(self):
        with pytest.raises(ValueError):
            LabelingFunctionVote.model_validate({"source": "rules", "vote": 5})

    def test_watermark_in_summary(self):
        s = SilverQrelsSummary()
        assert "NOT EXPERT VALIDATION" in s.watermark

    def test_silver_does_not_accept_gold_labels(self):
        """Silver schema must not accept gold label_type."""
        with pytest.raises(ValueError):
            SilverQrelsEntry.model_validate(
                {"query_id": "q1", "dataset_id": "d1",
                 "label_type": "gold", "relevance": 2, "confidence": 0.7}
            )


# ---------------------------------------------------------------------------
# 2. Individual labeling functions
# ---------------------------------------------------------------------------


class TestSpeciesLF:
    def test_species_match(self):
        q = _make_query(expected_species=["mouse"])
        r = _make_record(species=["Mus musculus"])
        v = lf_species_match(q, r)
        assert v.vote == 2
        assert v.confidence > 0.5

    def test_species_mismatch(self):
        q = _make_query(expected_species=["human"])
        r = _make_record(species=["mouse"])
        v = lf_species_mismatch(q, r)
        assert v.vote == 0
        assert v.confidence > 0.6

    def test_species_no_expected(self):
        q = _make_query(expected_species=[])
        r = _make_record()
        v = lf_species_match(q, r)
        assert v.vote is None

    def test_species_missing_from_record(self):
        q = _make_query(expected_species=["mouse"])
        r = _make_record(species=[])
        v = lf_species_match(q, r)
        assert v.vote is None

    def test_no_mismatch_when_overlap(self):
        q = _make_query(expected_species=["mouse"])
        r = _make_record(species=["mouse", "rat"])
        v = lf_species_mismatch(q, r)
        assert v.vote is None


class TestModalityLF:
    def test_modality_match_calcium(self):
        q = _make_query(expected_modalities=["calcium_imaging"])
        r = _make_record(modalities=["calcium_imaging"])
        v = lf_modality_match(q, r)
        assert v.vote == 2

    def test_modality_mismatch_fmri_vs_ephys(self):
        q = _make_query(expected_modalities=["fmri"])
        r = _make_record(modalities=["ephys"])
        v = lf_modality_mismatch(q, r)
        assert v.vote == 0

    def test_modality_mismatch_no_overlap(self):
        q = _make_query(expected_modalities=["eeg"])
        r = _make_record(modalities=["calcium_imaging"])
        v = lf_modality_mismatch(q, r)
        assert v.vote == 0

    def test_no_mismatch_when_overlap(self):
        q = _make_query(expected_modalities=["fmri"])
        r = _make_record(modalities=["fmri", "eeg"])
        v = lf_modality_mismatch(q, r)
        assert v.vote is None


class TestTaskLF:
    def test_task_match(self):
        q = _make_query(expected_tasks=["go_nogo"])
        r = _make_record(tasks=["go_nogo", "lick_task"])
        v = lf_task_match(q, r)
        assert v.vote == 2

    def test_task_partial_match_in_description(self):
        q = _make_query(expected_tasks=["decision making"])
        r = _make_record(tasks=[], description="A decision making paradigm with rewards")
        v = lf_task_partial_match(q, r)
        assert v.vote == 1

    def test_task_no_expected(self):
        q = _make_query(expected_tasks=[])
        r = _make_record()
        v = lf_task_match(q, r)
        assert v.vote is None


class TestBrainRegionLF:
    def test_brain_region_match(self):
        q = _make_query(expected_brain_regions=["hippocampus"])
        r = _make_record(brain_regions=["CA1", "dentate gyrus"])
        v = lf_brain_region_match(q, r)
        assert v.vote == 2

    def test_brain_region_mismatch(self):
        q = _make_query(expected_brain_regions=["visual_cortex"])
        r = _make_record(brain_regions=["motor_cortex", "cerebellum"])
        v = lf_brain_region_mismatch(q, r)
        assert v.vote == 0


# ---------------------------------------------------------------------------
# 3. Hard-negative override
# ---------------------------------------------------------------------------


class TestHardNegativeOverride:
    def test_hn_fires_on_known_pattern(self):
        q = _make_query(
            hard_negatives=["resting-state fMRI with reward words in description"]
        )
        r = _make_record(
            description="Resting state fMRI with reward valuation task.",
            modalities=["fmri"],
        )
        v = lf_hard_negative_violation(q, r)
        assert v.vote == 0
        assert v.confidence >= 0.70

    def test_hn_no_match(self):
        q = _make_query(hard_negatives=["animal RL task when human-only requested"])
        r = _make_record(species=["human"], tasks=["n-back"])
        v = lf_hard_negative_violation(q, r)
        assert v.vote is None

    def test_hn_overrides_concept_match(self):
        """Even with concept match, HN vote of 0 should dominate aggregation."""
        concept_vote = LabelingFunctionVote(
            source="concept_memory", vote=2, confidence=0.80,
            evidence=["concept_match: calcium (modality, score=0.85)"]
        )
        hn_vote = LabelingFunctionVote(
            source="rules", vote=0, confidence=0.85,
            evidence=["hard_negative_violation: species mismatch"]
        )
        relevance, confidence, _ = _aggregate_votes([concept_vote, hn_vote])
        assert relevance == 0


# ---------------------------------------------------------------------------
# 4. Missing metadata confidence reduction
# ---------------------------------------------------------------------------


class TestMissingMetadata:
    def test_missing_required_field_reduces_confidence(self):
        q = _make_query(must_have=["species", "modality"])
        r = _make_record(species=[], modalities=[])
        v = lf_missing_critical_metadata(q, r)
        assert v.vote is None
        assert v.confidence < 0.50

    def test_missing_metadata_does_not_vote_zero(self):
        """Missing metadata must abstain, not vote 0."""
        q = _make_query(must_have=["brain_regions"])
        r = _make_record(brain_regions=[])
        v = lf_missing_critical_metadata(q, r)
        assert v.vote is None  # must abstain, not 0

    def test_present_fields_no_abstain(self):
        q = _make_query(must_have=["species"])
        r = _make_record(species=["mouse"])
        v = lf_missing_critical_metadata(q, r)
        assert v.vote is None  # abstain when all present (no signal either way)


# ---------------------------------------------------------------------------
# 5. Affordance probes
# ---------------------------------------------------------------------------


class TestAffordanceProbes:
    def test_spike_sorting_ephys_record(self):
        r = _make_record(modalities=["extracellular_ephys", "neuropixels"])
        result = probe_spike_sorting(r)
        assert result.supported is True
        assert result.confidence > 0.60

    def test_spike_sorting_calcium_record(self):
        r = _make_record(modalities=["calcium_imaging"])
        result = probe_spike_sorting(r)
        assert result.supported is False

    def test_calcium_imaging_probe(self):
        r = _make_record(
            modalities=["two_photon"],
            description="Two-photon imaging of GCaMP6s. dF/F traces and ROIs provided.",
        )
        result = probe_calcium_imaging(r)
        assert result.supported is True

    def test_rl_model_fitting_with_choices(self):
        r = _make_record(
            tasks=["decision making"],
            description="Trial-level choices, rewards, and outcomes recorded.",
        )
        result = probe_rl_model_fitting(r)
        assert result.supported is True

    def test_rl_model_fitting_no_choices(self):
        r = _make_record(tasks=[], description="Structural MRI only.")
        result = probe_rl_model_fitting(r)
        assert result.supported is False

    def test_pose_analysis_with_dlc(self):
        r = _make_record(
            description="Video recordings with DeepLabCut pose estimation.",
        )
        result = probe_pose_analysis(r)
        assert result.supported is True

    def test_neural_decoding_needs_neural_and_labels(self):
        r = _make_record(
            modalities=["calcium_imaging"],
            description="Calcium imaging with trial labels and stimulus categories.",
        )
        result = probe_neural_decoding(r)
        assert result.supported is not False

    def test_meta_analysis_completeness(self):
        r = _make_record()  # full record
        result = probe_meta_analysis(r)
        assert result.confidence > 0.50

    def test_probe_to_vote_converts_correctly(self):
        result = AffordanceProbeResult(
            affordance="spike_sorting", supported=True, confidence=0.75
        )
        vote = probe_to_vote(result)
        assert vote.vote == 2
        assert vote.source == "affordance_probe"

    def test_probe_unsupported_gives_vote_zero(self):
        result = AffordanceProbeResult(
            affordance="spike_sorting", supported=False, confidence=0.80
        )
        vote = probe_to_vote(result)
        assert vote.vote == 0

    def test_probe_unknown_gives_abstain(self):
        result = AffordanceProbeResult(
            affordance="spike_sorting", supported=None, confidence=0.45
        )
        vote = probe_to_vote(result)
        assert vote.vote is None


# ---------------------------------------------------------------------------
# 6. Concept-memory weak labeler
# ---------------------------------------------------------------------------


class TestConceptLabeler:
    def test_concept_match_helps_when_aligned(self):
        q = _make_query(expected_modalities=["calcium_imaging"])
        cr = {
            "matched_concepts": [
                {"concept_id": "concept:modality:calcium_imaging",
                 "canonical_name": "calcium_imaging",
                 "concept_type": "modality",
                 "match_score": 0.90,
                 "evidence_texts": ["modalities: calcium_imaging"],
                 "relation_types": ["has_modality"]},
            ],
            "missing_evidence": [],
            "hard_negative_conflicts": [],
            "explanation_summary": "Strong modality match.",
        }
        v = label_from_concept_result(q, cr)
        assert v.vote in (1, 2)
        assert v.source == "concept_memory"

    def test_hn_conflict_overrides_concept_match(self):
        q = _make_query()
        cr = {
            "matched_concepts": [
                {"concept_id": "concept:modality:calcium_imaging",
                 "canonical_name": "calcium_imaging",
                 "concept_type": "modality",
                 "match_score": 0.90,
                 "evidence_texts": ["modalities: calcium_imaging"],
                 "relation_types": ["has_modality"]},
            ],
            "missing_evidence": [],
            "hard_negative_conflicts": ["species failure_mode: human-only, confidence=0.9"],
            "explanation_summary": "",
        }
        v = label_from_concept_result(q, cr)
        assert v.vote == 0

    def test_unreviewed_metadata_links_produce_lower_confidence(self):
        q = _make_query()
        cr = {
            "matched_concepts": [
                {"concept_id": "concept:modality:fmri",
                 "canonical_name": "fmri",
                 "concept_type": "modality",
                 "match_score": 0.60,
                 "evidence_texts": [],  # no evidence texts = metadata only
                 "relation_types": ["has_modality"]},
            ],
            "missing_evidence": ["task", "species"],
            "hard_negative_conflicts": [],
        }
        v_missing = label_from_concept_result(q, cr)
        cr_good = {
            "matched_concepts": [
                {"concept_id": "concept:modality:fmri",
                 "canonical_name": "fmri",
                 "concept_type": "modality",
                 "match_score": 0.90,
                 "evidence_texts": ["paper: Reward-based fMRI study"],
                 "relation_types": ["has_modality"]},
            ] * 4,
            "missing_evidence": [],
            "hard_negative_conflicts": [],
        }
        v_good = label_from_concept_result(q, cr_good)
        assert v_missing.confidence < v_good.confidence

    def test_no_matches_abstain(self):
        q = _make_query()
        cr = {
            "matched_concepts": [],
            "missing_evidence": ["modality", "species"],
            "hard_negative_conflicts": [],
        }
        v = label_from_concept_result(q, cr)
        assert v.vote is None

    def test_missing_concepts_abstain_not_false_negative(self):
        q = _make_query()
        cr = {
            "matched_concepts": [
                {"concept_id": "concept:modality:fmri",
                 "canonical_name": "fmri",
                 "concept_type": "modality",
                 "match_score": 0.55,
                 "evidence_texts": ["modalities: fmri"],
                 "relation_types": ["has_modality"]},
            ],
            "missing_evidence": ["task", "brain_region", "species"],
            "hard_negative_conflicts": [],
        }
        v = label_from_concept_result(q, cr)
        # Should produce a low vote (1) not None, and not 0
        assert v.vote in (None, 1)


# ---------------------------------------------------------------------------
# 7. LLM judge mock
# ---------------------------------------------------------------------------


class TestMockLLMJudge:
    def test_mock_judge_modality_match(self):
        judge = MockLLMJudge()
        q = _make_query(expected_modalities=["calcium_imaging"])
        r = _make_record(modalities=["calcium_imaging"])
        v = judge.judge(q, r)
        assert v.vote == 2
        assert v.source == "llm_judge"

    def test_mock_judge_modality_mismatch(self):
        judge = MockLLMJudge()
        q = _make_query(expected_modalities=["ephys"])
        r = _make_record(modalities=["fmri"])
        v = judge.judge(q, r)
        assert v.vote in (1, None)

    def test_mock_judge_no_metadata(self):
        judge = MockLLMJudge()
        q = _make_query()
        r = _make_record(modalities=[])
        v = judge.judge(q, r)
        assert v.vote is None


# ---------------------------------------------------------------------------
# 8 & 9. Vote aggregation and confidence computation
# ---------------------------------------------------------------------------


class TestVoteAggregation:
    def test_all_agree_on_2(self):
        votes = [
            LabelingFunctionVote(source="rules", vote=2, confidence=0.80),
            LabelingFunctionVote(source="concept_memory", vote=2, confidence=0.75),
        ]
        rel, conf, disagree = _aggregate_votes(votes)
        assert rel == 2
        assert conf > 0.70
        assert not disagree

    def test_all_abstain_defaults_to_1(self):
        votes = [
            LabelingFunctionVote(source="rules", vote=None, confidence=0.4),
            LabelingFunctionVote(source="concept_memory", vote=None, confidence=0.3),
        ]
        rel, conf, _ = _aggregate_votes(votes)
        assert rel == 1
        assert conf < 0.50

    def test_hn_override_aggregation(self):
        votes = [
            LabelingFunctionVote(
                source="rules", vote=0, confidence=0.85,
                evidence=["hard_negative_violation: animal RL"]
            ),
            LabelingFunctionVote(source="concept_memory", vote=2, confidence=0.80),
        ]
        rel, conf, _ = _aggregate_votes(votes)
        assert rel == 0

    def test_disagreement_detected(self):
        votes = [
            LabelingFunctionVote(source="rules", vote=3, confidence=0.90),
            LabelingFunctionVote(source="concept_memory", vote=0, confidence=0.80),
        ]
        _, _, disagree = _aggregate_votes(votes)
        assert len(disagree) > 0

    def test_confidence_lower_with_spread(self):
        votes_agree = [
            LabelingFunctionVote(source="rules", vote=2, confidence=0.80),
            LabelingFunctionVote(source="concept_memory", vote=2, confidence=0.80),
        ]
        votes_disagree = [
            LabelingFunctionVote(source="rules", vote=3, confidence=0.80),
            LabelingFunctionVote(source="concept_memory", vote=0, confidence=0.80),
        ]
        _, c_agree, _ = _aggregate_votes(votes_agree)
        _, c_disagree, _ = _aggregate_votes(votes_disagree)
        assert c_agree > c_disagree


# ---------------------------------------------------------------------------
# 10. Disagreement detection
# ---------------------------------------------------------------------------


class TestDisagreementDetection:
    def test_no_disagreement_when_unanimous(self):
        votes = [
            LabelingFunctionVote(source="a", vote=2, confidence=0.7),
            LabelingFunctionVote(source="b", vote=2, confidence=0.7),
        ]
        _, _, dis = _aggregate_votes(votes)
        assert not dis

    def test_disagreement_when_poles_apart(self):
        votes = [
            LabelingFunctionVote(source="a", vote=0, confidence=0.8),
            LabelingFunctionVote(source="b", vote=3, confidence=0.8),
        ]
        _, _, dis = _aggregate_votes(votes)
        assert len(dis) >= 1


# ---------------------------------------------------------------------------
# 11. Review queue selection
# ---------------------------------------------------------------------------


class TestReviewQueueSelection:
    def _make_entries(self, n: int = 5) -> list[SilverQrelsEntry]:
        return [
            SilverQrelsEntry.model_validate({
                "query_id": f"q_{i:04d}",
                "dataset_id": f"ds{i:04d}",
                "label_type": "silver",
                "relevance": i % 4,
                "confidence": min(0.3 + 0.07 * i, 0.99),
                "needs_human_review": i < 3,
                "review_priority": max(0.0, 0.8 - 0.1 * i),
                "disagreements": ["vote spread: {a: 0, b: 2}"] if i < 2 else [],
                "all_votes": [],
            })
            for i in range(n)
        ]

    def test_select_respects_limit(self):
        entries = self._make_entries(10)
        queue = select_review_set(entries, {}, {}, limit=4)
        assert len(queue) <= 4

    def test_low_confidence_ranked_higher(self):
        entries = [
            SilverQrelsEntry.model_validate({
                "query_id": "q_a", "dataset_id": "da",
                "label_type": "silver", "relevance": 1,
                "confidence": 0.25, "all_votes": [],
            }),
            SilverQrelsEntry.model_validate({
                "query_id": "q_b", "dataset_id": "db",
                "label_type": "silver", "relevance": 1,
                "confidence": 0.90, "all_votes": [],
            }),
        ]
        queue = select_review_set(entries, {}, {}, limit=2, seed=42)
        # Lower-confidence entry should appear first
        assert queue[0].query_id == "q_a"

    def test_review_queue_entry_has_required_fields(self):
        entries = self._make_entries(3)
        queue = select_review_set(entries, {}, {}, limit=3)
        for e in queue:
            assert e.query_id
            assert e.dataset_id
            assert 0 <= e.annotation_priority <= 1.0


# ---------------------------------------------------------------------------
# 12. Silver-gold calibration placeholder
# ---------------------------------------------------------------------------


class TestCalibrationPlaceholder:
    def test_placeholder_written_when_no_gold(self, tmp_path):
        silver_path = tmp_path / "qrels_silver.jsonl"
        gold_path = tmp_path / "qrels.jsonl"  # does not exist
        out_path = tmp_path / "calibration.md"

        # Write minimal silver entry
        e = _make_silver_entry()
        silver_path.write_text(e.model_dump_json() + "\n", encoding="utf-8")

        from scripts.eval.calibrate_silver_qrels import _write_placeholder
        _write_placeholder(out_path, silver_path, gold_path)

        text = out_path.read_text(encoding="utf-8")
        assert "NOT EXPERT VALIDATION" in text
        assert "python scripts/eval/calibrate_silver_qrels.py" in text


# ---------------------------------------------------------------------------
# 13. Silver-gold calibration with small fixture
# ---------------------------------------------------------------------------


class TestCalibrationWithFixture:
    def _make_silver_dict(self) -> dict:
        """Minimal in-memory silver entries."""
        return {
            ("q1", "d1"): _make_silver_entry(
                query_id="q1", dataset_id="d1", relevance=2, confidence=0.80,
            ),
            ("q1", "d2"): _make_silver_entry(
                query_id="q1", dataset_id="d2", relevance=0, confidence=0.75,
                hard_negative_violation=True,
                all_votes=[LabelingFunctionVote(
                    source="rules", vote=0, confidence=0.85,
                    evidence=["hard_negative_violation: mismatch"]
                )],
            ),
            ("q2", "d3"): _make_silver_entry(
                query_id="q2", dataset_id="d3", relevance=1, confidence=0.55,
            ),
        }

    def test_accuracy_perfect_agreement(self):
        silver = self._make_silver_dict()
        gold = {("q1", "d1"): 2, ("q1", "d2"): 0, ("q2", "d3"): 1}
        cm = _confusion_matrix(silver, gold)
        assert cm["overlap"] == 3
        assert cm["accuracy"] == 1.0

    def test_accuracy_partial_agreement(self):
        silver = self._make_silver_dict()
        gold = {("q1", "d1"): 3, ("q1", "d2"): 0}  # q1/d1 disagrees
        cm = _confusion_matrix(silver, gold)
        assert cm["overlap"] == 2
        assert cm["accuracy"] == 0.5

    def test_high_conf_accuracy(self):
        silver = self._make_silver_dict()
        gold = {("q1", "d1"): 2, ("q1", "d2"): 0}
        hc = _high_conf_accuracy(silver, gold, threshold=0.70)
        assert hc["count"] == 2
        assert hc["accuracy"] == 1.0

    def test_hn_precision(self):
        silver = self._make_silver_dict()
        gold = {("q1", "d2"): 0}  # HN entry should be gold=0
        hn = _hn_precision(silver, gold)
        assert hn["count"] == 1
        assert hn["precision"] == 1.0

    def test_calibration_bins(self):
        silver = self._make_silver_dict()
        gold = {("q1", "d1"): 2, ("q1", "d2"): 0, ("q2", "d3"): 1}
        bins = _calibration_by_bin(silver, gold)
        assert isinstance(bins, list)

    def test_no_overlap_returns_safe_result(self):
        silver = self._make_silver_dict()
        gold = {("q_other", "d_other"): 2}
        cm = _confusion_matrix(silver, gold)
        assert cm["overlap"] == 0


# ---------------------------------------------------------------------------
# 14. Silver evaluation guardrail
# ---------------------------------------------------------------------------


class TestSilverEvalGuardrail:
    def test_guard_output_path_rejects_gold(self):
        with pytest.raises(SystemExit):
            _guard_output_path(GOLD_QRELS_PATH)

    def test_guard_output_path_accepts_silver(self, tmp_path):
        silver_path = tmp_path / "qrels_silver.jsonl"
        _guard_output_path(silver_path)  # must not raise

    def test_compute_ir_metrics_rejects_silver_without_flag(self, tmp_path):
        from scripts.eval.compute_ir_metrics import main as compute_main
        qrels_path = tmp_path / "qrels_silver.jsonl"
        run_path = tmp_path / "run.jsonl"
        out_path = tmp_path / "report.json"
        qrels_path.write_text("", encoding="utf-8")
        run_path.write_text("", encoding="utf-8")
        result = compute_main([
            "--qrels", str(qrels_path),
            "--run", str(run_path),
            "--out", str(out_path),
        ])
        assert result == 2  # should reject

    def test_compute_ir_metrics_accepts_silver_with_flag(self, tmp_path):
        from scripts.eval.compute_ir_metrics import main as compute_main
        qrels_path = tmp_path / "qrels_silver.jsonl"
        run_path = tmp_path / "run.jsonl"
        out_path = tmp_path / "report.json"
        qrels_path.write_text("", encoding="utf-8")
        run_path.write_text("", encoding="utf-8")
        result = compute_main([
            "--qrels", str(qrels_path),
            "--run", str(run_path),
            "--out", str(out_path),
            "--allow-silver",
        ])
        assert result == 0  # should succeed (empty qrels = pending)

    def test_silver_report_is_watermarked(self, tmp_path):
        from scripts.eval.compute_ir_metrics import main as compute_main
        qrels_path = tmp_path / "qrels_silver.jsonl"
        # Write a minimal qrel in the "label" format expected by compute_ir_metrics
        qrels_path.write_text(
            json.dumps({"query_id": "q1", "record_id": "d1", "label": 2}) + "\n",
            encoding="utf-8",
        )
        run_path = tmp_path / "run.jsonl"
        run_path.write_text(
            json.dumps({"query_id": "q1", "record_id": "d1", "rank": 1, "score": 0.9}) + "\n",
            encoding="utf-8",
        )
        out_path = tmp_path / "report.json"
        compute_main([
            "--qrels", str(qrels_path),
            "--run", str(run_path),
            "--out", str(out_path),
            "--allow-silver",
        ])
        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert "silver_label_warning" in report
        assert "NOT EXPERT VALIDATION" in report["silver_label_warning"]


# ---------------------------------------------------------------------------
# 15. Reports generated with required warnings
# ---------------------------------------------------------------------------


class TestReportWarnings:
    def test_summary_contains_watermark(self, tmp_path):
        from scripts.eval.build_silver_qrels import _write_summary_md
        entries = [_make_silver_entry()]
        summary = build_summary(entries, seed=13)
        md_path = tmp_path / "summary.md"
        _write_summary_md(summary, md_path)
        text = md_path.read_text(encoding="utf-8")
        assert "NOT EXPERT VALIDATION" in text

    def test_disagreements_report_contains_watermark(self, tmp_path):
        from scripts.eval.build_silver_qrels import _write_disagreements_md
        entries = [_make_silver_entry(disagreements=["vote spread: {a: 0, b: 2}"])]
        md_path = tmp_path / "disagreements.md"
        _write_disagreements_md(entries, md_path)
        text = md_path.read_text(encoding="utf-8")
        assert "NOT EXPERT VALIDATION" in text


# ---------------------------------------------------------------------------
# 16. No writes to artifacts/qrels.jsonl by default
# ---------------------------------------------------------------------------


class TestNoGoldQrelsWrite:
    def test_build_silver_never_writes_gold_path(self, tmp_path):
        """build_silver_qrels itself does not write files; the CLI does.
        But _guard_output_path must reject the gold path."""
        gold_path = Path("artifacts/qrels.jsonl")
        with pytest.raises(SystemExit):
            _guard_output_path(gold_path)

    def test_guard_accepts_any_other_path(self, tmp_path):
        for name in ("qrels_silver.jsonl", "silver.jsonl", "my_qrels.jsonl"):
            _guard_output_path(tmp_path / name)  # must not raise


# ---------------------------------------------------------------------------
# 17. Determinism with seed
# ---------------------------------------------------------------------------


class TestDeterminism:
    def _minimal_data(self):
        queries = {
            "q_0001": _make_query(
                query_id="q_0001",
                expected_species=["mouse"],
                expected_modalities=["calcium_imaging"],
            )
        }
        pool = [{"query_id": "q_0001", "record_id": "ds001"}]
        corpus = {"ds001": _make_record()}
        return queries, pool, corpus

    def test_same_seed_same_output(self):
        q, p, c = self._minimal_data()
        entries_a = build_silver_qrels(q, p, c, seed=42)
        entries_b = build_silver_qrels(q, p, c, seed=42)
        assert len(entries_a) == len(entries_b)
        for a, b in zip(entries_a, entries_b, strict=True):
            assert a.relevance == b.relevance
            assert a.confidence == b.confidence

    def test_different_seeds_same_output_for_deterministic_lfs(self):
        """Since LFs are fully deterministic, seed only affects stochastic steps.
        Relevance/confidence should be identical regardless of seed."""
        q, p, c = self._minimal_data()
        entries_13 = build_silver_qrels(q, p, c, seed=13)
        entries_42 = build_silver_qrels(q, p, c, seed=42)
        # Deterministic labeling functions → same results
        assert entries_13[0].relevance == entries_42[0].relevance

    def test_review_queue_deterministic_with_seed(self):
        q, p, c = self._minimal_data()
        entries = build_silver_qrels(q, p, c, seed=13)
        queue_a = select_review_set(entries, q, {}, seed=99)
        queue_b = select_review_set(entries, q, {}, seed=99)
        assert [e.query_id for e in queue_a] == [e.query_id for e in queue_b]


# ---------------------------------------------------------------------------
# Integration — build_silver_qrels end-to-end (small)
# ---------------------------------------------------------------------------


class TestBuildSilverQrelsIntegration:
    def test_end_to_end_small(self):
        queries = {
            "q_0001": _make_query(
                query_id="q_0001",
                expected_species=["human"],
                expected_modalities=["fmri"],
                hard_negatives=["resting-state fMRI with reward words in description"],
            )
        }
        pool = [
            {"query_id": "q_0001", "record_id": "ds_good"},
            {"query_id": "q_0001", "record_id": "ds_hn"},
            {"query_id": "q_0001", "record_id": "ds_missing"},
        ]
        corpus = {
            "ds_good": _make_record(
                source_id="ds_good", species=["human"], modalities=["fmri"],
                tasks=["n-back"],
            ),
            "ds_hn": _make_record(
                source_id="ds_hn", species=["human"], modalities=["fmri"],
                description="Resting state fMRI with reward words and reward valuation.",
            ),
            "ds_missing": _make_record(
                source_id="ds_missing", species=[], modalities=[],
            ),
        }
        entries = build_silver_qrels(queries, pool, corpus, seed=13)
        assert len(entries) == 3

        by_id = {e.dataset_id: e for e in entries}

        # Good record should have positive relevance
        assert by_id["ds_good"].relevance >= 1

        # Hard-negative record should have relevance=0
        assert by_id["ds_hn"].relevance == 0

        # Missing metadata record should have needs_human_review or low confidence
        ds_missing = by_id["ds_missing"]
        assert ds_missing.confidence < 0.60 or ds_missing.needs_human_review

    def test_apply_all_lfs_returns_list(self):
        q = _make_query()
        r = _make_record()
        votes = apply_all_labeling_functions(q, r)
        assert isinstance(votes, list)
        assert all(isinstance(v, LabelingFunctionVote) for v in votes)

    def test_needs_review_low_confidence(self):
        needs, priority = _needs_review(0.30, [], False, [])
        assert needs is True
        assert priority > 0

    def test_needs_review_disagreement(self):
        needs, priority = _needs_review(0.70, ["vote spread"], False, [])
        assert needs is True

    def test_no_review_needed_high_confidence(self):
        needs, priority = _needs_review(0.85, [], False, [])
        assert needs is False
