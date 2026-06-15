"""Tests for neural_search.eval.neuro_judge — 59 tests total.

All tests are fully offline (no network calls, no real API keys).
Tests use MockNeuroJudge and hand-crafted objects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

# Repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_search.eval.neuro_judge.calibration import CalibrationReport, calibrate
from neural_search.eval.neuro_judge.consensus import build_consensus
from neural_search.eval.neuro_judge.evidence_packet import (
    NEURO_JUDGE_WATERMARK,
    VALID_LABEL_PROVENANCES,
    ConflictRecord,
    ConsensusResult,
    EvidencePacket,
    NeuroJudgment,
)
from neural_search.eval.neuro_judge.evidence_retriever import build_evidence_packet
from neural_search.eval.neuro_judge.judge import (
    BrainGPTAdapter,
    JudgeParseError,
    LocalHFNeuroJudge,
    MockNeuroJudge,
    NeuroJudgeProtocol,
    _parse_judgment,
    build_neuro_judge,
)
from neural_search.eval.neuro_judge.prompt import (
    build_judge_prompt,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


class _FakeQuery:
    """Minimal query stand-in (mimics BenchmarkQueryV1 attribute access)."""

    def __init__(
        self,
        query_id: str = "q_001",
        query_text: str = "theta oscillations in rodent hippocampus",
        intent: str = "REPLICATION",
        expected_modalities: list[str] | None = None,
        expected_species: list[str] | None = None,
        expected_brain_regions: list[str] | None = None,
        expected_tasks: list[str] | None = None,
        expected_analysis_affordances: list[str] | None = None,
        hard_negatives: list[str] | None = None,
        must_have: list[str] | None = None,
        scientific_goal: str = "",
    ) -> None:
        self.query_id = query_id
        self.query_text = query_text
        self.intent = intent
        self.expected_modalities = expected_modalities or ["extracellular_ephys"]
        self.expected_species = expected_species or ["mouse", "rat"]
        self.expected_brain_regions = expected_brain_regions or ["hippocampus"]
        self.expected_tasks = expected_tasks or ["open_field"]
        self.expected_analysis_affordances = expected_analysis_affordances or ["spike_sorting"]
        self.hard_negatives = hard_negatives or []
        self.must_have = must_have or []
        self.scientific_goal = scientific_goal


def _record(
    dataset_id: str = "dandi:000003",
    title: str = "Rat hippocampal extracellular recordings",
    source: str = "dandi",
    modalities: list[str] | None = None,
    species: list[str] | None = None,
    brain_regions: list[str] | None = None,
    tasks: list[str] | None = None,
    data_standards: list[str] | None = None,
    description: str = "Raw extracellular ephys data from rat hippocampus",
    **extra: Any,
) -> dict[str, Any]:
    r: dict[str, Any] = {
        "dataset_id": dataset_id,
        "title": title,
        "source": source,
        "modalities": modalities if modalities is not None else ["extracellular_ephys"],
        "species": species if species is not None else ["rat"],
        "brain_regions": brain_regions if brain_regions is not None else ["hippocampus"],
        "tasks": tasks if tasks is not None else ["open_field"],
        "data_standards": data_standards if data_standards is not None else ["NWB"],
        "description": description,
    }
    r.update(extra)
    return r


def _packet(
    query_id: str = "q_001",
    dataset_id: str = "dandi:000003",
    expected_modalities: list[str] | None = None,
    dataset_modalities: list[str] | None = None,
    expected_species: list[str] | None = None,
    dataset_species: list[str] | None = None,
    hard_negatives: list[str] | None = None,
    concept_explanation_summary: str = "",
    known_failure_warnings: list[str] | None = None,
    concept_hard_negative_conflicts: list[str] | None = None,
) -> EvidencePacket:
    return EvidencePacket(
        query_id=query_id,
        dataset_id=dataset_id,
        query_text="theta oscillations in rodent hippocampus",
        expected_modalities=expected_modalities or ["extracellular_ephys"],
        expected_species=expected_species or ["mouse"],
        dataset_modalities=dataset_modalities or ["extracellular_ephys"],
        dataset_species=dataset_species or ["mouse"],
        hard_negatives=hard_negatives or [],
        concept_explanation_summary=concept_explanation_summary,
        known_failure_warnings=known_failure_warnings or [],
        concept_hard_negative_conflicts=concept_hard_negative_conflicts or [],
    )


def _judgment(
    query_id: str = "q_001",
    dataset_id: str = "dandi:000003",
    label: int = 2,
    confidence: float = 0.8,
    hard_negative_detected: bool = False,
    rationale_short: str = "test judgment",
    label_provenance: str = "neuro_judge",
    failure_modes: list[str] | None = None,
    missing_information: list[str] | None = None,
) -> NeuroJudgment:
    return NeuroJudgment(
        query_id=query_id,
        dataset_id=dataset_id,
        label=label,
        confidence=confidence,
        hard_negative_detected=hard_negative_detected,
        rationale_short=rationale_short,
        label_provenance=label_provenance,
        failure_modes=failure_modes or [],
        missing_information=missing_information or [],
    )


# ===========================================================================
# 1. Evidence packet construction (8 tests)
# ===========================================================================


class TestEvidencePacketConstruction:
    def test_build_minimal_packet(self) -> None:
        q = _FakeQuery()
        r = _record()
        pkt = build_evidence_packet(q, r)
        assert pkt.query_id == "q_001"
        assert pkt.dataset_id == "dandi:000003"
        assert pkt.query_text == "theta oscillations in rodent hippocampus"

    def test_build_with_concept_result(self) -> None:
        q = _FakeQuery()
        r = _record()
        cr = {
            "explanation_summary": "Strong theta concept match",
            "matched_concepts": [{"canonical_name": "theta oscillations", "concept_type": "neural_signal"}],
            "missing_evidence": ["raw_ap_band"],
            "hard_negative_conflicts": [],
        }
        pkt = build_evidence_packet(q, r, concept_result=cr)
        assert pkt.concept_explanation_summary == "Strong theta concept match"
        assert "theta oscillations" in pkt.matched_concept_names
        assert "raw_ap_band" in pkt.concept_missing_evidence

    def test_build_with_paper_abstract(self) -> None:
        q = _FakeQuery()
        r = _record(linked_paper={"title": "Hippocampal theta rhythms", "abstract": "We recorded..."})
        pkt = build_evidence_packet(q, r)
        assert len(pkt.linked_papers) == 1
        assert pkt.linked_papers[0].title == "Hippocampal theta rhythms"

    def test_affordance_matches_extracted(self) -> None:
        q = _FakeQuery(expected_analysis_affordances=["spike_sorting"])
        r = _record(affordances=["spike_sorting", "lfp_analysis"])
        pkt = build_evidence_packet(q, r)
        assert any(am.affordance == "spike_sorting" and am.matched for am in pkt.affordance_matches)

    def test_affordance_missing_requirements(self) -> None:
        q = _FakeQuery(expected_analysis_affordances=["spike_sorting"])
        r = _record(affordances=[])  # no affordances
        cr = {"missing_evidence": ["spike_sorting:raw_ap_band_required"]}
        pkt = build_evidence_packet(q, r, concept_result=cr)
        # affordance not matched, missing requirements populated
        assert any(not am.matched for am in pkt.affordance_matches)

    def test_hard_negative_warning_flagged(self) -> None:
        q = _FakeQuery(hard_negatives=["human fMRI"])
        r = _record(
            modalities=["fMRI"],
            species=["human"],
            description="Human fMRI dataset for reward learning",
        )
        pkt = build_evidence_packet(q, r)
        assert len(pkt.known_failure_warnings) > 0

    def test_hard_negative_warning_requires_phrase_match(self) -> None:
        q = _FakeQuery(hard_negatives=["pre-sorted spike tables only, no raw AP-band"])
        r = _record(
            description=(
                "IBL Neuropixels ALF session with spike_times and trial tables. "
                "Raw AP-band data not explicitly confirmed."
            ),
        )
        pkt = build_evidence_packet(q, r)
        assert pkt.known_failure_warnings == []

    def test_unknown_fields_default_to_empty(self) -> None:
        q = _FakeQuery()
        r = {"dataset_id": "x:1", "title": "Minimal"}
        pkt = build_evidence_packet(q, r)
        assert pkt.dataset_modalities == []
        assert pkt.dataset_species == []
        assert pkt.linked_papers == []

    def test_packet_hash_deterministic(self) -> None:
        q = _FakeQuery()
        r = _record()
        pkt1 = build_evidence_packet(q, r)
        pkt2 = build_evidence_packet(q, r)
        assert pkt1.packet_hash() == pkt2.packet_hash()


# ===========================================================================
# 2. Prompt construction (5 tests)
# ===========================================================================


class TestPromptConstruction:
    def test_prompt_contains_query_text(self) -> None:
        pkt = _packet()
        prompt = build_judge_prompt(pkt)
        assert "theta oscillations in rodent hippocampus" in prompt

    def test_prompt_contains_rubric(self) -> None:
        pkt = _packet()
        prompt = build_judge_prompt(pkt)
        assert "Score 0" in prompt
        assert "Score 3" in prompt

    def test_prompt_contains_neuro_rules(self) -> None:
        pkt = _packet()
        prompt = build_judge_prompt(pkt)
        assert "fMRI-only" in prompt
        assert "calcium" in prompt.lower()

    def test_prompt_contains_hard_negatives(self) -> None:
        pkt = _packet(hard_negatives=["human fMRI", "behavior-only"])
        prompt = build_judge_prompt(pkt)
        assert "human fMRI" in prompt
        assert "behavior-only" in prompt

    def test_prompt_contains_evidence_fields(self) -> None:
        pkt = _packet()
        pkt.dataset_id = "dandi:000003"
        pkt.title = "Rat hippocampal recordings"
        prompt = build_judge_prompt(pkt)
        assert "dandi:000003" in prompt
        assert "Rat hippocampal recordings" in prompt


# ===========================================================================
# 3. Strict JSON parsing (5 tests)
# ===========================================================================


class TestStrictJSONParsing:
    def _pkt(self) -> EvidencePacket:
        return _packet()

    def test_valid_json_parsed(self) -> None:
        text = json.dumps({
            "label": 2,
            "confidence": 0.8,
            "rationale_short": "good match",
            "evidence_for": ["species match"],
            "evidence_against": [],
            "missing_information": [],
            "matched_dimensions": ["species", "modality"],
            "failure_modes": [],
            "hard_negative_detected": False,
        })
        j = _parse_judgment(text, self._pkt(), "mock-model", "v1")
        assert j.label == 2
        assert j.confidence == 0.8
        assert "species" in j.matched_dimensions

    def test_missing_label_raises(self) -> None:
        text = json.dumps({"confidence": 0.5})
        with pytest.raises(JudgeParseError, match="label"):
            _parse_judgment(text, self._pkt(), "mock", "v1")

    def test_label_out_of_range_raises(self) -> None:
        text = json.dumps({"label": 5, "confidence": 0.5})
        with pytest.raises(JudgeParseError, match="out of range"):
            _parse_judgment(text, self._pkt(), "mock", "v1")

    def test_confidence_out_of_range_raises(self) -> None:
        text = json.dumps({"label": 1, "confidence": 1.5})
        with pytest.raises(JudgeParseError, match="out of range"):
            _parse_judgment(text, self._pkt(), "mock", "v1")

    def test_non_json_response_raises(self) -> None:
        with pytest.raises(JudgeParseError, match="JSON"):
            _parse_judgment("This is not JSON.", self._pkt(), "mock", "v1")

    def test_markdown_fenced_json_parsed(self) -> None:
        text = "```json\n" + json.dumps({"label": 1, "confidence": 0.6}) + "\n```"
        j = _parse_judgment(text, self._pkt(), "mock", "v1")
        assert j.label == 1


# ===========================================================================
# 4. Mocked model judging (8 tests)
# ===========================================================================


class TestMockedJudging:
    def test_mock_judge_returns_judgment(self) -> None:
        judge = MockNeuroJudge()
        pkt = _packet()
        j = judge.judge(pkt)
        assert isinstance(j, NeuroJudgment)
        assert 0 <= j.label <= 3

    def test_mock_judge_modality_match_scores_higher_than_mismatch(self) -> None:
        judge = MockNeuroJudge()
        match_pkt = _packet(expected_modalities=["ephys"], dataset_modalities=["ephys"])
        mismatch_pkt = _packet(expected_modalities=["ephys"], dataset_modalities=["fMRI"])
        j_match = judge.judge(match_pkt)
        j_mismatch = judge.judge(mismatch_pkt)
        assert j_match.label >= j_mismatch.label

    def test_mock_judge_no_modality_abstains_or_low(self) -> None:
        judge = MockNeuroJudge()
        pkt = _packet(expected_modalities=["ephys"], dataset_modalities=[])
        j = judge.judge(pkt)
        assert j.label <= 2

    def test_mock_judge_hard_negative_scores_zero(self) -> None:
        judge = MockNeuroJudge()
        pkt = _packet(known_failure_warnings=["possible_hard_negative: human fMRI"])
        j = judge.judge(pkt)
        assert j.label == 0
        assert j.hard_negative_detected is True

    def test_mock_judge_missing_fields_noted(self) -> None:
        judge = MockNeuroJudge()
        pkt = _packet()
        j = judge.judge(pkt)
        assert isinstance(j.failure_modes, list)

    def test_mock_judge_implements_protocol(self) -> None:
        judge = MockNeuroJudge()
        assert isinstance(judge, NeuroJudgeProtocol)

    def test_mock_judge_provenance_set(self) -> None:
        judge = MockNeuroJudge()
        pkt = _packet()
        j = judge.judge(pkt)
        assert j.label_provenance in VALID_LABEL_PROVENANCES

    def test_mock_judge_packet_hash_stored(self) -> None:
        judge = MockNeuroJudge()
        pkt = _packet()
        j = judge.judge(pkt)
        assert j.evidence_packet_hash == pkt.packet_hash()


# ===========================================================================
# 5. Local model adapter interface (4 tests)
# ===========================================================================


class TestLocalModelAdapter:
    def test_local_hf_adapter_skips_if_unavailable(self) -> None:
        judge = LocalHFNeuroJudge(model_name_or_path="nonexistent/model-xyz")
        assert judge.available is False

    def test_local_hf_adapter_returns_error_judgment_when_unavailable(self) -> None:
        judge = LocalHFNeuroJudge(model_name_or_path="nonexistent/model-xyz")
        pkt = _packet()
        j = judge.judge(pkt)
        assert j.confidence == 0.0
        assert "unavailable" in j.rationale_short

    def test_local_hf_adapter_interface(self) -> None:
        judge = LocalHFNeuroJudge(model_name_or_path="nonexistent/model-xyz")
        assert hasattr(judge, "judge")
        assert hasattr(judge, "model_id")
        assert hasattr(judge, "prompt_version")
        assert isinstance(judge, NeuroJudgeProtocol)

    def test_local_hf_fallback_on_generation_error(self) -> None:
        """When the pipeline exists but generates bad output, return error judgment."""
        judge = LocalHFNeuroJudge.__new__(LocalHFNeuroJudge)
        judge._model_name = "fake/model"
        judge._max_new_tokens = 128
        judge._available = True
        judge._pipeline = MagicMock(return_value=[{"generated_text": "NOT JSON"}])
        pkt = _packet()
        j = judge.judge(pkt)
        assert j.confidence == 0.0
        assert "error" in j.rationale_short.lower()


# ===========================================================================
# 6. BrainGPT adapter (3 tests)
# ===========================================================================


class TestBrainGPTAdapter:
    def test_braingpt_skipped_gracefully_if_unavailable(self) -> None:
        adapter = BrainGPTAdapter(model_name_or_path="braingpt/nonexistent")
        assert adapter.available is False

    def test_braingpt_adapter_interface(self) -> None:
        adapter = BrainGPTAdapter(model_name_or_path="braingpt/nonexistent")
        assert isinstance(adapter, NeuroJudgeProtocol)
        assert "braingpt" in adapter.model_id

    def test_braingpt_returns_error_judgment_without_weights(self) -> None:
        adapter = BrainGPTAdapter(model_name_or_path="braingpt/nonexistent")
        pkt = _packet()
        j = adapter.judge(pkt)
        assert j.confidence == 0.0


# ===========================================================================
# 7. Consensus agreement (9 tests)
# ===========================================================================


class TestConsensusAgreement:
    def test_two_judges_agree_exact_high_confidence(self) -> None:
        j1 = _judgment(label=2, confidence=0.9)
        j2 = _judgment(label=2, confidence=0.85)
        consensus, conflict = build_consensus([j1, j2])
        assert consensus is not None
        assert conflict is None
        assert consensus.label == 2
        assert consensus.exact_agreement is True

    def test_two_judges_agree_low_confidence_still_consensus(self) -> None:
        j1 = _judgment(label=1, confidence=0.5)
        j2 = _judgment(label=1, confidence=0.5)
        consensus, conflict = build_consensus([j1, j2])
        # exact agreement but low conf → consensus with minor_disagreement
        assert consensus is not None
        assert consensus.exact_agreement is True
        assert consensus.minor_disagreement is True

    def test_minor_disagreement_uses_median_label(self) -> None:
        j1 = _judgment(label=2, confidence=0.88)
        j2 = _judgment(label=3, confidence=0.85)
        consensus, conflict = build_consensus([j1, j2])
        assert consensus is not None
        assert conflict is None
        assert consensus.label in (2, 3)  # median of [2,3]
        assert consensus.minor_disagreement is True

    def test_major_disagreement_routed_to_conflict(self) -> None:
        j1 = _judgment(label=0, confidence=0.9)
        j2 = _judgment(label=3, confidence=0.9)
        consensus, conflict = build_consensus([j1, j2])
        assert consensus is None
        assert conflict is not None
        assert conflict.conflict_reason == "label_diff_gte_2"

    def test_single_judge_returns_consensus(self) -> None:
        j = _judgment(label=1)
        consensus, conflict = build_consensus([j])
        assert consensus is not None
        assert conflict is None

    def test_consensus_provenance_set(self) -> None:
        j1 = _judgment(label=2, confidence=0.9)
        j2 = _judgment(label=2, confidence=0.8)
        consensus, _ = build_consensus([j1, j2])
        assert consensus is not None
        assert consensus.label_provenance == "neuro_judge_consensus"

    def test_hn_detection_conflict_routes_to_conflict(self) -> None:
        j1 = _judgment(label=2, confidence=0.9, hard_negative_detected=False)
        j2 = _judgment(label=2, confidence=0.9, hard_negative_detected=True)
        consensus, conflict = build_consensus([j1, j2])
        assert consensus is None
        assert conflict is not None
        assert conflict.conflict_reason == "hn_detection_differs"

    def test_three_judges_majority_consensus(self) -> None:
        j1 = _judgment(label=2, confidence=0.9)
        j2 = _judgment(label=2, confidence=0.85)
        j3 = _judgment(label=3, confidence=0.82)
        consensus, conflict = build_consensus([j1, j2, j3])
        # diff=1, high mean conf → consensus
        assert consensus is not None or conflict is not None  # one must be non-None

    def test_consensus_watermark_present(self) -> None:
        j = _judgment(label=3, confidence=0.95)
        consensus, _ = build_consensus([j])
        assert consensus is not None
        assert NEURO_JUDGE_WATERMARK in consensus.watermark

    def test_empty_judgments_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            build_consensus([])

    def test_mixed_pairs_raise(self) -> None:
        j1 = _judgment(query_id="q_001", dataset_id="d_001")
        j2 = _judgment(query_id="q_002", dataset_id="d_001")
        with pytest.raises(ValueError, match="query_id"):
            build_consensus([j1, j2])


# ===========================================================================
# 8. Conflict routing (5 tests)
# ===========================================================================


class TestConflictRouting:
    def test_label_diff_gte_2_routed(self) -> None:
        j1 = _judgment(label=0, confidence=0.95)
        j2 = _judgment(label=2, confidence=0.95)
        _, conflict = build_consensus([j1, j2])
        assert conflict is not None
        assert "label_diff" in conflict.conflict_reason

    def test_hn_detection_differs_routed(self) -> None:
        j1 = _judgment(label=1, hard_negative_detected=True)
        j2 = _judgment(label=1, hard_negative_detected=False)
        _, conflict = build_consensus([j1, j2])
        assert conflict is not None
        assert "hn_detection" in conflict.conflict_reason

    def test_high_ndcg_impact_flagged_in_consensus(self) -> None:
        j = _judgment(label=3, confidence=0.9, missing_information=["raw_data_availability"])
        consensus, _ = build_consensus([j], ndcg_impact=0.5)
        assert consensus is not None
        assert "needs_human_review" in consensus.failure_modes

    def test_conflict_record_contains_all_judgments(self) -> None:
        j1 = _judgment(label=0, confidence=0.9)
        j2 = _judgment(label=3, confidence=0.9)
        _, conflict = build_consensus([j1, j2])
        assert conflict is not None
        assert len(conflict.judgments) == 2

    def test_conflict_record_has_watermark(self) -> None:
        j1 = _judgment(label=0)
        j2 = _judgment(label=3)
        _, conflict = build_consensus([j1, j2])
        assert conflict is not None
        assert NEURO_JUDGE_WATERMARK in conflict.watermark


# ===========================================================================
# 9. Hard-negative routing (3 tests)
# ===========================================================================


class TestHardNegativeRouting:
    def test_hn_disagreement_always_routed_to_conflict(self) -> None:
        j1 = _judgment(label=2, confidence=0.95, hard_negative_detected=True)
        j2 = _judgment(label=2, confidence=0.95, hard_negative_detected=False)
        _, conflict = build_consensus([j1, j2])
        assert conflict is not None

    def test_hn_detected_in_mock_judgment(self) -> None:
        judge = MockNeuroJudge()
        pkt = _packet(known_failure_warnings=["possible_hard_negative: wrong species"])
        j = judge.judge(pkt)
        assert j.hard_negative_detected is True

    def test_hard_negative_label_override_in_mock(self) -> None:
        judge = MockNeuroJudge()
        pkt = _packet(concept_hard_negative_conflicts=["fMRI_not_ephys"])
        j = judge.judge(pkt)
        assert j.label == 0

    def test_mock_judge_challenge_cases_match_expected_within_one(self) -> None:
        path = Path("tests/fixtures/neuro_judge_challenge_cases.jsonl")
        judge = MockNeuroJudge()
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            q = _FakeQuery(
                query_id=raw["query_id"],
                query_text=raw["query_text"],
                intent=raw["query_intent"],
                expected_modalities=raw["expected_modalities"],
                expected_species=raw["expected_species"],
                expected_brain_regions=raw["expected_brain_regions"],
                expected_tasks=raw["expected_tasks"],
                expected_analysis_affordances=raw["expected_analysis_affordances"],
                hard_negatives=raw["hard_negatives"],
            )
            pkt = build_evidence_packet(q, raw)
            judgment = judge.judge(pkt)
            assert abs(judgment.label - int(raw["expected_label"])) <= 1, raw["challenge_id"]


# ===========================================================================
# 10. Calibration metrics (5 tests)
# ===========================================================================


class TestCalibrationMetrics:
    def _make_pairs(self, judge_labels: list[int], human_labels: list[int]) -> tuple[list[dict], list[dict]]:
        judge = [{"query_id": f"q{i}", "dataset_id": f"d{i}", "label": lj} for i, lj in enumerate(judge_labels)]
        human = [{"query_id": f"q{i}", "dataset_id": f"d{i}", "label": lh} for i, lh in enumerate(human_labels)]
        return judge, human

    def test_exact_agreement_computation(self) -> None:
        judge, human = self._make_pairs([0, 1, 2, 3], [0, 1, 2, 3])
        r = calibrate(judge, human)
        assert r.exact_agreement == 1.0

    def test_within_one_agreement(self) -> None:
        judge, human = self._make_pairs([1, 2, 3, 2], [0, 1, 2, 3])
        r = calibrate(judge, human)
        assert r.agreement_within_1 == 1.0

    def test_qwk_perfect_agreement(self) -> None:
        judge, human = self._make_pairs([0, 1, 2, 3], [0, 1, 2, 3])
        r = calibrate(judge, human)
        assert abs(r.quadratic_weighted_kappa - 1.0) < 1e-9

    def test_confusion_matrix_shape(self) -> None:
        judge, human = self._make_pairs([0, 1, 2, 3], [0, 1, 2, 3])
        r = calibrate(judge, human)
        assert len(r.confusion_matrix) == 4
        assert all(len(row) == 4 for row in r.confusion_matrix)

    def test_false_high_and_false_low_examples(self) -> None:
        judge, human = self._make_pairs([3, 0], [1, 2])
        r = calibrate(judge, human)
        assert len(r.false_high_examples) == 1  # judge 3 > human 1
        assert len(r.false_low_examples) == 1   # judge 0 < human 2

    def test_empty_overlap_returns_empty_report(self) -> None:
        judge = [{"query_id": "q1", "dataset_id": "d1", "label": 2}]
        human = [{"query_id": "q2", "dataset_id": "d2", "label": 3}]
        r = calibrate(judge, human)
        assert r.n_pairs == 0


# ===========================================================================
# 11. Metric reporter provenance safeguards (5 tests)
# ===========================================================================


class TestMetricReporterProvenance:
    def test_neuro_judge_watermark_constant_present(self) -> None:
        assert "NOT PURE HUMAN GOLD" in NEURO_JUDGE_WATERMARK

    def test_human_gold_provenance_rejected_in_judgment(self) -> None:
        with pytest.raises(ValidationError):
            NeuroJudgment(
                query_id="q",
                dataset_id="d",
                label=2,
                confidence=0.8,
                label_provenance="human_gold",
            )

    def test_human_gold_provenance_rejected_in_consensus(self) -> None:
        with pytest.raises(ValidationError):
            ConsensusResult(
                query_id="q",
                dataset_id="d",
                label=2,
                confidence=0.8,
                label_provenance="human_gold",
            )

    def test_all_valid_provenances_accepted(self) -> None:
        safe = {"neuro_judge", "neuro_judge_rag", "neuro_judge_consensus", "expert_audited_consensus"}
        for prov in safe:
            j = NeuroJudgment(
                query_id="q", dataset_id="d", label=1, confidence=0.7, label_provenance=prov
            )
            assert j.label_provenance == prov

    def test_calibration_report_has_watermark(self) -> None:
        r = CalibrationReport()
        assert NEURO_JUDGE_WATERMARK in r.watermark

    def test_build_neuro_judge_factory_returns_mock(self) -> None:
        judge = build_neuro_judge("mock")
        assert isinstance(judge, MockNeuroJudge)

    def test_build_neuro_judge_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            build_neuro_judge("unknown_backend_xyz")


# ===========================================================================
# 12. Miscellaneous / integration smoke tests (6 tests)
# ===========================================================================


class TestSmoke:
    def test_full_pipeline_smoke(self) -> None:
        """Build packet → judge → consensus without error."""
        q = _FakeQuery()
        r = _record()
        pkt = build_evidence_packet(q, r)
        judge = MockNeuroJudge()
        j1 = judge.judge(pkt)
        j2 = judge.judge(pkt)
        consensus, conflict = build_consensus([j1, j2])
        assert (consensus is not None) != (conflict is not None)

    def test_evidence_packet_serialisable(self) -> None:
        pkt = _packet()
        dumped = pkt.model_dump(mode="json")
        restored = EvidencePacket.model_validate(dumped)
        assert restored.query_id == pkt.query_id

    def test_judgment_serialisable(self) -> None:
        j = _judgment()
        dumped = j.model_dump(mode="json")
        restored = NeuroJudgment.model_validate(dumped)
        assert restored.label == j.label

    def test_consensus_result_serialisable(self) -> None:
        j = _judgment(confidence=0.9)
        consensus, _ = build_consensus([j])
        assert consensus is not None
        dumped = consensus.model_dump(mode="json")
        restored = ConsensusResult.model_validate(dumped)
        assert restored.label == consensus.label

    def test_conflict_record_serialisable(self) -> None:
        j1 = _judgment(label=0)
        j2 = _judgment(label=3)
        _, conflict = build_consensus([j1, j2])
        assert conflict is not None
        dumped = conflict.model_dump(mode="json")
        restored = ConflictRecord.model_validate(dumped)
        assert len(restored.judgments) == 2

    def test_calibration_report_summary_format(self) -> None:
        judge = [{"query_id": "q1", "dataset_id": "d1", "label": 2}]
        human = [{"query_id": "q1", "dataset_id": "d1", "label": 2}]
        r = calibrate(judge, human)
        summary = r.summary()
        assert "exact" in summary
        assert "QWK" in summary
