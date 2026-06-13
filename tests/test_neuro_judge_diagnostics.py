from __future__ import annotations

from neural_search.eval.neuro_judge.evidence_packet import EvidencePacket
from neural_search.eval.neuro_judge.judge import MockNeuroJudge
from scripts.eval.diagnose_neuro_judge_collapse import diagnose, render_markdown
from scripts.eval.select_neuro_judge_validation_set import select_validation_sample


def _packet(
    query_id: str,
    dataset_id: str,
    *,
    expected_modalities: list[str],
    dataset_modalities: list[str],
    expected_species: list[str] | None = None,
    dataset_species: list[str] | None = None,
    description: str = "Raw extracellular spike sorting data with raw AP-band available.",
    has_raw_data: bool | None = True,
) -> dict:
    return EvidencePacket(
        query_id=query_id,
        query_text="mouse extracellular spike sorting with raw AP-band",
        query_intent="PIPELINE_REUSE",
        expected_species=expected_species or ["mouse"],
        expected_modalities=expected_modalities,
        expected_brain_regions=["visual_cortex"],
        expected_tasks=["visual_stimulation"],
        expected_analysis_affordances=["spike_sorting", "raw_ap_band"],
        dataset_id=dataset_id,
        title=dataset_id,
        description=description,
        dataset_species=dataset_species or ["mouse"],
        dataset_modalities=dataset_modalities,
        dataset_brain_regions=["visual_cortex"],
        dataset_tasks=["visual_stimulation"],
        has_raw_data=has_raw_data,
    ).model_dump(mode="json")


def test_diagnose_reports_label_distribution_and_missing_fields() -> None:
    evidence = [
        _packet("q1", "d1", expected_modalities=["extracellular_ephys"], dataset_modalities=["extracellular_ephys"]),
        _packet("q1", "d2", expected_modalities=["extracellular_ephys"], dataset_modalities=["fmri"], description=""),
    ]
    judge = MockNeuroJudge()
    judgments = [
        judge.judge(EvidencePacket.model_validate(row)).model_dump(mode="json")
        for row in evidence
    ]

    summary = diagnose(evidence, judgments, judgments)
    markdown = render_markdown(summary)

    assert summary["judgments"]["label_distribution"]["0"] == 1
    assert summary["judgments"]["label_distribution"]["3"] == 1
    assert summary["evidence"]["absent_field_counts"]["description"] == 1
    assert "Collapse Assessment" in markdown


def test_select_validation_sample_is_seed_deterministic_and_diverse() -> None:
    evidence = [
        _packet("q1", "d1", expected_modalities=["extracellular_ephys"], dataset_modalities=["extracellular_ephys"]),
        _packet("q1", "d2", expected_modalities=["extracellular_ephys"], dataset_modalities=["fmri"], description=""),
        _packet("q2", "d3", expected_modalities=["fmri"], dataset_modalities=["fmri"], dataset_species=["human"]),
    ]
    judge = MockNeuroJudge()
    judgments = [
        judge.judge(EvidencePacket.model_validate(row)).model_dump(mode="json")
        for row in evidence
    ]
    candidates = [
        {"query_id": "q1", "record_id": "d1", "rank": 1},
        {"query_id": "q1", "record_id": "d2", "rank": 50},
        {"query_id": "q2", "record_id": "d3", "rank": 5},
    ]

    sample_a, summary_a = select_validation_sample(
        evidence,
        judgments,
        candidates,
        n=2,
        seed=7,
        require_diversity=True,
        include_high_impact=True,
        include_missing_evidence=True,
    )
    sample_b, summary_b = select_validation_sample(
        evidence,
        judgments,
        candidates,
        n=2,
        seed=7,
        require_diversity=True,
        include_high_impact=True,
        include_missing_evidence=True,
    )

    assert sample_a == sample_b
    assert summary_a == summary_b
    assert len(sample_a) == 2
    assert any("_selection_reason" in row for row in sample_a)
