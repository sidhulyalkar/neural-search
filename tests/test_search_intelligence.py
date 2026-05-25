from __future__ import annotations

from pathlib import Path

import yaml

from neural_search.intelligence import (
    build_search_coverage_plan,
    plan_search_intelligence,
    write_search_coverage_plan,
)
from neural_search.normalized import make_dataset_id, write_jsonl
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=f"label:{label_type}:{label}",
        label=label,
        label_type=label_type,
        confidence=0.95,
    )


def test_search_intelligence_plan_handles_cross_modal_hard_negatives() -> None:
    plan = plan_search_intelligence(
        "human EEG BCI decoding with behavior without fMRI",
        corpus_profile={
            "data_form_counts": {"eeg_meg": 0, "behavior": 4},
            "underrepresented_data_forms": ["eeg_meg"],
        },
    )

    assert plan.intent == "hard_negative"
    assert plan.mode == "constraint_filter_first"
    assert "eeg_meg" in plan.required_data_forms
    assert "mri" in plan.excluded_data_forms
    assert "preserve_hard_negative_filtering" in plan.quality_checks
    assert "awareness" in plan.retrieval_weights
    assert any("eeg_meg" in warning for warning in plan.warnings)


def test_search_intelligence_plan_promotes_cross_modal_fit() -> None:
    plan = plan_search_intelligence(
        "mouse Neuropixels and calcium imaging population dynamics"
    )

    assert plan.intent == "cross_modal"
    assert plan.mode == "cross_modal_fit"
    assert "extracellular_ephys" in plan.required_data_forms
    assert "optical_imaging" in plan.required_data_forms
    assert "verify_cross_modal_alignment" in plan.quality_checks
    assert plan.retrieval_weights["awareness"] >= 0.2


def test_coverage_plan_prioritizes_missing_data_forms(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    benchmark_path = tmp_path / "benchmark.yaml"
    output_dir = tmp_path / "reports"
    record = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("fixture", "001"),
        source="fixture",
        source_id="001",
        title="Mouse Neuropixels decision behavior",
        description="NWB units, spike times, events, and behavior trials.",
        species=[_label("species", "mouse")],
        modalities=[_label("modality", "neuropixels")],
        behavioral_events=[_label("behavior", "choice")],
        data_standards=[_label("data_standard", "NWB")],
    )
    write_jsonl([record], records_path)
    benchmark_path.write_text(
        yaml.safe_dump(
            {
                "benchmark_queries": [
                    {
                        "id": "q1",
                        "query": "mouse Neuropixels decoding without EEG",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    plan = build_search_coverage_plan(
        records_path,
        benchmark_path,
        target_corpus_count=1,
        target_benchmark_query_count=1,
    )
    paths = write_search_coverage_plan(plan, output_dir)

    gap_ids = {gap.data_form for gap in plan.gaps}
    assert "mri" in gap_ids
    assert "molecular" in gap_ids
    assert "extracellular_ephys" not in gap_ids
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
