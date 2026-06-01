from __future__ import annotations

from pathlib import Path

from neural_search.awareness.report import (
    build_awareness_report,
    write_awareness_report,
)
from neural_search.awareness.scoring import score_dataset_awareness
from neural_search.awareness.taxonomy import detect_data_forms, infer_query_awareness
from neural_search.normalized import make_dataset_id
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=f"label:{label_type}:{label}",
        label=label,
        label_type=label_type,
        confidence=0.95,
    )


def test_detects_broad_neuroscience_data_forms_from_language() -> None:
    forms = detect_data_forms(
        "Need fMRI BOLD data, EEG sleep recordings, calcium imaging, and single-cell transcriptomics"
    )

    assert "mri" in forms
    assert "eeg_meg" in forms
    assert "optical_imaging" in forms
    assert "molecular" in forms


def test_query_awareness_tracks_analysis_species_and_exclusions() -> None:
    awareness = infer_query_awareness(
        "Human EEG BCI decoding with behavior but not fMRI"
    )

    assert "eeg_meg" in awareness.requested_data_forms
    assert "behavior" in awareness.requested_data_forms
    assert "mri" in awareness.excluded_data_forms
    assert "bci_decoding" in awareness.analysis_families
    assert "decoding" in awareness.analysis_families
    assert "human" in awareness.species_terms


def test_dataset_awareness_scores_general_cross_modal_fit() -> None:
    query = infer_query_awareness(
        "mouse Neuropixels event aligned decoding with behavior"
    )
    dataset = {
        "id": "DSET",
        "title": "Mouse Neuropixels behavior task",
        "description": "Units, spike times, trials, events, choice labels, and behavior tracking.",
        "species": ["mouse"],
        "modalities": ["neuropixels", "behavior_video"],
        "tasks": ["visual_decision_making"],
        "behaviors": ["choice"],
        "data_standards": ["NWB"],
    }

    score = score_dataset_awareness(dataset, query)

    assert score.score > 0.6
    assert "extracellular_ephys" in score.matched_data_forms
    assert "behavior" in score.matched_data_forms
    assert "event_aligned_analysis" in score.matched_analysis_families
    assert score.warnings == ()


def test_awareness_penalizes_excluded_forms() -> None:
    query = infer_query_awareness("human behavior study without EEG")
    dataset = {
        "id": "EEG_DATASET",
        "title": "Human EEG behavior dataset",
        "description": "EEG channels and task events.",
        "species": ["human"],
        "modalities": ["eeg"],
        "behaviors": ["response"],
    }

    score = score_dataset_awareness(dataset, query)

    assert score.score < 0.5
    assert score.warnings


def test_awareness_report_loads_normalized_records(tmp_path: Path) -> None:
    record = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("fixture", "001"),
        source="fixture",
        source_id="001",
        title="Human BIDS EEG motor imagery",
        description="BIDS EEG channels, events, and participant metadata.",
        species=[_label("species", "human")],
        modalities=[_label("modality", "eeg")],
        tasks=[_label("task", "motor_imagery")],
        data_standards=[_label("data_standard", "BIDS")],
    )
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(record.model_dump_json() + "\n", encoding="utf-8")

    report = build_awareness_report(records_path)
    paths = write_awareness_report(records_path, tmp_path / "report")

    assert report["dataset_count"] == 1
    assert report["data_form_counts"]["eeg_meg"] == 1
    assert Path(paths["json"]).exists()
    assert Path(paths["markdown"]).exists()
