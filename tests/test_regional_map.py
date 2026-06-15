from pathlib import Path

from neural_search.corpus.regional_map import (
    build_regional_map,
    load_dataset_records,
    load_region_targets,
    render_regional_map_report,
    write_regional_map_artifacts,
)
from neural_search.normalized import write_jsonl
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "data" / "config" / "regional_map_targets.yaml"


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=f"label:{label_type}:{label}",
        label=label,
        label_type=label_type,
        confidence=0.95,
        evidence_text=label,
    )


def test_regional_map_tracks_verified_and_candidate_regions():
    records = [
        NormalizedDatasetRecord(
            dataset_id="dataset:test:verified",
            source="dandi",
            source_id="verified",
            title="Rat mPFC LFP theta",
            species=[_label("species", "rat")],
            modalities=[_label("modality", "lfp")],
            brain_regions=[
                _label("brain_region", "mPFC"),
                _label("brain_region", "prefrontal_cortex"),
            ],
        ),
        NormalizedDatasetRecord(
            dataset_id="dataset:test:candidate",
            source="dandi",
            source_id="candidate",
            title="Hippocampus calcium imaging with CA1 place cells",
            species=[_label("species", "mouse")],
            modalities=[_label("modality", "calcium_imaging")],
        ),
    ]
    targets = load_region_targets(TARGETS)

    regional_map = build_regional_map(records, targets)

    assert regional_map["summary"]["records_with_verified_regions"] == 1
    assert regional_map["summary"]["regionless_records_with_candidate_mentions"] == 1
    assert regional_map["by_source"]["dandi"]["mpfc"] == 1
    assert "hippocampus" in regional_map["candidate_region_mentions"]
    assert "ca1" in regional_map["candidate_region_mentions"]
    assert "cortical" in regional_map["system_totals"]
    assert regional_map["frontend_regions"][0]["label"]
    assert regional_map["frontend_regions"][0]["system"]


def test_regional_map_report_includes_review_queue():
    records = [
        NormalizedDatasetRecord(
            dataset_id="dataset:test:barrel",
            source="dandi",
            source_id="barrel",
            title="Mouse barrel cortex electrophysiology",
            modalities=[_label("modality", "extracellular_ephys")],
        ),
    ]
    regional_map = build_regional_map(records, load_region_targets(TARGETS))

    report = render_regional_map_report(regional_map)

    assert "Regional Coverage Map" in report
    assert "System Coverage" in report
    assert "barrel_cortex" in report
    assert "dataset:test:barrel" in report


def test_regional_map_writer_outputs_json_report_and_queue(tmp_path):
    records = [
        NormalizedDatasetRecord(
            dataset_id="dataset:test:visual",
            source="openneuro",
            source_id="visual",
            title="Task fMRI visual cortex localizer",
            species=[_label("species", "human")],
            modalities=[_label("modality", "fmri")],
            brain_regions=[_label("brain_region", "visual_cortex")],
        ),
    ]
    records_path = tmp_path / "records.jsonl"
    write_jsonl(records, records_path)

    outputs = write_regional_map_artifacts([records_path], TARGETS, tmp_path / "regional")
    loaded = load_dataset_records([records_path])

    assert len(loaded) == 1
    assert outputs["json"].exists()
    assert outputs["report"].exists()
    assert outputs["review_queue"].exists()
