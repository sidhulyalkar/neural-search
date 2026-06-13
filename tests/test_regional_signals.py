from pathlib import Path

from neural_search.corpus.regional_signals import (
    build_acquisition_backlog,
    build_regional_signal_overlay,
    load_signal_rules,
    write_regional_signal_artifacts,
)
from neural_search.normalized import write_jsonl
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "data" / "config" / "regional_map_targets.yaml"
RULES = ROOT / "data" / "config" / "regional_signal_rules.yaml"


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=f"label:{label_type}:{label}",
        label=label,
        label_type=label_type,
        confidence=0.95,
        evidence_text=label,
    )


def test_regional_signal_rules_find_high_yield_shorthand():
    records = [
        {
            "dataset_id": "dataset:dandi:001079",
            "source": "dandi",
            "source_id": "001079",
            "title": "Sharp Wave Ripples in Publicly Available Neuropixels Datasets",
            "species": ["mouse"],
            "modalities": ["neuropixels", "lfp"],
            "brain_regions": [],
        },
        {
            "dataset_id": "dataset:dandi:000030",
            "source": "dandi",
            "source_id": "000030",
            "title": "Allen Brain Observatory Neuropixels recording",
            "species": ["mouse"],
            "modalities": ["neuropixels"],
            "brain_regions": [],
        },
    ]

    overlay = build_regional_signal_overlay(
        records,
        targets=_load_targets(),
        rules=load_signal_rules(RULES),
    )

    by_id = {item["record_id"]: item for item in overlay}
    assert _suggested_regions(by_id["dataset:dandi:001079"]) >= {"hippocampus"}
    assert _suggested_regions(by_id["dataset:dandi:000030"]) >= {"visual_cortex"}


def test_regional_signal_overlay_skips_already_verified_regions():
    record = NormalizedDatasetRecord(
        dataset_id="dataset:test:visual",
        source="dandi",
        source_id="visual",
        title="Allen Brain Observatory visual cortex recording",
        brain_regions=[_label("brain_region", "visual_cortex")],
    )

    overlay = build_regional_signal_overlay(
        [record],
        targets=_load_targets(),
        rules=load_signal_rules(RULES),
    )

    assert overlay == []


def test_acquisition_backlog_prioritizes_high_confidence_regions():
    records = [
        {
            "dataset_id": f"dataset:test:swr{i}",
            "source": "dandi",
            "source_id": f"swr{i}",
            "title": "Sharp Wave Ripples dataset",
            "brain_regions": [],
        }
        for i in range(5)
    ]
    overlay = build_regional_signal_overlay(records, _load_targets(), load_signal_rules(RULES))
    backlog = build_acquisition_backlog(overlay, _load_targets())

    hippocampus = next(item for item in backlog if item["region"] == "hippocampus")
    assert hippocampus["priority"] == "critical"
    assert hippocampus["high_confidence_count"] == 5


def test_regional_signal_writer_outputs_artifacts(tmp_path):
    records = [
        {
            "dataset_id": "dataset:test:pons",
            "source": "dandi",
            "source_id": "pons",
            "title": "Pons8-BIDS-16xdownsampled",
            "brain_regions": [],
        },
    ]
    records_path = tmp_path / "records.jsonl"
    write_jsonl(records, records_path)

    outputs = write_regional_signal_artifacts([records_path], TARGETS, RULES, tmp_path)

    assert outputs["overlay"].exists()
    assert outputs["backlog"].exists()
    assert outputs["report"].exists()
    assert "pons" in outputs["report"].read_text(encoding="utf-8")


def _load_targets():
    from neural_search.corpus.regional_map import load_region_targets

    return load_region_targets(TARGETS)


def _suggested_regions(item: dict) -> set[str]:
    return {suggestion["region"] for suggestion in item["suggested_regions"]}
