from pathlib import Path

from neural_search.corpus.coverage_depth import (
    build_coverage_depth_records,
    render_coverage_depth_report,
    write_coverage_depth_pack,
)
from neural_search.normalized import load_normalized_records
from neural_search.schemas import NormalizedDatasetRecord

ROOT = Path(__file__).resolve().parents[1]
SEED_PATH = ROOT / "data" / "seed" / "coverage_depth_datasets.yaml"


def _labels(record: NormalizedDatasetRecord, field: str) -> set[str]:
    return {label.label for label in getattr(record, field)}


def test_coverage_depth_seed_builds_reviewed_normalized_records():
    records, source_targets = build_coverage_depth_records(SEED_PATH)

    assert len(records) >= 10
    assert all(isinstance(record, NormalizedDatasetRecord) for record in records)
    assert any(target["id"] == "macaque_mt_single_unit" for target in source_targets)

    mt_control = next(record for record in records if record.source_id == "000347")
    assert {"macaque"} <= _labels(mt_control, "species")
    assert {"v1", "v2", "visual_cortex"} <= _labels(mt_control, "brain_regions")

    rat_pfc = next(record for record in records if record.source_id == "000067")
    assert {"rat"} <= _labels(rat_pfc, "species")
    assert {"mPFC", "prefrontal_cortex"} <= _labels(rat_pfc, "brain_regions")
    assert {"lfp", "extracellular_ephys"} <= _labels(rat_pfc, "modalities")

    dorsal_striatum = next(record for record in records if record.source_id == "000559")
    assert {"dorsal_striatum", "dorsolateral_striatum", "striatum"} <= _labels(
        dorsal_striatum,
        "brain_regions",
    )


def test_coverage_depth_report_names_open_mt_gap():
    records, source_targets = build_coverage_depth_records(SEED_PATH)

    report = render_coverage_depth_report(records, source_targets)

    assert "macaque_mt_single_unit" in report
    assert "area MT/MST single-unit" in report
    assert "fmri_glm_analysis" in report


def test_coverage_depth_writer_outputs_loadable_jsonl(tmp_path):
    paths = write_coverage_depth_pack(SEED_PATH, tmp_path)

    records = load_normalized_records(paths["records"])

    assert paths["report"].exists()
    assert paths["source_targets"].exists()
    assert len(records) >= 10
    assert all(isinstance(record, NormalizedDatasetRecord) for record in records)

