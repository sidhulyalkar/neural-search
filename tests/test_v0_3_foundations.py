import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from neural_search.evaluation.run_benchmark import (
    benchmark_path_for_suite,
    run_full_benchmark,
    write_reports,
)
from neural_search.ingestion.dandi import normalize_dandiset_record
from neural_search.ingestion.openalex import normalize_work_record
from neural_search.ingestion.openneuro import normalize_openneuro_record
from neural_search.normalized import dump_records_jsonl, load_records_jsonl
from neural_search.reports.corpus_report import summarize_corpus, write_corpus_reports
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord
from neural_search.search import search_datasets

FIXTURES = Path(__file__).parent / "fixtures" / "ingestion"


def _json(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_benchmark_suite_selection_writes_suite_specific_latest_json(tmp_path):
    report = run_full_benchmark(benchmark_path_for_suite("real_corpus"), suite="real_corpus")
    paths = write_reports(report, tmp_path / "real_corpus")

    assert report.suite == "real_corpus"
    assert Path(paths["latest"]).name == "latest.json"
    assert json.loads(Path(paths["latest"]).read_text(encoding="utf-8"))["suite"] == "real_corpus"


def test_evidence_label_confidence_validation():
    with pytest.raises(ValidationError):
        EvidenceLabel(
            id="x",
            label="bad confidence",
            label_type="task",
            confidence=1.5,
        )


def test_normalized_schema_roundtrip_jsonl(tmp_path):
    record = normalize_dandiset_record(_json("dandi_record.json"), "raw/dandi.json")
    path = dump_records_jsonl([record], tmp_path / "records.jsonl")

    loaded = load_records_jsonl(path)

    assert loaded == [record]
    assert isinstance(loaded[0], NormalizedDatasetRecord)
    assert loaded[0].raw_payload_path == "raw/dandi.json"
    assert loaded[0].species[0].confidence <= 1.0


def test_fixture_backed_normalizers_emit_provenance_records():
    dandi = normalize_dandiset_record(_json("dandi_record.json"))
    openneuro = normalize_openneuro_record(_json("openneuro_record.json"))
    openalex = normalize_work_record(_json("openalex_work.json"))

    assert dandi.dataset_id == "dataset:dandi:000001"
    assert dandi.usability_flags.has_standard_format is True
    assert openneuro.source == "openneuro"
    assert any(label.id == "eeg" for label in openneuro.modalities)
    assert openalex.paper_id == "paper:openalex:W123"
    assert any(label.label_type == "modality" for label in openalex.extracted_labels)


def test_corpus_report_generation_from_tiny_fixture(tmp_path):
    records = [
        normalize_dandiset_record(_json("dandi_record.json")),
        normalize_work_record(_json("openalex_work.json")),
    ]
    summary = summarize_corpus(records)
    paths = write_corpus_reports(summary, tmp_path)

    assert summary["source_counts"]["dandi"] == 1
    assert "coverage" in paths
    assert (tmp_path / "low_confidence_labels.json").exists()


def test_search_result_exposes_v0_3_score_heads_and_negative_constraints():
    response = search_datasets(
        "visual cortex mouse recordings not EEG",
        datasets=[
            {
                "dataset": {
                    "id": "EEG_NEGATIVE",
                    "source": "demo",
                    "source_id": "EEG_NEGATIVE",
                    "title": "Visual cortex EEG",
                    "description": "Mouse visual cortex EEG recordings",
                    "species": ["mouse"],
                    "modalities": ["eeg"],
                    "brain_regions": ["visual_cortex"],
                    "tasks": ["visual_decision_making"],
                    "behaviors": [],
                    "data_standards": ["BIDS"],
                    "has_behavior": False,
                    "has_trials": False,
                    "metadata_json": {},
                },
                "card": {
                    "dataset_id": "EEG_NEGATIVE",
                    "summary": "Mouse visual cortex EEG recordings",
                    "scientific_labels": {},
                    "analysis_readiness": {"score": 40},
                    "missing_fields": [],
                    "suggested_analyses": [],
                    "provenance": {},
                },
            }
        ],
    )

    assert response.results == []
    assert response.parsed_query["filtered_negative_constraints"] == [
        {"dataset_id": "EEG_NEGATIVE", "violations": ["eeg"]}
    ]
