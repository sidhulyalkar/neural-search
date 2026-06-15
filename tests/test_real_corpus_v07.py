from __future__ import annotations

import json
from pathlib import Path

import pytest

from neural_search.awareness.scoring import infer_dataset_awareness
from neural_search.corpus.ingest_manifest import ingest_manifest
from neural_search.corpus.manifest import load_manifest
from neural_search.corpus.real_reports import generate_real_corpus_report
from neural_search.file_inspection import inspect_dataset_files
from neural_search.graph.builder import build_graph_from_records, split_records
from neural_search.normalized import load_normalized_records


def test_real_v07_manifest_validates_and_dry_runs(tmp_path: Path) -> None:
    manifest = load_manifest("data/corpus/manifests/real_v07.yaml")

    summary = ingest_manifest(manifest, out_dir=tmp_path, dry_run=True)

    assert manifest.corpus_tag == "real_v07"
    assert len(manifest.entries) >= 7
    assert summary["dry_run"] is True
    assert summary["counts"]["would_dataset"] == 6
    assert summary["counts"]["would_paper"] == 1
    assert summary["counts"]["source:modeldb"] == 1
    assert summary["counts"]["source:cellxgene"] == 1
    assert summary["counts"]["source:microns"] == 1
    assert all(entry.scientific_rationale for entry in manifest.entries)


def test_file_inspection_emits_evidence_backed_claims() -> None:
    claims = inspect_dataset_files(
        ["data/corpus/fixtures/real_v07/bids/ds003505"],
        "dataset:openneuro:ds003505",
    )

    fields = {claim.field for claim in claims}
    assert "data_standards" in fields
    assert "participants" in fields
    assert "has_event_timestamps" in fields
    assert all(claim.evidence for claim in claims)
    assert all(claim.source_path for claim in claims)


def test_real_v07_ingestion_outputs_claim_aware_artifacts(tmp_path: Path) -> None:
    manifest = load_manifest("data/corpus/manifests/real_v07.yaml")
    claims_path = tmp_path / "claims" / "real_v07.claims.jsonl"

    summary = ingest_manifest(
        manifest,
        out_dir=tmp_path / "normalized",
        claims_out=claims_path,
        prefix="real_v07",
    )

    assert summary["counts"]["normalized_datasets"] == 6
    assert summary["counts"]["claims"] >= 10
    records = load_normalized_records(tmp_path / "normalized" / "real_v07.records.jsonl")
    datasets, papers = split_records(records)
    assert len(datasets) == 6
    assert len(papers) == 1
    source_ids = {dataset.dataset_id for dataset in datasets}
    assert "dataset:modeldb:modeldb_87284" in source_ids
    assert "dataset:cellxgene:allen_mouse_motor_cortex_cells" in source_ids
    assert "dataset:microns:minnie65_public" in source_ids
    assert any(
        label.extractor_name.startswith("neural_search.file_inspection")
        for dataset in datasets
        for label in [*dataset.modalities, *dataset.data_standards]
    )

    graph = build_graph_from_records(datasets, papers)
    assert graph.metadata["dataset_count"] == 6
    assert graph.edges

    report = generate_real_corpus_report(
        manifest_path="data/corpus/manifests/real_v07.yaml",
        records_path=tmp_path / "normalized" / "real_v07.records.jsonl",
        claims_path=claims_path,
    )
    assert report["claim_count"] == summary["counts"]["claims"]
    assert report["normalized_dataset_count"] == 6
    assert report["source_counts"]["modeldb"] == 1
    assert report["source_counts"]["cellxgene"] == 1
    assert report["source_counts"]["microns"] == 1


def test_real_v07_new_source_families_map_to_data_forms(tmp_path: Path) -> None:
    manifest = load_manifest("data/corpus/manifests/real_v07.yaml")

    ingest_manifest(manifest, out_dir=tmp_path / "normalized", prefix="real_v07")
    records = load_normalized_records(tmp_path / "normalized" / "real_v07.records.jsonl")
    datasets, _ = split_records(records)
    by_id = {dataset.dataset_id: dataset for dataset in datasets}

    assert "computational_model" in infer_dataset_awareness(
        by_id["dataset:modeldb:modeldb_87284"]
    ).data_forms
    assert "molecular" in infer_dataset_awareness(
        by_id["dataset:cellxgene:allen_mouse_motor_cortex_cells"]
    ).data_forms
    assert "connectomics" in infer_dataset_awareness(
        by_id["dataset:microns:minnie65_public"]
    ).data_forms


def test_generated_real_v07_files_are_valid_jsonl() -> None:
    paths = [
        Path("data/corpus/normalized/real_v07.datasets.jsonl"),
        Path("data/corpus/normalized/real_v07.papers.jsonl"),
        Path("data/corpus/claims/real_v07.claims.jsonl"),
    ]
    if not paths[0].exists():
        pytest.skip("real_v07 data files not present in this environment")
    for path in paths:
        assert path.exists()
        assert path.read_text(encoding="utf-8").strip()
        for line in path.read_text(encoding="utf-8").splitlines():
            assert json.loads(line)
