"""Realistic fixture corpus for adaptive search-intelligence evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from neural_search.normalized import make_dataset_id, write_jsonl
from neural_search.schemas import (
    AnalysisAffordance,
    EvidenceLabel,
    NormalizedDatasetRecord,
    UsabilityFlags,
)

FIXTURE_CREATED_AT = "2026-05-24T00:00:00+00:00"


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=f"label:{label_type}:{label.replace(' ', '_').lower()}",
        label=label,
        label_type=label_type,
        confidence=0.95,
        evidence_text=f"Task 23 fixture label: {label}",
        source_field=label_type,
        source_value=label,
        extractor_name="neural_search.task23_fixture",
        extractor_version="v0.7.0",
    )


def _affordance(analysis_id: str, evidence: str) -> AnalysisAffordance:
    return AnalysisAffordance(
        analysis_id=analysis_id,
        support_level="high",
        confidence=0.9,
        required_fields_present=["metadata", "primary_data"],
        helpful_fields_present=["source_provenance"],
        evidence=[evidence],
        detector_name="task23_fixture_affordance",
        detector_version="v0.7.0",
    )


def _record(
    *,
    source: str,
    source_id: str,
    title: str,
    description: str,
    species: list[str],
    modalities: list[str],
    brain_regions: list[str],
    tasks: list[str],
    behaviors: list[str],
    standards: list[str],
    file_formats: list[str],
    analysis_ids: list[str],
    flags: dict[str, bool],
) -> NormalizedDatasetRecord:
    return NormalizedDatasetRecord(
        dataset_id=make_dataset_id(source, source_id),
        source=source,
        source_id=source_id,
        title=title,
        description=description,
        url=f"https://example.org/{source}/{source_id}",
        species=[_label("species", item) for item in species],
        modalities=[_label("modality", item) for item in modalities],
        brain_regions=[_label("brain_region", item) for item in brain_regions],
        tasks=[_label("task", item) for item in tasks],
        behavioral_events=[_label("behavior", item) for item in behaviors],
        data_standards=[_label("data_standard", item) for item in standards],
        file_formats=[_label("file_format", item) for item in file_formats],
        usability_flags=UsabilityFlags(**flags),
        analysis_affordances=[
            _affordance(analysis_id, f"{title} supports {analysis_id}")
            for analysis_id in analysis_ids
        ],
        created_at=FIXTURE_CREATED_AT,
        extractor_version="v0.7.0",
    )


def build_realistic_fixture_records() -> list[NormalizedDatasetRecord]:
    """Build a compact but broad neuroscience fixture corpus."""

    return [
        _record(
            source="microns",
            source_id="visual_connectome_v1",
            title="MICrONS visual cortex connectomics and morphology",
            description=(
                "Mouse visual cortex electron microscopy connectome with cells, "
                "edges, synapses, morphology traces, and circuit mapping metadata."
            ),
            species=["mouse"],
            modalities=["electron_microscopy", "morphology", "tracing"],
            brain_regions=["visual_cortex"],
            tasks=["natural_scene_viewing"],
            behaviors=[],
            standards=["zarr", "swc", "ngff"],
            file_formats=["zarr", "swc"],
            analysis_ids=["circuit_mapping", "connectivity", "morphology_analysis"],
            flags={"has_neural_data": True, "has_standard_format": True, "has_raw_data": True},
        ),
        _record(
            source="cellxgene",
            source_id="human_cortex_single_cell_v1",
            title="Human cortex single-cell transcriptomics atlas",
            description=(
                "Human cortical single cell RNA transcriptomics with cells, genes, "
                "metadata, cell type labels, and spatial transcriptomics annotations."
            ),
            species=["human"],
            modalities=["single_cell_rna", "transcriptomics", "spatial_transcriptomics"],
            brain_regions=["cortex"],
            tasks=[],
            behaviors=[],
            standards=["h5ad", "zarr"],
            file_formats=["h5ad"],
            analysis_ids=["cell_type_mapping", "differential_expression", "spatial_mapping"],
            flags={"has_neural_data": True, "has_standard_format": True, "has_processed_data": True},
        ),
        _record(
            source="dandi",
            source_id="patch_clamp_mouse_cortex_v1",
            title="Mouse cortical patch clamp intracellular electrophysiology",
            description=(
                "NWB intracellular electrophysiology with patch clamp sweeps, membrane "
                "voltage, current clamp, voltage clamp, and excitability metadata."
            ),
            species=["mouse"],
            modalities=["patch_clamp", "intracellular_ephys"],
            brain_regions=["motor_cortex"],
            tasks=[],
            behaviors=[],
            standards=["NWB", "DANDI"],
            file_formats=["nwb"],
            analysis_ids=["cellular_physiology", "excitability_analysis"],
            flags={"has_neural_data": True, "has_standard_format": True, "has_raw_data": True},
        ),
        _record(
            source="modeldb",
            source_id="hippocampal_biophysical_model_v1",
            title="Hippocampal biophysical simulation model",
            description=(
                "Computational neural simulation with model parameters, outputs, "
                "biophysical mechanisms, spike outputs, and model comparison metadata."
            ),
            species=["rat"],
            modalities=["model_output", "simulation"],
            brain_regions=["hippocampus"],
            tasks=[],
            behaviors=[],
            standards=["json", "hdf5"],
            file_formats=["json", "hdf5"],
            analysis_ids=["model_comparison", "parameter_inference", "mechanistic_modeling"],
            flags={"has_neural_data": True, "has_standard_format": True, "has_processed_data": True},
        ),
        _record(
            source="openneuro",
            source_id="ds_meg_language_v1",
            title="Human MEG language time-frequency dataset",
            description=(
                "BIDS MEG recordings with channels, events, participants, sampling "
                "rate, language stimuli, and time frequency analysis metadata."
            ),
            species=["human"],
            modalities=["meg"],
            brain_regions=["temporal_cortex"],
            tasks=["language_task"],
            behaviors=["button_press"],
            standards=["BIDS", "OpenNeuro"],
            file_formats=["fif", "tsv", "json"],
            analysis_ids=["time_frequency", "connectivity"],
            flags={
                "has_trials": True,
                "has_behavior": True,
                "has_neural_data": True,
                "has_event_timestamps": True,
                "has_standard_format": True,
                "has_raw_data": True,
            },
        ),
        _record(
            source="openneuro",
            source_id="ds_fmri_connectivity_v1",
            title="Human fMRI connectivity and behavior dataset",
            description=(
                "BIDS fMRI BOLD images with participants, events, behavior labels, "
                "structural MRI, and functional connectivity time series."
            ),
            species=["human"],
            modalities=["fmri", "mri"],
            brain_regions=["whole_brain"],
            tasks=["resting_state", "decision_making"],
            behaviors=["choice", "reaction_time"],
            standards=["BIDS", "OpenNeuro", "NIfTI"],
            file_formats=["nii.gz", "tsv", "json"],
            analysis_ids=["connectivity", "encoding_modeling", "clinical_prediction"],
            flags={
                "has_trials": True,
                "has_behavior": True,
                "has_neural_data": True,
                "has_event_timestamps": True,
                "has_standard_format": True,
                "has_raw_data": True,
            },
        ),
    ]


def build_realistic_fixture_benchmark() -> dict[str, Any]:
    """Build benchmark queries for the realistic fixture corpus."""

    return {
        "benchmark_queries": [
            {
                "id": "task23_connectomics_mapping",
                "query": "connectome morphology tracing for circuit mapping without fMRI",
                "expected_dataset_ids": [make_dataset_id("microns", "visual_connectome_v1")],
                "expected_modalities_any": ["electron_microscopy", "morphology", "tracing"],
                "expected_analysis_any": ["circuit_mapping", "connectivity"],
                "hard_negative_dataset_ids": [
                    make_dataset_id("openneuro", "ds_fmri_connectivity_v1")
                ],
                "minimum_precision_at_5": 0.2,
                "minimum_label_recall_at_10": 0.2,
            },
            {
                "id": "task23_single_cell_mapping",
                "query": "single cell transcriptomics cell type mapping human cortex",
                "expected_dataset_ids": [
                    make_dataset_id("cellxgene", "human_cortex_single_cell_v1")
                ],
                "expected_modalities_any": ["single_cell_rna", "transcriptomics"],
                "expected_species": ["human"],
                "expected_analysis_any": ["cell_type_mapping", "differential_expression"],
                "minimum_precision_at_5": 0.2,
                "minimum_label_recall_at_10": 0.2,
            },
            {
                "id": "task23_patch_clamp_excitability",
                "query": "mouse patch clamp membrane voltage excitability without fMRI",
                "expected_dataset_ids": [make_dataset_id("dandi", "patch_clamp_mouse_cortex_v1")],
                "expected_modalities_any": ["patch_clamp", "intracellular_ephys"],
                "expected_species": ["mouse"],
                "expected_analysis_any": ["cellular_physiology", "excitability_analysis"],
                "hard_negative_dataset_ids": [
                    make_dataset_id("openneuro", "ds_fmri_connectivity_v1")
                ],
                "minimum_precision_at_5": 0.2,
                "minimum_label_recall_at_10": 0.2,
            },
            {
                "id": "task23_computational_model",
                "query": "biophysical simulation model parameter inference hippocampus",
                "expected_dataset_ids": [
                    make_dataset_id("modeldb", "hippocampal_biophysical_model_v1")
                ],
                "expected_modalities_any": ["model_output", "simulation"],
                "expected_regions_any": ["hippocampus"],
                "expected_analysis_any": ["parameter_inference", "mechanistic_modeling"],
                "minimum_precision_at_5": 0.2,
                "minimum_label_recall_at_10": 0.2,
            },
            {
                "id": "task23_meg_time_frequency",
                "query": "human MEG language time frequency with events",
                "expected_dataset_ids": [make_dataset_id("openneuro", "ds_meg_language_v1")],
                "expected_modalities_any": ["meg"],
                "expected_species": ["human"],
                "expected_analysis_any": ["time_frequency"],
                "minimum_precision_at_5": 0.2,
                "minimum_label_recall_at_10": 0.2,
            },
            {
                "id": "task23_fmri_connectivity_no_eeg",
                "query": "human fMRI connectivity behavior without EEG",
                "expected_dataset_ids": [make_dataset_id("openneuro", "ds_fmri_connectivity_v1")],
                "expected_modalities_any": ["fmri", "mri"],
                "expected_species": ["human"],
                "expected_analysis_any": ["connectivity"],
                "hard_negative_dataset_ids": [make_dataset_id("openneuro", "ds_meg_language_v1")],
                "minimum_precision_at_5": 0.2,
                "minimum_label_recall_at_10": 0.2,
            },
        ],
        "metadata": {
            "source": "neural_search.intelligence.fixtures",
            "review_required": False,
            "note": "Task 23 realistic fixture queries with deterministic expected IDs.",
        },
    }


def build_realistic_fixture_judgments() -> list[dict[str, Any]]:
    """Build seed review judgments for the Task 23 fixture benchmark."""

    benchmark = build_realistic_fixture_benchmark()["benchmark_queries"]
    judgments: list[dict[str, Any]] = []
    for query in benchmark:
        for dataset_id in query.get("expected_dataset_ids", []):
            judgments.append(
                {
                    "query_id": query["id"],
                    "query_text": query["query"],
                    "dataset_id": dataset_id,
                    "relevance": "exact",
                    "task_match": 3,
                    "modality_match": 3,
                    "species_match": 3,
                    "analysis_fit": 3,
                    "reviewer_id": "task23_fixture_review",
                    "confidence": 0.9,
                    "notes": "Deterministic fixture judgment for promotion-gate plumbing.",
                }
            )
    return judgments


def write_realistic_fixture_files(
    *,
    records_path: str | Path,
    benchmark_path: str | Path,
    judgments_path: str | Path,
) -> dict[str, str]:
    records_output = write_jsonl(build_realistic_fixture_records(), records_path)
    benchmark_output = Path(benchmark_path)
    benchmark_output.parent.mkdir(parents=True, exist_ok=True)
    benchmark_output.write_text(
        yaml.safe_dump(
            build_realistic_fixture_benchmark(),
            sort_keys=False,
            allow_unicode=False,
        ),
        encoding="utf-8",
    )
    judgments_output = Path(judgments_path)
    judgments_output.parent.mkdir(parents=True, exist_ok=True)
    with judgments_output.open("w", encoding="utf-8") as handle:
        for judgment in build_realistic_fixture_judgments():
            handle.write(json.dumps(judgment, sort_keys=True))
            handle.write("\n")
    return {
        "records": str(records_output),
        "benchmark": str(benchmark_output),
        "judgments": str(judgments_output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build Task 23 realistic search intelligence fixtures."
    )
    parser.add_argument(
        "--records",
        default="data/corpus/normalized/search_intelligence_task23.datasets.jsonl",
    )
    parser.add_argument(
        "--benchmark",
        default="data/eval/benchmark_queries_search_intelligence_task23.yaml",
    )
    parser.add_argument(
        "--judgments",
        default="data/eval/human_judgments_search_intelligence_task23.jsonl",
    )
    args = parser.parse_args(argv)
    paths = write_realistic_fixture_files(
        records_path=args.records,
        benchmark_path=args.benchmark,
        judgments_path=args.judgments,
    )
    print(json.dumps(paths, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
