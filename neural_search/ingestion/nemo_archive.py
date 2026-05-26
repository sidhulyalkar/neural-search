"""NeMO Archive connector for BRAIN Initiative neurogenomic data."""

from __future__ import annotations

import argparse
import json
from typing import Any

import httpx

from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL
from neural_search.ingestion.live import (
    print_cli_error,
    print_normalized_records,
    save_dataset_records,
    save_raw_response,
)
from neural_search.normalized import stable_normalized_id
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord, UsabilityFlags

# NeMO Archive API endpoints
NEMO_API_BASE = "https://assets.nemoarchive.org/api/v1"

# Curated NeMO Archive datasets (BRAIN Initiative Cell Census Network)
NEMO_DATASETS = [
    {
        "id": "nemo_biccn_mouse_brain",
        "title": "BICCN Mouse Brain Cell Atlas",
        "description": """
            The BRAIN Initiative Cell Census Network (BICCN) comprehensive single-cell
            atlas of the adult mouse brain. Integrates single-nucleus RNA-seq,
            single-nucleus ATAC-seq, and spatial transcriptomics across all major brain
            regions from multiple contributing laboratories.
        """,
        "url": "https://nemoarchive.org/data/biccn-mouse",
        "modalities": ["single_nucleus_rnaseq", "single_nucleus_atacseq",
                       "spatial_transcriptomics", "multiome"],
        "species": ["mouse"],
        "brain_regions": ["whole_brain", "cortex", "hippocampus", "striatum",
                         "thalamus", "cerebellum", "hypothalamus"],
        "data_standards": ["h5ad", "zarr", "loom"],
        "num_cells": 10000000,
        "project": "BICCN",
    },
    {
        "id": "nemo_biccn_human_cortex",
        "title": "BICCN Human Cortex Cell Atlas",
        "description": """
            Multi-modal human cortex cell type atlas from the BRAIN Initiative.
            Includes single-nucleus RNA-seq, snATAC-seq, and Patch-seq from multiple
            cortical regions including motor cortex, temporal cortex, and prefrontal cortex.
            Postmortem and neurosurgical tissue samples.
        """,
        "url": "https://nemoarchive.org/data/biccn-human",
        "modalities": ["single_nucleus_rnaseq", "single_nucleus_atacseq", "patch_seq"],
        "species": ["human"],
        "brain_regions": ["cortex", "motor_cortex", "temporal_cortex", "prefrontal_cortex"],
        "data_standards": ["h5ad", "NWB"],
        "num_cells": 2000000,
        "project": "BICCN",
    },
    {
        "id": "nemo_biccn_marmoset",
        "title": "BICCN Marmoset Motor Cortex",
        "description": """
            Single-nucleus transcriptomic atlas of marmoset primary motor cortex
            for cross-species comparison of cell types. Part of BRAIN Initiative
            motor cortex study alongside human and mouse datasets.
        """,
        "url": "https://nemoarchive.org/data/biccn-nhp",
        "modalities": ["single_nucleus_rnaseq"],
        "species": ["marmoset"],
        "brain_regions": ["motor_cortex"],
        "data_standards": ["h5ad"],
        "num_cells": 200000,
        "project": "BICCN",
    },
    {
        "id": "nemo_bican_whole_brain",
        "title": "BICAN Whole Mouse Brain Atlas",
        "description": """
            Next-generation whole mouse brain cell atlas from the BRAIN Initiative
            Cell Atlas Network (BICAN). Single-cell multiome (RNA + ATAC) profiling
            at unprecedented scale covering all brain structures.
        """,
        "url": "https://nemoarchive.org/data/bican",
        "modalities": ["multiome", "single_nucleus_rnaseq", "single_nucleus_atacseq"],
        "species": ["mouse"],
        "brain_regions": ["whole_brain"],
        "data_standards": ["h5ad", "zarr"],
        "num_cells": 30000000,
        "project": "BICAN",
    },
    {
        "id": "nemo_devhu_brain",
        "title": "Developing Human Brain Atlas",
        "description": """
            Single-cell atlas of the developing human brain across prenatal and
            early postnatal stages. Maps cell type emergence, differentiation
            trajectories, and regional specification during brain development.
        """,
        "url": "https://nemoarchive.org/data/developing-brain",
        "modalities": ["single_cell_rnaseq", "single_nucleus_rnaseq",
                       "spatial_transcriptomics"],
        "species": ["human"],
        "brain_regions": ["whole_brain", "cortex", "cerebellum"],
        "data_standards": ["h5ad"],
        "developmental_stage": ["prenatal", "postnatal"],
        "project": "DevHuBrain",
    },
    {
        "id": "nemo_snm3seq",
        "title": "Single-Nucleus Methyl-3C-seq Brain Atlas",
        "description": """
            Multi-omic atlas profiling DNA methylation and chromatin conformation
            at single-cell resolution in mouse and human brain. Combines single-nucleus
            methyl-seq with Hi-C to map regulatory architecture of brain cell types.
        """,
        "url": "https://nemoarchive.org/data/epigenomics",
        "modalities": ["methylation", "single_nucleus_rnaseq"],
        "species": ["mouse", "human"],
        "brain_regions": ["cortex", "hippocampus"],
        "data_standards": ["h5ad", "cooler"],
        "num_cells": 500000,
        "project": "BICCN",
    },
    {
        "id": "nemo_psych_encode",
        "title": "PsychENCODE Brain Transcriptomics",
        "description": """
            Large-scale transcriptomic and epigenomic profiling of postmortem
            human brain tissue from neurotypical individuals and those with
            psychiatric disorders including schizophrenia, bipolar disorder,
            and autism spectrum disorder.
        """,
        "url": "https://nemoarchive.org/data/psychencode",
        "modalities": ["bulk_rnaseq", "single_nucleus_rnaseq", "chip_seq"],
        "species": ["human"],
        "brain_regions": ["cortex", "prefrontal_cortex", "temporal_cortex"],
        "data_standards": ["h5ad", "bam"],
        "clinical_conditions": ["schizophrenia", "bipolar_disorder", "autism"],
        "project": "PsychENCODE",
    },
    {
        "id": "nemo_brain_aging",
        "title": "Brain Aging Cell Atlas",
        "description": """
            Single-cell atlas of aging mouse and human brain mapping cell type
            specific transcriptional changes during aging. Includes young, adult,
            and aged timepoints with matched samples across brain regions.
        """,
        "url": "https://nemoarchive.org/data/aging",
        "modalities": ["single_nucleus_rnaseq"],
        "species": ["mouse", "human"],
        "brain_regions": ["cortex", "hippocampus", "hypothalamus"],
        "data_standards": ["h5ad"],
        "developmental_stage": ["adult", "aged"],
        "project": "SEA-AD",
    },
    {
        "id": "nemo_sea_ad",
        "title": "Seattle Alzheimer's Disease Brain Cell Atlas",
        "description": """
            Comprehensive single-cell atlas of Alzheimer's disease progression
            in human brain. Multimodal profiling across disease stages including
            cognitively normal, MCI, and Alzheimer's dementia samples.
        """,
        "url": "https://nemoarchive.org/data/sea-ad",
        "modalities": ["single_nucleus_rnaseq", "single_nucleus_atacseq"],
        "species": ["human"],
        "brain_regions": ["cortex", "hippocampus", "entorhinal_cortex"],
        "data_standards": ["h5ad"],
        "clinical_conditions": ["alzheimers", "mci"],
        "project": "SEA-AD",
    },
    {
        "id": "nemo_spatial_brain_map",
        "title": "Spatial Brain Cell Map",
        "description": """
            High-resolution spatial transcriptomics atlas of mouse brain using
            MERFISH, Visium, and Slide-seq technologies. Maps cell type locations
            and spatial organization across coronal brain sections.
        """,
        "url": "https://nemoarchive.org/data/spatial",
        "modalities": ["spatial_transcriptomics", "MERFISH"],
        "species": ["mouse"],
        "brain_regions": ["whole_brain"],
        "data_standards": ["zarr", "h5ad"],
        "num_cells": 5000000,
        "project": "BICCN",
    },
]


def normalize_nemo_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Normalize a NeMO Archive dataset entry to standard format."""
    source_id = dataset["id"]
    title = dataset["title"]
    description = dataset.get("description", "").strip()

    return {
        "source": "nemo",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": dataset.get("url", "https://nemoarchive.org/"),
        "license": "BRAIN Initiative Data Sharing Policy",
        "species": dataset.get("species", []),
        "modalities": dataset.get("modalities", []),
        "brain_regions": dataset.get("brain_regions", []),
        "tasks": [],
        "behaviors": [],
        "data_standards": dataset.get("data_standards", []),
        "has_behavior": False,
        "has_trials": False,
        "has_raw_data": True,
        "has_processed_data": True,
        "metadata_json": {
            "raw_source": "nemo",
            "num_cells": dataset.get("num_cells"),
            "project": dataset.get("project"),
            "clinical_conditions": dataset.get("clinical_conditions", []),
            "developmental_stage": dataset.get("developmental_stage", []),
        },
    }


def normalize_nemo_record(
    dataset: dict[str, Any],
    raw_payload_path: str | None = None,
) -> NormalizedDatasetRecord:
    """Normalize NeMO dataset to v0.3 provenance-aware schema."""
    legacy = normalize_nemo_dataset(dataset)
    source_value = f"{legacy.get('title')} {legacy.get('description')}"

    species_labels = [
        EvidenceLabel(
            id=s,
            label=s.replace("_", " ").title(),
            label_type="species",
            confidence=0.95,
            source_field="curated_metadata",
            source_value=source_value[:200],
        )
        for s in legacy.get("species", [])
    ]

    modality_labels = [
        EvidenceLabel(
            id=m,
            label=m.replace("_", " ").title(),
            label_type="modality",
            confidence=0.95,
            source_field="curated_metadata",
            source_value=source_value[:200],
        )
        for m in legacy.get("modalities", [])
    ]

    brain_region_labels = [
        EvidenceLabel(
            id=r,
            label=r.replace("_", " ").title(),
            label_type="brain_region",
            confidence=0.90,
            source_field="curated_metadata",
            source_value=source_value[:200],
        )
        for r in legacy.get("brain_regions", [])
    ]

    data_standard_labels = [
        EvidenceLabel(
            id=s,
            label=s,
            label_type="data_standard",
            confidence=0.95,
            source_field="curated_metadata",
            source_value=source_value[:200],
        )
        for s in legacy.get("data_standards", [])
    ]

    return NormalizedDatasetRecord(
        dataset_id=stable_normalized_id("dataset", "nemo", legacy["source_id"]),
        source="nemo",
        source_id=legacy["source_id"],
        title=legacy["title"],
        description=legacy.get("description"),
        url=legacy.get("url"),
        raw_payload_path=raw_payload_path,
        species=species_labels,
        modalities=modality_labels,
        brain_regions=brain_region_labels,
        tasks=[],
        behavioral_events=[],
        data_standards=data_standard_labels,
        usability_flags=UsabilityFlags(
            has_trials=False,
            has_behavior=False,
            has_neural_data=True,
            has_raw_data=True,
            has_processed_data=True,
            has_standard_format=any(
                s in ["NWB", "h5ad", "BIDS"] for s in legacy.get("data_standards", [])
            ),
        ),
        missing_fields=[],
    )


def get_curated_nemo_datasets() -> list[dict[str, Any]]:
    """Return curated list of NeMO Archive datasets."""
    return NEMO_DATASETS


def records_from_curated(limit: int | None = None) -> list[dict[str, Any]]:
    """Get normalized records from curated dataset list."""
    datasets = get_curated_nemo_datasets()
    if limit:
        datasets = datasets[:limit]
    return [normalize_nemo_dataset(d) for d in datasets]


def normalized_records_from_curated(
    limit: int | None = None,
) -> list[NormalizedDatasetRecord]:
    """Get v0.3 normalized records from curated dataset list."""
    datasets = get_curated_nemo_datasets()
    if limit:
        datasets = datasets[:limit]
    return [normalize_nemo_record(d) for d in datasets]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ingestion.nemo_archive")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of datasets")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--save", action="store_true", help="Save to database")
    parser.add_argument("--save-raw", action="store_true", help="Save raw JSON")
    parser.add_argument("--force", action="store_true", help="Force overwrite")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    args = parser.parse_args(argv)

    try:
        datasets = get_curated_nemo_datasets()
        if args.limit:
            datasets = datasets[:args.limit]

        if args.save_raw:
            raw_path = save_raw_response("nemo", "curated", datasets)
            print(json.dumps({"raw_saved": str(raw_path)}, indent=2))

        records = [normalize_nemo_dataset(d) for d in datasets]
        print_normalized_records(records)

        if args.dry_run or not args.save:
            return 0

        summary = save_dataset_records(records, args.database_url, args.force)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print_cli_error("nemo_archive", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
