"""Allen Brain Atlas/Map connector for neurogenomic data."""

from __future__ import annotations

import argparse
import json
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL
from neural_search.ingestion.live import (
    print_cli_error,
    print_normalized_records,
    save_dataset_records,
    save_raw_response,
)
from neural_search.normalized import (
    evidence_label_from_extraction,
    stable_normalized_id,
)
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord, UsabilityFlags

# Allen Brain Map API endpoints
ALLEN_BRAIN_MAP_API = "https://celltypes.brain-map.org/api/v2"
ALLEN_CELL_TYPES_API = "https://celltypes.brain-map.org/api/v2"
ALLEN_MOUSE_CONNECTIVITY_API = "https://connectivity.brain-map.org/api/v2"

# Allen Brain Atlas datasets metadata (curated list since API is complex)
ALLEN_DATASETS = [
    {
        "id": "allen_whole_brain_taxonomy",
        "title": "Allen Whole Brain Cell Type Taxonomy",
        "description": """
            Comprehensive single-cell transcriptomic atlas of the adult mouse brain
            covering all major brain structures. Includes single-nucleus RNA sequencing
            (snRNA-seq) from over 4 million cells, identifying 5,000+ cell types organized
            into a hierarchical taxonomy. Part of the BRAIN Initiative Cell Census Network.
        """,
        "url": "https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas",
        "modalities": ["single_nucleus_rnaseq", "spatial_transcriptomics"],
        "species": ["mouse"],
        "brain_regions": ["whole_brain", "cortex", "hippocampus", "thalamus", "cerebellum",
                         "striatum", "hypothalamus", "midbrain", "hindbrain"],
        "data_standards": ["h5ad", "zarr"],
        "num_cells": 4000000,
        "num_cell_types": 5000,
    },
    {
        "id": "allen_human_mtg",
        "title": "Allen Human MTG Cell Types",
        "description": """
            Single-nucleus RNA sequencing of human middle temporal gyrus (MTG) cortex
            from neurosurgical tissue. Defines human cortical cell type taxonomy with
            detailed transcriptomic profiles. Companion electrophysiology (Patch-seq)
            data available for subset of cell types.
        """,
        "url": "https://portal.brain-map.org/atlases-and-data/rnaseq/human-mtg-smart-seq",
        "modalities": ["single_nucleus_rnaseq", "patch_seq"],
        "species": ["human"],
        "brain_regions": ["temporal_cortex", "mtg"],
        "data_standards": ["h5ad", "NWB"],
        "num_cells": 75000,
        "num_cell_types": 120,
    },
    {
        "id": "allen_mouse_vis_ctx",
        "title": "Allen Mouse Visual Cortex Cell Types",
        "description": """
            Multi-modal cell type characterization of mouse visual cortex combining
            single-cell RNA-seq, single-nucleus RNA-seq, ATAC-seq, and Patch-seq
            electrophysiology. Defines canonical cortical cell type taxonomy with
            transcriptomic, epigenomic, and physiological signatures.
        """,
        "url": "https://portal.brain-map.org/atlases-and-data/rnaseq/mouse-v1-and-alm-smart-seq",
        "modalities": ["single_cell_rnaseq", "single_nucleus_rnaseq",
                       "single_nucleus_atacseq", "patch_seq"],
        "species": ["mouse"],
        "brain_regions": ["visual_cortex", "v1", "alm"],
        "data_standards": ["h5ad", "NWB"],
        "num_cells": 500000,
        "num_cell_types": 100,
    },
    {
        "id": "allen_mouse_whole_cortex",
        "title": "Allen Mouse Whole Cortex & Hippocampus",
        "description": """
            Single-cell transcriptomic atlas spanning all cortical areas and hippocampus
            in adult mouse. 10x Chromium single-nucleus RNA-seq from ~1.2 million cells.
            Defines cortical area-specific cell type compositions and regional variations.
        """,
        "url": "https://portal.brain-map.org/atlases-and-data/rnaseq/mouse-whole-cortex-and-hippocampus-10x",
        "modalities": ["single_nucleus_rnaseq"],
        "species": ["mouse"],
        "brain_regions": ["cortex", "hippocampus"],
        "data_standards": ["h5ad"],
        "num_cells": 1200000,
        "num_cell_types": 300,
    },
    {
        "id": "allen_merfish",
        "title": "Allen Mouse Brain MERFISH",
        "description": """
            Spatial transcriptomics atlas of the adult mouse brain using MERFISH technology.
            Measures expression of 500 genes at single-cell resolution with spatial
            coordinates, enabling cell type identification and spatial organization analysis.
        """,
        "url": "https://portal.brain-map.org/atlases-and-data/bkp/abc-atlas",
        "modalities": ["spatial_transcriptomics", "MERFISH"],
        "species": ["mouse"],
        "brain_regions": ["whole_brain"],
        "data_standards": ["zarr", "h5ad"],
        "num_cells": 10000000,
    },
    {
        "id": "allen_cell_types_ephys",
        "title": "Allen Cell Types Database - Electrophysiology",
        "description": """
            Standardized whole-cell patch-clamp recordings from mouse and human cortical
            neurons with morphological reconstructions. Includes intrinsic membrane
            properties, spike features, and 3D morphology. Part of systematic cell type
            characterization pipeline.
        """,
        "url": "https://celltypes.brain-map.org/",
        "modalities": ["patch_clamp", "morphology"],
        "species": ["mouse", "human"],
        "brain_regions": ["cortex", "visual_cortex"],
        "data_standards": ["NWB", "swc"],
        "num_cells": 2000,
    },
    {
        "id": "allen_mouse_connectivity",
        "title": "Allen Mouse Brain Connectivity Atlas",
        "description": """
            Comprehensive mesoscale connectome of the mouse brain using viral tracing.
            Maps axonal projections from ~2,000 injection sites across the brain,
            providing whole-brain connectivity matrix and projection patterns.
        """,
        "url": "https://connectivity.brain-map.org/",
        "modalities": ["viral_tracing", "connectivity"],
        "species": ["mouse"],
        "brain_regions": ["whole_brain"],
        "data_standards": ["nrrd", "json"],
        "num_experiments": 2000,
    },
    {
        "id": "allen_human_m1",
        "title": "Allen Human Primary Motor Cortex",
        "description": """
            Multimodal characterization of human primary motor cortex (M1) including
            single-nucleus RNA-seq, snATAC-seq, and spatial transcriptomics. Part of
            BRAIN Initiative cross-species motor cortex study. Companion marmoset and
            mouse data available.
        """,
        "url": "https://portal.brain-map.org/atlases-and-data/rnaseq/human-m1-10x",
        "modalities": ["single_nucleus_rnaseq", "single_nucleus_atacseq",
                       "spatial_transcriptomics"],
        "species": ["human"],
        "brain_regions": ["motor_cortex", "m1"],
        "data_standards": ["h5ad"],
        "num_cells": 150000,
        "num_cell_types": 100,
    },
]


def normalize_allen_dataset(dataset: dict[str, Any]) -> dict[str, Any]:
    """Normalize an Allen Brain dataset entry to standard format."""
    source_id = dataset["id"]
    title = dataset["title"]
    description = dataset.get("description", "").strip()

    # Build full text for extraction
    text = f"{title} {description}"

    return {
        "source": "allen",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": dataset.get("url", f"https://portal.brain-map.org/"),
        "license": "Allen Institute Terms of Use",
        "species": dataset.get("species", []),
        "modalities": dataset.get("modalities", []),
        "brain_regions": dataset.get("brain_regions", []),
        "tasks": [],  # Atlas data typically doesn't have behavioral tasks
        "behaviors": [],
        "data_standards": dataset.get("data_standards", []),
        "has_behavior": False,
        "has_trials": False,
        "has_raw_data": True,
        "has_processed_data": True,
        "metadata_json": {
            "raw_source": "allen",
            "num_cells": dataset.get("num_cells"),
            "num_cell_types": dataset.get("num_cell_types"),
            "num_experiments": dataset.get("num_experiments"),
        },
    }


def normalize_allen_record(
    dataset: dict[str, Any],
    raw_payload_path: str | None = None,
) -> NormalizedDatasetRecord:
    """Normalize Allen dataset to v0.3 provenance-aware schema."""
    legacy = normalize_allen_dataset(dataset)
    source_value = f"{legacy.get('title')} {legacy.get('description')}"

    # Create evidence labels from dataset metadata
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
        dataset_id=stable_normalized_id("dataset", "allen", legacy["source_id"]),
        source="allen",
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


def fetch_allen_cell_types(limit: int = 100) -> list[dict[str, Any]]:
    """Fetch cell type specimens from Allen Cell Types API."""
    url = f"{ALLEN_CELL_TYPES_API}/data/query.json"
    params = {
        "criteria": "model::Specimen",
        "include": "structure,donor(transgenic_lines)",
        "num_rows": limit,
    }
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("msg", [])
    except Exception:
        return []


def get_curated_allen_datasets() -> list[dict[str, Any]]:
    """Return curated list of Allen Brain datasets."""
    return ALLEN_DATASETS


def records_from_curated(limit: int | None = None) -> list[dict[str, Any]]:
    """Get normalized records from curated dataset list."""
    datasets = get_curated_allen_datasets()
    if limit:
        datasets = datasets[:limit]
    return [normalize_allen_dataset(d) for d in datasets]


def normalized_records_from_curated(
    limit: int | None = None,
) -> list[NormalizedDatasetRecord]:
    """Get v0.3 normalized records from curated dataset list."""
    datasets = get_curated_allen_datasets()
    if limit:
        datasets = datasets[:limit]
    return [normalize_allen_record(d) for d in datasets]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ingestion.allen_brain")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of datasets")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    parser.add_argument("--save", action="store_true", help="Save to database")
    parser.add_argument("--save-raw", action="store_true", help="Save raw JSON")
    parser.add_argument("--force", action="store_true", help="Force overwrite")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    args = parser.parse_args(argv)

    try:
        datasets = get_curated_allen_datasets()
        if args.limit:
            datasets = datasets[:args.limit]

        if args.save_raw:
            raw_path = save_raw_response("allen", "curated", datasets)
            print(json.dumps({"raw_saved": str(raw_path)}, indent=2))

        records = [normalize_allen_dataset(d) for d in datasets]
        print_normalized_records(records)

        if args.dry_run or not args.save:
            return 0

        summary = save_dataset_records(records, args.database_url, args.force)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print_cli_error("allen_brain", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
