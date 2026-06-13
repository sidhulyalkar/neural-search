"""Allen Brain Atlas/Map connector for neurogenomic data."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import httpx

from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL
from neural_search.ingestion.live import (
    print_cli_error,
    print_normalized_records,
    save_dataset_records,
    save_raw_response,
)
from neural_search.ingestion.registry import register
from neural_search.normalized import (
    stable_normalized_id,
)
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord, UsabilityFlags

logger = logging.getLogger(__name__)

# Allen Brain Map API endpoints
ALLEN_BRAIN_MAP_API = "https://celltypes.brain-map.org/api/v2"
ALLEN_CELL_TYPES_API = "https://celltypes.brain-map.org/api/v2"
ALLEN_MOUSE_CONNECTIVITY_API = "https://connectivity.brain-map.org/api/v2"
ALLEN_API = "https://api.brain-map.org/api/v2/data"

# Maps Allen CCF structure acronyms to our vocabulary
_STRUCTURE_TO_REGION: dict[str, str] = {
    "VISp": "visual_cortex", "VISl": "visual_cortex", "VISal": "visual_cortex",
    "VISam": "visual_cortex", "VISpm": "visual_cortex", "VISrl": "visual_cortex",
    "VISli": "visual_cortex", "VISa": "visual_cortex", "VISpa": "visual_cortex",
    "CA1": "hippocampus", "CA2": "hippocampus", "CA3": "hippocampus",
    "DG": "hippocampus", "HIP": "hippocampus", "SUB": "hippocampus",
    "MOp": "motor_cortex", "MOs": "motor_cortex",
    "SSp": "somatosensory_cortex", "SSs": "somatosensory_cortex",
    "RSPv": "retrosplenial_cortex", "RSPagl": "retrosplenial_cortex", "RSPd": "retrosplenial_cortex",
    "AUDp": "auditory_cortex",
    "ACA": "anterior_cingulate_cortex",
    "PL": "prefrontal_cortex", "ILA": "prefrontal_cortex", "ORBm": "prefrontal_cortex",
    "LP": "thalamus", "LGd": "thalamus", "POL": "thalamus", "CL": "thalamus", "LD": "thalamus",
    "MRN": "midbrain", "MB": "midbrain", "SC": "superior_colliculus",
}

_STIMULUS_TO_TASKS: dict[str, list[str]] = {
    "three_session_A": ["visual_stimulation"],
    "three_session_B": ["visual_stimulation"],
    "three_session_C": ["visual_stimulation"],
    "three_session_C2": ["visual_stimulation"],
    "OPHYS_1_images_A": ["change_detection", "visual_stimulation"],
    "OPHYS_2_images_A_passive": ["passive_viewing", "visual_stimulation"],
    "OPHYS_2_images_B_passive": ["passive_viewing", "visual_stimulation"],
    "OPHYS_3_images_A": ["change_detection", "visual_stimulation"],
    "OPHYS_3_images_B": ["change_detection", "visual_stimulation"],
    "OPHYS_4_images_A": ["change_detection", "visual_stimulation"],
    "OPHYS_4_images_B": ["change_detection", "visual_stimulation"],
    "OPHYS_5_images_A_passive": ["passive_viewing", "visual_stimulation"],
    "OPHYS_5_images_B_passive": ["passive_viewing", "visual_stimulation"],
    "OPHYS_6_images_A": ["change_detection", "visual_stimulation"],
    "OPHYS_6_images_B": ["change_detection", "visual_stimulation"],
    "spontaneous": ["spontaneous_activity"],
}

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

    return {
        "source": "allen",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": dataset.get("url", "https://portal.brain-map.org/"),
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


def _allen_paginate(
    model: str,
    include: str | None = None,
    extra_params: dict[str, Any] | None = None,
    limit: int | None = None,
    page_size: int = 500,
) -> list[dict[str, Any]]:
    """Generic Allen API paginator for any model endpoint."""
    records: list[dict[str, Any]] = []
    start_row = 0
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        while True:
            params: dict[str, Any] = {"num_rows": page_size, "start_row": start_row}
            if include:
                params["include"] = include
            if extra_params:
                params.update(extra_params)
            try:
                resp = client.get(f"{ALLEN_API}/{model}/query.json", params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.warning("Allen API %s error at start_row=%d: %s", model, start_row, exc)
                break
            if not data.get("success"):
                break
            batch = [r for r in (data.get("msg") or []) if isinstance(r, dict)]
            records.extend(batch)
            if limit is not None and len(records) >= limit:
                return records[:limit]
            total = data.get("total_rows", 0)
            start_row += len(batch)
            if start_row >= total or not batch:
                break
    return records if limit is None else records[:limit]


def normalize_ophys_experiment(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one OphysExperiment row to a flat corpus record."""
    exp_id = row["id"]
    depth = row.get("imaging_depth")
    stim = row.get("stimulus_name") or ""
    structure = row.get("targeted_structure") or {}
    structure_acronym = structure.get("acronym", "")
    structure_name = structure.get("name") or structure_acronym
    specimen = row.get("specimen") or {}
    specimen_name = specimen.get("name") or ""
    donor = specimen.get("donor") or {}
    genotype = donor.get("full_genotype") or ""

    cre_line = specimen_name.split(";")[0].strip() if specimen_name else ""
    depth_str = f" at {depth}μm depth" if depth else ""
    is_visual_behavior = stim.startswith("OPHYS_")

    title = (
        f"Allen Brain Observatory: {cre_line} 2P calcium imaging"
        f" — {stim} in {structure_name}{depth_str}"
    )
    description = (
        f"Allen Brain Observatory two-photon calcium imaging session. "
        f"Mouse line: {cre_line}."
        + (f" Genotype: {genotype}." if genotype else "")
        + f" Brain region: {structure_name} ({structure_acronym})."
        + (f" Imaging depth: {depth}μm." if depth else "")
        + f" Stimulus protocol: {stim}."
        + " Data available in NWB format via AllenSDK."
    )
    brain_regions = []
    if structure_acronym:
        mapped = _STRUCTURE_TO_REGION.get(structure_acronym)
        brain_regions = [mapped or structure_acronym.lower()]

    portal_url = (
        "https://portal.brain-map.org/explore/circuits/visual-behavior-2p"
        if is_visual_behavior
        else "https://observatory.brain-map.org/visualcoding"
    )
    tasks = _STIMULUS_TO_TASKS.get(stim, ["visual_stimulation"])

    return {
        "source": "allen",
        "source_id": f"ophys_{exp_id}",
        "source_type": "canonical_dataset",
        "title": title,
        "description": description,
        "url": portal_url,
        "identifier": str(exp_id),
        "license": "Allen Institute Terms of Use (CC-BY-NC)",
        "species": ["mouse"],
        "modalities": ["two_photon_calcium_imaging", "calcium_imaging"],
        "brain_regions": brain_regions,
        "tasks": tasks,
        "behaviors": ["licking", "running"] if is_visual_behavior else [],
        "data_standards": ["NWB"],
        "has_behavior": is_visual_behavior,
        "has_trials": True,
        "has_raw_data": True,
        "has_processed_data": True,
        "analysis_affordances": [
            "calcium_event_detection",
            "population_coding",
            "stimulus_response",
            "cell_type_characterization",
        ],
        "metadata_json": {
            "raw_source": "allen_brain_observatory",
            "experiment_id": exp_id,
            "container_id": row.get("experiment_container_id"),
            "imaging_depth": depth,
            "stimulus_name": stim,
            "targeted_structure": structure_acronym,
            "cre_line": cre_line,
            "full_genotype": genotype,
            "data_type": "visual_behavior" if is_visual_behavior else "visual_coding",
        },
    }


def normalize_ecephys_session(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize one EcephysSession row to a flat corpus record."""
    sess_id = row["id"]
    stim = row.get("stimulus_name") or ""

    title = f"Allen Visual Coding Neuropixels: {stim} session {sess_id}"
    description = (
        f"Allen Brain Observatory Visual Coding Neuropixels extracellular electrophysiology session. "
        f"Stimulus: {stim}. "
        f"Simultaneous Neuropixels probes across visual cortex, hippocampus, and thalamus. "
        f"Data available in NWB format via AllenSDK."
    )
    return {
        "source": "allen",
        "source_id": f"ecephys_{sess_id}",
        "source_type": "canonical_dataset",
        "title": title,
        "description": description,
        "url": "https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels",
        "identifier": str(sess_id),
        "license": "Allen Institute Terms of Use (CC-BY-NC)",
        "species": ["mouse"],
        "modalities": ["extracellular_electrophysiology", "neuropixels"],
        "brain_regions": ["visual_cortex", "hippocampus", "thalamus"],
        "tasks": _STIMULUS_TO_TASKS.get(stim, ["visual_stimulation"]),
        "behaviors": [],
        "data_standards": ["NWB"],
        "has_behavior": False,
        "has_trials": True,
        "has_raw_data": True,
        "has_processed_data": True,
        "analysis_affordances": [
            "spike_sorting",
            "population_coding",
            "stimulus_response",
            "multi_area_coordination",
        ],
        "metadata_json": {
            "raw_source": "allen_visual_coding_neuropixels",
            "session_id": sess_id,
            "stimulus_name": stim,
        },
    }


def fetch_brain_observatory_records(limit: int = 400) -> list[dict[str, Any]]:
    """Fetch Allen Brain Observatory OphysExperiment and EcephysSession records."""
    records: list[dict[str, Any]] = []

    # Two-photon calcium imaging: fetch up to (limit - 58) ophys experiments
    ophys_limit = max(0, limit - 58)
    if ophys_limit > 0:
        ophys_rows = _allen_paginate(
            "OphysExperiment",
            include="targeted_structure,specimen(donor(transgenic_lines))",
            limit=ophys_limit,
        )
        logger.info("Allen OphysExperiment: fetched %d rows", len(ophys_rows))
        for row in ophys_rows:
            try:
                records.append(normalize_ophys_experiment(row))
            except Exception as exc:
                logger.warning("Allen ophys normalize error for id=%s: %s", row.get("id"), exc)

    # Neuropixels: fetch all 58 sessions (small set, always include)
    ecephys_rows = _allen_paginate("EcephysSession", limit=100)
    logger.info("Allen EcephysSession: fetched %d rows", len(ecephys_rows))
    for row in ecephys_rows:
        try:
            records.append(normalize_ecephys_session(row))
        except Exception as exc:
            logger.warning("Allen ecephys normalize error for id=%s: %s", row.get("id"), exc)

    return records


@register("allen")
def fetch_allen_records(limit: int = 500) -> list[dict[str, Any]]:
    """Registry adapter: curated Allen projects + Brain Observatory live sessions."""
    curated = records_from_curated()
    live = fetch_brain_observatory_records(limit=limit - len(curated))
    all_records = curated + live
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for rec in all_records:
        sid = str(rec.get("source_id", ""))
        if sid and sid not in seen:
            seen.add(sid)
            deduped.append(rec)
    logger.info("Allen: %d total records (%d curated + %d live)", len(deduped), len(curated), len(live))
    return deduped[:limit]


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
