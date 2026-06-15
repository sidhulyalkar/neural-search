"""DANDI live ingestion connector with safe dry-run mode."""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import httpx

from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.demo_seed import DEFAULT_DATABASE_URL
from neural_search.ingestion.doi_utils import (
    dois_to_paper_ids,
    extract_dois_from_dandi_metadata,
)
from neural_search.ingestion.live import (
    print_cli_error,
    print_normalized_records,
    save_dataset_records,
    save_raw_response,
)
from neural_search.ingestion.registry import register
from neural_search.normalized import (
    evidence_label_from_extraction,
    stable_normalized_id,
)
from neural_search.schemas import NormalizedDatasetRecord, UsabilityFlags

logger = logging.getLogger(__name__)

DANDI_API_URL = "https://api.dandiarchive.org/api"

# DANDI assetsSummary.variableMeasured → canonical modality IDs
_DANDI_VARIABLE_MEASURED_MAP: dict[str, str] = {
    # Extracellular ephys
    "electricalseries": "extracellular_ephys",
    "units": "extracellular_ephys",
    "spikeeventeries": "extracellular_ephys",
    "electrodes": "extracellular_ephys",
    # LFP
    "lfp": "lfp",
    "decompositionseries": "lfp",
    # Intracellular
    "patchchlamperies": "patch_clamp",
    "currenclampseries": "patch_clamp",
    "voltageclampseries": "patch_clamp",
    "intracellularelectrodeseries": "patch_clamp",
    # Calcium imaging / two-photon
    "twophotonseries": "calcium_imaging",
    "roiresponseseries": "calcium_imaging",
    "fluorescenceseries": "calcium_imaging",
    "imagingplane": "calcium_imaging",
    # One-photon / widefield
    "onephotonseries": "calcium_imaging",
    # EEG / iEEG
    "eegseries": "eeg",
    "ecog": "ecog",
    "ieeg": "ieeg",
    # Behavioral
    "timeseries": "behavioral",
    "spatialseries": "behavioral",
    "processedseries": "behavioral",
    "behavioraleras": "behavioral",
    "behavioraltimeseries": "behavioral",
    "eyetracking": "eye_tracking",
    "pupiltracking": "eye_tracking",
    "position": "behavioral",
    "events": "behavioral",
    # fMRI / BOLD
    "bold": "fmri",
}

# DANDI approach names → canonical modality IDs
_DANDI_APPROACH_MAP: dict[str, str] = {
    "electrophysiological approach": "extracellular_ephys",
    "imaging approach": "calcium_imaging",
    "behavioral approach": "behavioral",
    "calcium imaging": "calcium_imaging",
    "two-photon microscopy": "calcium_imaging",
    "two photon microscopy": "calcium_imaging",
    "patch clamp": "patch_clamp",
    "whole cell patch clamp": "patch_clamp",
    "extracellular electrophysiology": "extracellular_ephys",
    "intracellular electrophysiology": "patch_clamp",
    "multi-unit recording": "extracellular_ephys",
    "single-unit recording": "extracellular_ephys",
    "eeg": "eeg",
    "ecog": "ecog",
    "functional magnetic resonance imaging": "fmri",
    "fmri": "fmri",
    "optogenetics": "optogenetics",
}

# DANDI species names → canonical IDs
_DANDI_SPECIES_MAP: dict[str, str] = {
    "house mouse": "mouse",
    "mus musculus": "mouse",
    "mouse": "mouse",
    "norway rat": "rat",
    "rattus norvegicus": "rat",
    "rat": "rat",
    "homo sapiens": "human",
    "human": "human",
    "rhesus macaque": "macaque",
    "macaca mulatta": "macaque",
    "macaque": "macaque",
    "drosophila melanogaster": "drosophila",
    "fruit fly": "drosophila",
    "danio rerio": "zebrafish",
    "zebrafish": "zebrafish",
    "caenorhabditis elegans": "c_elegans",
}


# DANDI about[].name UBERON anatomy strings → canonical brain region IDs
_DANDI_ABOUT_REGION_MAP: dict[str, str] = {
    # Broad neocortex
    "cortex": "neocortex",
    "neocortical": "neocortex",
    # Motor cortex
    "premotor cortex": "premotor_cortex",
    "primary motor cortex": "motor_cortex",
    "motor cortex": "motor_cortex",
    # Prefrontal / frontal
    "temporal lobe": "temporal_cortex",
    "frontal lobe": "prefrontal_cortex",
    "frontal cortex": "prefrontal_cortex",
    "prefrontal cortex": "prefrontal_cortex",
    "medial prefrontal cortex": "mPFC",
    "orbitofrontal cortex": "OFC",
    "anterior cingulate cortex": "ACC",
    "cingulate cortex": "ACC",
    "posterior cingulate cortex": "posterior_cingulate_cortex",
    # Visual
    "visual cortex": "visual_cortex",
    "primary visual cortex": "v1",
    "anterolateral visual area": "visual_cortex",
    "anteromedial visual area": "visual_cortex",
    "posteromedial visual area": "visual_cortex",
    # Somatosensory
    "barrel cortex": "somatosensory_cortex",
    "somatosensory cortex": "somatosensory_cortex",
    "primary somatosensory cortex": "somatosensory_cortex",
    "vibrissa unit": "somatosensory_cortex",
    # Parietal
    "posterior parietal cortex": "posterior_parietal_cortex",
    "parietal cortex": "parietal_cortex",
    # Temporal / auditory
    "temporal cortex": "temporal_cortex",
    "auditory cortex": "auditory_cortex",
    "primary auditory cortex": "auditory_cortex",
    # Retrosplenial
    "retrosplenial cortex": "retrosplenial_cortex",
    # Insula
    "insula": "insula",
    "insular cortex": "insula",
    # Broad neocortex
    "neocortex": "neocortex",
    "cerebral cortex": "neocortex",
    # Hippocampal formation
    "hippocampus": "hippocampus",
    "hippocampal formation": "hippocampus",
    "ammon's horn": "hippocampus",
    "cornu ammonis": "hippocampus",
    "ca1 field of hippocampus": "ca1",
    "ca1 field": "ca1",
    "ca2 field of hippocampus": "ca2",
    "ca3 field of hippocampus": "ca3",
    "ca3 field": "ca3",
    "dentate gyrus": "dentate_gyrus",
    "subiculum": "subiculum",
    # Entorhinal
    "medial entorhinal cortex": "medial_entorhinal_cortex",
    "lateral entorhinal cortex": "lateral_entorhinal_cortex",
    "entorhinal cortex": "entorhinal_cortex",
    # Amygdala
    "amygdala": "amygdala",
    "basolateral amygdala": "basolateral_amygdala",
    "basolateral amygdaloid complex": "basolateral_amygdala",
    "central amygdala": "central_amygdala",
    "central amygdaloid nucleus": "central_amygdala",
    # Thalamus
    "thalamus": "thalamus",
    "dorsal plus ventral thalamus": "thalamus",
    "lateral geniculate nucleus": "lateral_geniculate_nucleus",
    "medial geniculate nucleus": "medial_geniculate_nucleus",
    "mediodorsal nucleus": "mediodorsal_thalamus",
    "mediodorsal thalamus": "mediodorsal_thalamus",
    "ventral posterolateral nucleus": "thalamus",
    "pulvinar": "pulvinar",
    # Striatum / basal ganglia
    "striatum": "striatum",
    "caudate nucleus": "caudate",
    "caudate putamen": "caudate",
    "putamen": "putamen",
    "nucleus accumbens": "nucleus_accumbens",
    "globus pallidus": "globus_pallidus",
    "subthalamic nucleus": "subthalamic_nucleus",
    # Substantia nigra / VTA
    "substantia nigra": "substantia_nigra",
    "substantia nigra pars compacta": "substantia_nigra",
    "substantia nigra pars reticulata": "substantia_nigra",
    "ventral tegmental area": "vta",
    # Midbrain
    "superior colliculus": "superior_colliculus",
    "inferior colliculus": "inferior_colliculus",
    "periaqueductal gray": "periaqueductal_gray",
    # Brainstem
    "brainstem": "brainstem",
    "pons": "brainstem",
    "medulla oblongata": "brainstem",
    # LC
    "locus coeruleus": "locus_coeruleus",
    # Cerebellum
    "cerebellum": "cerebellum",
    # Hypothalamus
    "hypothalamus": "hypothalamus",
    # Septum
    "lateral septum": "septum",
    "medial septum": "septum",
    "lateral septum complex": "septum",
    "septal nuclei": "septum",
    # Olfactory
    "olfactory bulb": "olfactory_bulb",
    "piriform cortex": "piriform_cortex",
    # Language
    "broca's area": "broca_area",
    "broca area": "broca_area",
    # Retina
    "retina": "retina",
    # Spinal cord
    "spinal cord": "spinal_cord",
    "parietal lobe": "parietal_cortex",
    "occipital lobe": "visual_cortex",
}

# DANDI keywords → canonical brain region IDs (plain string match)
_DANDI_KEYWORD_REGION_MAP: dict[str, str] = {
    "hippocampus": "hippocampus",
    "hippocampal": "hippocampus",
    "neocortical": "neocortex",
    "cortex": "neocortex",
    "ca1": "ca1",
    "ca3": "ca3",
    "dentate gyrus": "dentate_gyrus",
    "entorhinal cortex": "entorhinal_cortex",
    "medial entorhinal cortex": "medial_entorhinal_cortex",
    "thalamus": "thalamus",
    "cerebellum": "cerebellum",
    "neocortex": "neocortex",
    "motor cortex": "motor_cortex",
    "primary motor cortex": "motor_cortex",
    "premotor cortex": "premotor_cortex",
    "barrel cortex": "somatosensory_cortex",
    "somatosensory cortex": "somatosensory_cortex",
    "visual cortex": "visual_cortex",
    "prefrontal cortex": "prefrontal_cortex",
    "medial prefrontal cortex": "mPFC",
    "anterior cingulate cortex": "ACC",
    "striatum": "striatum",
    "nucleus accumbens": "nucleus_accumbens",
    "amygdala": "amygdala",
    "substantia nigra": "substantia_nigra",
    "vta": "vta",
    "ventral tegmental area": "vta",
    "superior colliculus": "superior_colliculus",
    "brainstem": "brainstem",
    "olfactory bulb": "olfactory_bulb",
    "piriform cortex": "piriform_cortex",
    "insula": "insula",
    "retrosplenial cortex": "retrosplenial_cortex",
    "auditory cortex": "auditory_cortex",
    "posterior parietal cortex": "posterior_parietal_cortex",
    "parietal cortex": "parietal_cortex",
    "temporal cortex": "temporal_cortex",
    "hypothalamus": "hypothalamus",
    "septum": "septum",
    "lateral septum": "septum",
    "locus coeruleus": "locus_coeruleus",
    "subiculum": "subiculum",
    "v1": "v1",
}


def _map_dandi_about_to_regions(about_list: list[dict]) -> list[str]:
    """Map DANDI about[] UBERON anatomy names to canonical brain region IDs."""
    canonical: set[str] = set()
    for item in about_list:
        name = item.get("name", "").lower().strip() if isinstance(item, dict) else ""
        if name in _DANDI_ABOUT_REGION_MAP:
            canonical.add(_DANDI_ABOUT_REGION_MAP[name])
    return sorted(canonical)


def _map_dandi_keywords_to_regions(keywords: list[str]) -> list[str]:
    """Map DANDI keywords list to canonical brain region IDs.

    Handles both simple keywords and comma/semicolon-separated keyword strings
    (DANDI sometimes stores comma-lists as a single keyword entry).
    """
    import re as _re
    canonical: set[str] = set()
    for kw in keywords:
        # Some DANDI records store comma-separated lists as one keyword entry
        parts = _re.split(r"[,;]+", str(kw))
        for part in parts:
            key = part.lower().strip()
            if key in _DANDI_KEYWORD_REGION_MAP:
                canonical.add(_DANDI_KEYWORD_REGION_MAP[key])
    return sorted(canonical)


def _map_dandi_variable_measured(variable_measured: list[str]) -> list[str]:
    """Map DANDI variableMeasured NWB type names to canonical modality IDs."""
    canonical: set[str] = set()
    for vm in variable_measured:
        key = vm.lower().strip()
        if key in _DANDI_VARIABLE_MEASURED_MAP:
            canonical.add(_DANDI_VARIABLE_MEASURED_MAP[key])
    return sorted(canonical)


def _map_dandi_approach(approach_list: list[dict[str, str]]) -> list[str]:
    """Map DANDI approach objects to canonical modality IDs."""
    canonical: set[str] = set()
    for item in approach_list:
        name = item.get("name", "").lower().strip()
        if name in _DANDI_APPROACH_MAP:
            canonical.add(_DANDI_APPROACH_MAP[name])
    return sorted(canonical)


def _map_dandi_species(species_list: list[dict[str, str]]) -> list[str]:
    """Map DANDI species objects to canonical species IDs."""
    canonical: list[str] = []
    seen: set[str] = set()
    for item in species_list:
        name = item.get("name", "").lower().strip()
        if name in _DANDI_SPECIES_MAP:
            cid = _DANDI_SPECIES_MAP[name]
            if cid not in seen:
                seen.add(cid)
                canonical.append(cid)
    return canonical


def fetch_dandiset_rich_metadata(source_id: str) -> dict[str, Any]:
    """Fetch rich assetsSummary via the DANDI Python client.

    Returns a dict with keys: variableMeasured, approach, measurementTechnique,
    species, about, description, keywords, studyTarget. Returns empty dict on failure.
    """
    try:
        from dandi.dandiapi import DandiAPIClient
        with DandiAPIClient() as client:
            ds = client.get_dandiset(source_id, "draft")
            meta = ds.get_metadata()
            d = meta.model_dump(mode="json", exclude_none=True)
            assets = d.get("assetsSummary") or {}
            study_targets = d.get("studyTarget") or []
            # studyTarget can be list of strings or list of dicts with 'name'
            study_target_strs = [
                (t.get("name") if isinstance(t, dict) else str(t))
                for t in study_targets if t
            ]
            return {
                "variableMeasured": assets.get("variableMeasured") or [],
                "approach": assets.get("approach") or [],
                "measurementTechnique": assets.get("measurementTechnique") or [],
                "species": assets.get("species") or [],
                "about": d.get("about") or [],
                "description": d.get("description") or "",
                "keywords": d.get("keywords") or [],
                "studyTarget": study_target_strs,
            }
    except Exception as exc:
        logger.debug("Could not fetch rich metadata for DANDI %s: %s", source_id, exc)
        return {}


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _asset_summary(dandiset: dict[str, Any]) -> dict[str, Any]:
    assets = dandiset.get("assetsSummary") or dandiset.get("assets_summary") or {}
    if isinstance(assets, dict):
        return assets
    return {}


def normalize_dandiset(
    raw: dict[str, Any],
    *,
    rich_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize a raw DANDI listing record.

    ``rich_metadata`` can be pre-fetched via ``fetch_dandiset_rich_metadata``
    to supply assetsSummary variableMeasured / approach / species that the
    listing API omits.
    """
    version = raw.get("most_recent_published_version") or raw.get("draft_version") or raw
    metadata = version.get("metadata") or raw.get("metadata") or {}
    assets = _asset_summary(raw) or _asset_summary(version)
    source_id = str(raw.get("identifier") or raw.get("id") or metadata.get("identifier"))
    title = (
        metadata.get("name")
        or version.get("name")
        or raw.get("name")
        or f"DANDI {source_id}"
    )
    rich = rich_metadata or {}
    description = (
        metadata.get("description")
        or version.get("description")
        or raw.get("description")
        or rich.get("description")
    )
    text = " ".join(str(part) for part in [title, description, metadata, assets])
    study_targets = rich.get("studyTarget") or []
    extraction = extract_dataset_labels(
        title=title,
        description=description,
        file_paths=[],
        source_metadata={**metadata, "assets": assets},
        linked_paper_abstracts=study_targets,
    )

    # Supplement with DANDI-client-derived modalities, species, and brain regions
    rich_modalities: list[str] = (
        _map_dandi_variable_measured(rich.get("variableMeasured") or []) +
        _map_dandi_approach(rich.get("approach") or [])
    )
    rich_species = _map_dandi_species(rich.get("species") or [])
    rich_brain_regions: list[str] = (
        _map_dandi_about_to_regions(rich.get("about") or []) +
        _map_dandi_keywords_to_regions(rich.get("keywords") or [])
    )

    return {
        "source": "dandi",
        "source_id": source_id,
        "title": title,
        "description": description,
        "url": raw.get("url") or f"https://dandiarchive.org/dandiset/{source_id}",
        "license": metadata.get("license") or raw.get("license"),
        "species": list(dict.fromkeys(rich_species + [item.id for item in extraction.species])),
        "modalities": sorted({*rich_modalities, *(item.id for item in extraction.modalities)}),
        "brain_regions": list(dict.fromkeys(rich_brain_regions + [item.id for item in extraction.brain_regions])),
        "tasks": [item.id for item in extraction.tasks],
        "behaviors": [item.id for item in extraction.behaviors],
        "data_standards": sorted({item.id for item in extraction.data_standards} | {"NWB"}),
        "has_behavior": bool(extraction.behaviors) or "behavior" in text.casefold(),
        "has_trials": any(term in text.casefold() for term in ["trial", "trials", "event"]),
        "has_raw_data": True,
        "has_processed_data": any(term in text.casefold() for term in ["processed", "derived"]),
        "metadata_json": {
            "raw_source": "dandi",
            "version": version.get("version"),
            "asset_summary": assets,
            "measurement_technique": _as_list(metadata.get("measurementTechnique")),
        },
    }


def normalize_dandiset_record(
    raw: dict[str, Any],
    raw_payload_path: str | None = None,
) -> NormalizedDatasetRecord:
    """Normalize a raw DANDI payload into the v0.3 provenance-aware schema."""

    legacy = normalize_dandiset(raw)
    # Extract DANDI metadata dict (same path as normalize_dandiset uses)
    _version = raw.get("most_recent_published_version") or raw.get("draft_version") or raw
    _dandi_metadata: dict[str, Any] = _version.get("metadata") or raw.get("metadata") or {}
    _extracted_dois = extract_dois_from_dandi_metadata(_dandi_metadata)
    metadata = legacy.get("metadata_json", {})
    extraction = extract_dataset_labels(
        title=legacy.get("title"),
        description=legacy.get("description"),
        file_paths=[],
        source_metadata=metadata,
        linked_paper_abstracts=[],
    )
    source_value = " ".join(
        str(part) for part in [legacy.get("title"), legacy.get("description")] if part
    )
    return NormalizedDatasetRecord(
        dataset_id=stable_normalized_id("dataset", "dandi", legacy["source_id"]),
        source="dandi",
        source_id=legacy["source_id"],
        title=legacy["title"],
        description=legacy.get("description"),
        url=legacy.get("url"),
        raw_payload_path=raw_payload_path,
        species=[
            evidence_label_from_extraction(
                label, "species", source_field="metadata", source_value=source_value
            )
            for label in extraction.species
        ],
        modalities=[
            evidence_label_from_extraction(
                label, "modality", source_field="metadata", source_value=source_value
            )
            for label in extraction.modalities
        ],
        brain_regions=[
            evidence_label_from_extraction(
                label, "brain_region", source_field="metadata", source_value=source_value
            )
            for label in extraction.brain_regions
        ],
        tasks=[
            evidence_label_from_extraction(
                label, "task", source_field="metadata", source_value=source_value
            )
            for label in extraction.tasks
        ],
        behavioral_events=[
            evidence_label_from_extraction(
                label, "behavioral_event", source_field="metadata", source_value=source_value
            )
            for label in extraction.behaviors
        ],
        data_standards=[
            evidence_label_from_extraction(
                label, "data_standard", source_field="metadata", source_value=source_value
            )
            for label in extraction.data_standards
        ],
        usability_flags=UsabilityFlags(
            has_trials=legacy.get("has_trials"),
            has_behavior=legacy.get("has_behavior"),
            has_neural_data=bool(legacy.get("modalities")),
            has_raw_data=legacy.get("has_raw_data"),
            has_processed_data=legacy.get("has_processed_data"),
            has_standard_format="NWB" in legacy.get("data_standards", []),
        ),
        linked_papers=dois_to_paper_ids(_extracted_dois),
        missing_fields=extraction.missing_fields,
    )


def fetch_all_dandisets(
    *,
    start_url: str | None = None,
    page_size: int = 100,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Page through all DANDI dandisets and return normalized records."""
    url: str | None = start_url or f"{DANDI_API_URL}/dandisets/?page=1&page_size={page_size}"
    all_records: list[dict[str, Any]] = []

    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        while url:
            try:
                resp = client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("DANDI page fetch failed: %s - %s", url, exc)
                logger.warning("Harvest terminated early; %d records collected before failure", len(all_records))
                break

            data = resp.json()
            for raw in data.get("results", []):
                all_records.append(normalize_dandiset(raw))
                if max_records is not None and len(all_records) >= max_records:
                    return all_records

            url = data.get("next")
            logger.info("DANDI harvest: %d records so far, next=%s", len(all_records), url)

    return all_records


def fetch_dandi(query: str, limit: int) -> dict[str, Any]:
    params = {"search": query, "page_size": limit}
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        response = client.get(f"{DANDI_API_URL}/dandisets/", params=params)
        response.raise_for_status()
        return response.json()


def records_from_response(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    results = payload.get("results", payload if isinstance(payload, list) else [])
    return [normalize_dandiset(item) for item in results[:limit]]


@register("dandi")
def fetch_dandi_records(limit: int = 1000) -> list[dict[str, Any]]:
    """Registry adapter for full DANDI pagination."""
    return fetch_all_dandisets(max_records=limit)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.ingestion.dandi")
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--database-url", default=DEFAULT_DATABASE_URL)
    args = parser.parse_args(argv)

    try:
        payload = fetch_dandi(args.query, args.limit)
        if args.save or args.save_raw:
            raw_path = save_raw_response("dandi", args.query, payload)
            print(json.dumps({"raw_saved": str(raw_path)}, indent=2))
        records = records_from_response(payload, args.limit)
        print_normalized_records(records)
        if args.dry_run or not args.save:
            return 0
        summary = save_dataset_records(records, args.database_url, args.force)
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print_cli_error("dandi", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
