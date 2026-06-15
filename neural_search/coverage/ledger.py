"""Coverage ledger and corpus gap reports.

The ledger is intentionally row-oriented: every dataset label becomes an
indexable coverage entry with provenance, confidence, access tier, and analysis
level context. Gap reports are then deterministic aggregations over those rows.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from neural_search.normalized import record_to_dict
from neural_search.species import canonical_species_id

DEFAULT_SNAPSHOT_ID = "current"
DEFAULT_MIN_CONFIDENCE = 0.65

COVERAGE_DIMENSIONS: tuple[str, ...] = (
    "species",
    "brain_regions",
    "modalities",
    "recording_scales",
    "tasks",
    "behavioral_events",
    "analysis_levels",
    "access_tiers",
)

VALUE_COVERAGE_STATES: tuple[str, ...] = (
    "observed",
    "derived_from_file",
    "derived_from_source_default",
    "inferred_silver",
)

COVERAGE_STATES: tuple[str, ...] = (
    *VALUE_COVERAGE_STATES,
    "not_applicable",
    "restricted_unavailable",
    "unknown_needs_review",
)

LABEL_FIELDS: tuple[str, ...] = (
    "species",
    "brain_regions",
    "modalities",
    "recording_scales",
    "tasks",
    "behavioral_events",
)

OPEN_ACCESS_SOURCES = {
    "allen",
    "brain_image_library",
    "bluebrain",
    "buzsaki",
    "crcns",
    "dandi",
    "figshare",
    "gin",
    "ibl",
    "nemo",
    "neuromorpho",
    "neurovault",
    "openneuro",
    "osf",
    "zenodo",
}

REGISTERED_ACCESS_SOURCES = {"ebrains", "human_connectome_project"}
CONTROLLED_ACCESS_SOURCES = {"dbgap", "hcp", "uk_biobank"}
RESTRICTED_ACCESS_SOURCES = {"pi_request", "lab_archive"}

MODALITY_COMPATIBILITY: dict[tuple[str, str], str] = {
    ("calcium_imaging", "extracellular_ephys"): "single_cell_comparable",
    ("calcium_imaging", "neuropixels"): "single_cell_comparable",
    ("ecog", "seeg"): "intracranial_field_comparable",
    ("eeg", "lfp"): "mesoscale_oscillation_comparable",
    ("eeg", "meg"): "sensor_space_comparable",
    ("extracellular_ephys", "neuropixels"): "single_unit_comparable",
    ("extracellular_ephys", "tetrode"): "single_unit_comparable",
    ("fmri", "calcium_widefield"): "mesoscale_spatial_comparable",
    ("fmri", "lfp"): "not_directly_comparable",
    ("neuropixels", "tetrode"): "single_unit_comparable",
}

MODALITY_TO_RECORDING_SCALE: dict[str, tuple[str, ...]] = {
    "bold": ("bold_voxel_timeseries",),
    "calcium_imaging": ("calcium_roi_fluorescence",),
    "calcium_widefield": ("widefield_fluorescence",),
    "connectivity": ("connectomic_edge",),
    "connectomics": ("connectomic_edge",),
    "diffusion_mri": ("structural_mri_voxel",),
    "ecog": ("ecog_surface_potential",),
    "eeg": ("eeg_sensor_timeseries",),
    "electron_microscopy": ("connectomic_edge",),
    "extracellular_electrophysiology": ("raw_extracellular_voltage",),
    "extracellular_ephys": ("raw_extracellular_voltage",),
    "fiber_photometry": ("fiber_photometry_trace",),
    "fmri": ("bold_voxel_timeseries",),
    "ieeg": ("ecog_surface_potential", "seeg_depth_potential"),
    "lfp": ("local_field_potential",),
    "meg": ("meg_sensor_timeseries",),
    "merfish": ("spatial_transcriptomic_spot",),
    "microscopy": ("connectomic_edge",),
    "miniscope": ("calcium_roi_fluorescence",),
    "mri": ("structural_mri_voxel",),
    "neuron_morphology": ("connectomic_edge",),
    "neuropixels": ("raw_extracellular_voltage",),
    "patch_clamp": ("intracellular_membrane_signal",),
    "polysomnography": ("eeg_sensor_timeseries",),
    "seeg": ("seeg_depth_potential",),
    "single_cell_rna": ("transcriptomic_cell_profile",),
    "single_cell_rnaseq": ("transcriptomic_cell_profile",),
    "single_nucleus_rnaseq": ("transcriptomic_cell_profile",),
    "spatial_transcriptomics": ("spatial_transcriptomic_spot",),
    "tetrode": ("raw_extracellular_voltage",),
    "tracing": ("connectomic_edge",),
    "two_photon": ("calcium_roi_fluorescence",),
    "two_photon_calcium_imaging": ("calcium_roi_fluorescence",),
    "utah_array": ("raw_extracellular_voltage",),
    "visium": ("spatial_transcriptomic_spot",),
    "widefield_imaging": ("widefield_fluorescence",),
}

RECORDING_SCALE_LABELS: dict[str, str] = {
    "bold_voxel_timeseries": "BOLD voxel time series",
    "calcium_roi_fluorescence": "Calcium ROI fluorescence",
    "connectomic_edge": "Connectomic edge",
    "ecog_surface_potential": "ECoG surface potential",
    "eeg_sensor_timeseries": "EEG sensor time series",
    "fiber_photometry_trace": "Fiber photometry trace",
    "intracellular_membrane_signal": "Intracellular membrane signal",
    "local_field_potential": "Local field potential",
    "meg_sensor_timeseries": "MEG sensor time series",
    "multi_unit_activity": "Multi-unit activity",
    "raw_extracellular_voltage": "Raw extracellular voltage",
    "seeg_depth_potential": "SEEG depth potential",
    "single_unit_spikes": "Single-unit spikes",
    "spatial_transcriptomic_spot": "Spatial transcriptomic spot",
    "structural_mri_voxel": "Structural MRI voxel",
    "transcriptomic_cell_profile": "Transcriptomic cell profile",
    "widefield_fluorescence": "Widefield fluorescence",
}

TEXT_REFINABLE_SCALE_MODALITIES = {
    "calcium_imaging",
    "ecog",
    "eeg",
    "extracellular_electrophysiology",
    "extracellular_ephys",
    "fiber_photometry",
    "ieeg",
    "lfp",
    "meg",
    "miniscope",
    "neuropixels",
    "patch_clamp",
    "polysomnography",
    "seeg",
    "tetrode",
    "two_photon",
    "two_photon_calcium_imaging",
    "utah_array",
    "widefield_imaging",
}

STATIC_BEHAVIOR_NA_SOURCES = {
    "brain_image_library",
    "cellxgene",
    "neuromorpho",
}

STATIC_BEHAVIOR_NA_MODALITIES = {
    "connectomics",
    "diffusion_mri",
    "electron_microscopy",
    "merfish",
    "microscopy",
    "mri",
    "neuron_morphology",
    "single_cell_rna",
    "single_cell_rnaseq",
    "single_nucleus_rnaseq",
    "spatial_transcriptomics",
    "tracing",
    "visium",
}

SOURCE_DEFAULT_STATE_REASONS: dict[str, dict[str, str]] = {
    "neuromorpho": {
        "modalities": "source default: NeuroMorpho records are neuronal morphology reconstructions",
        "recording_scales": "source default: NeuroMorpho samples reconstructed cellular morphology",
    },
    "neurovault": {
        "modalities": "source default: NeuroVault records are neuroimaging map collections",
        "recording_scales": "source default: NeuroVault records are usually voxelwise imaging derivatives",
        "behavioral_events": "source default: NeuroVault stores derived maps rather than event tables",
    },
    "openneuro": {
        "tasks": "source default: OpenNeuro/BIDS task labels can be derived from task-* files",
        "behavioral_events": "source default: OpenNeuro/BIDS events can be derived from events.tsv/HED files",
    },
    "dandi": {
        "behavioral_events": "source default: DANDI/NWB behavioral events can be derived from trials/epochs/stimulus tables",
    },
}

DIMENSION_COMPLETION_ACTIONS: dict[str, str] = {
    "species": "resolve species from source specimen metadata, participants.tsv, or linked publication organism terms",
    "brain_regions": "resolve regions from atlas IDs, electrode coordinates, specimen metadata, masks, or linked paper methods",
    "modalities": "resolve modality from file inventory, source technique metadata, or dataset methods text",
    "recording_scales": "resolve recording scale from NWB/BIDS/file contents or modality-specific sampling defaults",
    "tasks": "resolve task labels from task-* filenames, NWB trials, stimulus tables, protocols, or linked publications",
    "behavioral_events": "resolve behavioral events from BIDS events.tsv/HED, NWB trials/epochs/stimulus tables, or protocol text",
    "analysis_levels": "resolve analysis level from file products, standards, and available raw/processed flags",
    "access_tiers": "resolve access tier from source policy, license, terms of use, or dataset metadata",
}

SOURCE_DIMENSION_COMPLETION_ACTIONS: dict[str, dict[str, str]] = {
    "dandi": {
        "species": "query DANDI dandiset assets and NWB subject/specimen metadata",
        "brain_regions": "inspect NWB electrodes/optical physiology imaging_plane locations and DANDI metadata",
        "modalities": "inspect NWB acquisition/processing groups and dandiset measurementTechnique metadata",
        "recording_scales": "inspect NWB acquisition objects, units tables, electrodes, imaging planes, and processing modules",
        "tasks": "inspect NWB trials/epochs/intervals/stimulus metadata and dandiset protocol text",
        "behavioral_events": "extract NWB trials, epochs, TimeIntervals, stimulus presentations, and ndx-events/HED annotations",
    },
    "openneuro": {
        "species": "parse BIDS participants.tsv, samples.tsv, dataset_description.json, and linked publication metadata",
        "brain_regions": "parse BIDS derivatives, masks, coordsystem/electrodes files, and linked paper region terms",
        "modalities": "derive modality from BIDS suffixes, datatype folders, scans.tsv, and dataset_description.json",
        "recording_scales": "derive scale from BIDS datatype/suffix plus EEG/MEG/iEEG/fMRI sidecars",
        "tasks": "extract task labels from BIDS task-* entities and task events files",
        "behavioral_events": "extract events from *_events.tsv and HED sidecar annotations",
    },
    "neurovault": {
        "species": "default to human when collection/image metadata confirms human neuroimaging; otherwise inspect linked papers",
        "brain_regions": "reverse-map image masks/coordinates to atlas regions and mine collection cognitive atlas tags",
        "modalities": "derive imaging modality from NeuroVault image type, map type, and collection metadata",
        "recording_scales": "mark voxelwise imaging derivative after confirming image/map type",
        "tasks": "mine task/cognitive contrast labels from collection metadata and linked publications",
    },
    "neuromorpho": {
        "species": "resolve species from NeuroMorpho animal/species fields",
        "brain_regions": "resolve brain region from NeuroMorpho archive metadata and normalize to atlas IDs",
        "modalities": "apply NeuroMorpho neuronal morphology source default after record validation",
        "recording_scales": "apply cellular morphology/connectomic-edge scale after record validation",
    },
    "brain_image_library": {
        "species": "resolve species from specimen metadata",
        "brain_regions": "resolve regions from specimen, atlas, and image metadata",
        "modalities": "derive modality from instrument/imaging metadata",
        "recording_scales": "derive imaging scale from image metadata and acquisition modality",
    },
}

DIMENSION_PRIORITY: dict[str, int] = {
    "brain_regions": 100,
    "species": 95,
    "modalities": 90,
    "recording_scales": 88,
    "behavioral_events": 82,
    "tasks": 80,
    "access_tiers": 70,
    "analysis_levels": 60,
}

RECORDING_SCALE_TEXT_HINTS: dict[str, tuple[str, ...]] = {
    "bold_voxel_timeseries": (
        "activation map",
        "bold",
        "bold time series",
        "functional mri",
        "fmri",
    ),
    "calcium_roi_fluorescence": (
        "calcium imaging",
        "calcium roi",
        "delta f over f",
        "dff",
        "gcamp",
        "roi fluorescence",
        "two photon",
        "two-photon",
    ),
    "connectomic_edge": (
        "connectome",
        "morphology",
        "neuron reconstruction",
        "skeleton",
        "swc",
        "synapse graph",
        "tracing",
    ),
    "ecog_surface_potential": (
        "ecog",
        "electrocorticography",
        "high gamma",
        "subdural grid",
    ),
    "eeg_sensor_timeseries": (
        "eeg",
        "eeg channel",
        "eeg time series",
        "erp",
        "scalp eeg",
    ),
    "fiber_photometry_trace": (
        "fiber photometry",
        "photometry trace",
        "bulk fluorescence",
    ),
    "intracellular_membrane_signal": (
        "current clamp",
        "membrane potential",
        "patch clamp",
        "voltage clamp",
        "whole-cell",
    ),
    "local_field_potential": (
        "field potential",
        "gamma oscillation",
        "lfp",
        "local field potential",
        "spectral power",
        "theta oscillation",
    ),
    "meg_sensor_timeseries": (
        "gradiometer",
        "magnetometer",
        "meg",
        "meg time series",
    ),
    "multi_unit_activity": (
        "multi unit",
        "multi-unit",
        "multiunit",
        "mua",
        "threshold crossing",
    ),
    "raw_extracellular_voltage": (
        "ap band",
        "continuous ephys",
        "extracellular voltage",
        "raw traces",
        "raw voltage",
    ),
    "seeg_depth_potential": (
        "depth electrode",
        "intracranial eeg",
        "seeg",
        "stereo eeg",
    ),
    "single_unit_spikes": (
        "action potentials",
        "isolated neuron",
        "single unit",
        "single-unit",
        "sorted unit",
        "spike times",
        "spike train",
    ),
    "spatial_transcriptomic_spot": (
        "merfish",
        "spatial gene expression",
        "spatial transcriptomics",
        "visium",
    ),
    "structural_mri_voxel": (
        "diffusion mri",
        "dti",
        "structural mri",
        "t1w",
        "t2w",
    ),
    "transcriptomic_cell_profile": (
        "cell type atlas",
        "gene expression",
        "single cell rna",
        "single nucleus rna",
        "single-cell transcriptomics",
    ),
    "widefield_fluorescence": (
        "cortical widefield",
        "mesoscale imaging",
        "wide field",
        "widefield",
    ),
}


@dataclass(frozen=True)
class CoverageEntry:
    """One indexable coverage assertion for a dataset."""

    entry_id: str
    dataset_id: str
    source: str
    source_id: str
    dimension: str
    value_id: str
    label: str
    confidence: float
    evidence_tier: str
    access_tier: str
    analysis_level: str
    source_field: str | None = None
    evidence_text: str | None = None
    first_seen: str | None = None
    ingested_at: str | None = None
    snapshot_id: str = DEFAULT_SNAPSHOT_ID

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageStateEntry:
    """One dataset × dimension coverage-state assertion."""

    state_id: str
    dataset_id: str
    source: str
    source_id: str
    dimension: str
    coverage_state: str
    confidence: float
    access_tier: str
    reason: str | None = None
    value_ids: list[str] = field(default_factory=list)
    evidence_tier: str | None = None
    source_field: str | None = None
    evidence_text: str | None = None
    first_seen: str | None = None
    ingested_at: str | None = None
    snapshot_id: str = DEFAULT_SNAPSHOT_ID

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageCompletionItem:
    """Actionable work item for resolving an incomplete coverage state."""

    item_id: str
    dataset_id: str
    source: str
    source_id: str
    title: str
    dimension: str
    current_state: str
    priority: int
    recommended_action: str
    reason: str
    access_tier: str
    snapshot_id: str = DEFAULT_SNAPSHOT_ID

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModalityCompatibility:
    """Compatibility relationship between two sampling modalities."""

    modality_a: str
    modality_b: str
    compatibility_class: str
    confidence: float

    @property
    def comparable(self) -> bool:
        return self.compatibility_class != "not_directly_comparable"

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageGapReport:
    """Machine-readable summary of corpus coverage gaps."""

    generated_at: str
    snapshot_id: str
    dataset_count: int
    entry_count: int
    dimension_counts: dict[str, int]
    value_counts: dict[str, dict[str, int]]
    missing_dimension_counts: dict[str, int]
    low_confidence_counts: dict[str, int]
    access_tier_counts: dict[str, int]
    analysis_level_counts: dict[str, int]
    open_access_dimension_counts: dict[str, int]
    coverage_rates: dict[str, float]
    field_state_coverage_rates: dict[str, float]
    actionable_state_rates: dict[str, float]
    coverage_state_counts: dict[str, dict[str, int]]
    not_applicable_counts: dict[str, int]
    unknown_state_counts: dict[str, int]
    evidence_tier_counts: dict[str, int]
    inferred_dimension_counts: dict[str, int]
    source_state_coverage: list[dict[str, Any]] = field(default_factory=list)
    completion_worklist_summary: list[dict[str, Any]] = field(default_factory=list)
    source_coverage: list[dict[str, Any]] = field(default_factory=list)
    thin_species_region_pairs: list[dict[str, Any]] = field(default_factory=list)
    thin_species_modality_pairs: list[dict[str, Any]] = field(default_factory=list)
    thin_region_modality_pairs: list[dict[str, Any]] = field(default_factory=list)
    dark_species_region_pairs: list[dict[str, Any]] = field(default_factory=list)
    dark_species_modality_pairs: list[dict[str, Any]] = field(default_factory=list)
    dark_region_modality_pairs: list[dict[str, Any]] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _norm(value: Any) -> str:
    cleaned = str(value or "").casefold().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in cleaned.split("_") if part)


def _phrase_norm(value: Any) -> str:
    cleaned = str(value or "").casefold()
    cleaned = re.sub(r"[/_-]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9+]+", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _contains_phrase(normalized_text: str, phrase: str) -> bool:
    normalized_phrase = _phrase_norm(phrase)
    if not normalized_phrase:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized_text) is not None


def _record_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, BaseModel):
        return record_to_dict(record)
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError(f"unsupported record type: {type(record)!r}")


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, Mapping)]
    if isinstance(payload, Mapping):
        return [dict(payload)]
    return []


def load_dataset_mappings(path: str | Path) -> list[dict[str, Any]]:
    """Load dataset-like mappings from JSON, JSONL, or a directory."""

    root = Path(path)
    if not root.exists():
        return []
    if root.is_dir():
        records: list[dict[str, Any]] = []
        for child in sorted([*root.glob("*.jsonl"), *root.glob("*.json")]):
            records.extend(load_dataset_mappings(child))
        return records
    if root.suffix == ".jsonl":
        records = []
        with root.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if isinstance(item, Mapping) and _is_dataset_mapping(item):
                    records.append(dict(item))
        return records
    return [item for item in _read_json_records(root) if _is_dataset_mapping(item)]


def _is_dataset_mapping(record: Mapping[str, Any]) -> bool:
    if "dataset_id" in record:
        return True
    return bool(record.get("source") and record.get("source_id") and record.get("title"))


def _label_items(record: Mapping[str, Any], field_name: str) -> list[dict[str, Any]]:
    raw_values = record.get(field_name) or []
    if isinstance(raw_values, str):
        raw_values = [raw_values]
    items: list[dict[str, Any]] = []
    for raw in raw_values:
        if isinstance(raw, Mapping):
            label = str(raw.get("label") or raw.get("id") or raw.get("value") or "").strip()
            value_id = str(raw.get("id") or label).strip()
            confidence = float(raw.get("confidence", 0.8) or 0.8)
            source_field = raw.get("source_field")
            evidence_text = raw.get("evidence_text") or raw.get("evidence")
        else:
            label = str(raw).strip()
            value_id = label
            confidence = 0.75
            source_field = field_name
            evidence_text = label
        if not label:
            continue
        if field_name == "species":
            value_id = canonical_species_id(label) or _norm(value_id)
        else:
            value_id = _norm(value_id)
        items.append(
            {
                "value_id": value_id,
                "label": label,
                "confidence": max(0.0, min(confidence, 1.0)),
                "source_field": str(source_field) if source_field else None,
                "evidence_text": str(evidence_text) if evidence_text else None,
            }
        )
    return items


def _record_text(record: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in ("title", "description", "abstract"):
        value = record.get(key)
        if value:
            parts.append(str(value))
    metadata = record.get("metadata_json")
    if isinstance(metadata, Mapping):
        for key in ("measurementTechnique", "approach", "data_type", "modality"):
            value = metadata.get(key)
            if value:
                parts.append(str(value))
    return "\n".join(parts)


def _recording_scale_text_matches(text: str) -> list[dict[str, Any]]:
    normalized = _phrase_norm(text)
    if not normalized:
        return []
    matches: list[dict[str, Any]] = []
    for scale_id, hints in RECORDING_SCALE_TEXT_HINTS.items():
        for hint in hints:
            if _contains_phrase(normalized, hint):
                matches.append(
                    {
                        "value_id": scale_id,
                        "label": RECORDING_SCALE_LABELS.get(scale_id, scale_id),
                        "confidence": 0.78,
                        "source_field": "title_description",
                        "evidence_text": hint,
                        "inferred": True,
                    }
                )
                break
    return matches


def _record_flags(record: Mapping[str, Any]) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    nested = record.get("usability_flags")
    if isinstance(nested, Mapping):
        flags.update(nested)
    for key in (
        "has_behavior",
        "has_trials",
        "has_raw_data",
        "has_processed_data",
        "has_standard_format",
    ):
        if key in record:
            flags[key] = record[key]
    return flags


def _inferred_recording_scale_items(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    cache_key = "__coverage_recording_scale_items"
    if isinstance(record, dict) and cache_key in record:
        return list(record[cache_key])

    explicit = _label_items(record, "recording_scales")
    inferred: dict[str, dict[str, Any]] = {item["value_id"]: item for item in explicit}
    modalities = {_norm(item["value_id"]) for item in _label_items(record, "modalities")}

    for modality_id in modalities:
        for scale_id in MODALITY_TO_RECORDING_SCALE.get(modality_id, ()):
            inferred.setdefault(
                scale_id,
                {
                    "value_id": scale_id,
                    "label": RECORDING_SCALE_LABELS.get(scale_id, scale_id),
                    "confidence": 0.72,
                    "source_field": "modalities",
                    "evidence_text": f"recording scale inferred from modality `{modality_id}`",
                    "inferred": True,
                },
            )

    should_scan_text = not inferred or bool(modalities & TEXT_REFINABLE_SCALE_MODALITIES)
    text = _record_text(record)
    if text and should_scan_text:
        for match in _recording_scale_text_matches(text):
            inferred.setdefault(
                match["value_id"],
                match,
            )
    items = list(inferred.values())
    if isinstance(record, dict):
        record[cache_key] = items
    return items


def _items_for_dimension(record: Mapping[str, Any], field_name: str) -> list[dict[str, Any]]:
    if field_name == "recording_scales":
        return _inferred_recording_scale_items(record)
    return _label_items(record, field_name)


def infer_access_tier(record: Mapping[str, Any]) -> str:
    """Infer coarse data-reuse access tier from source and license metadata."""

    source = _norm(record.get("source"))
    license_text = _norm(record.get("license") or record.get("license_id"))
    metadata = record.get("metadata_json")
    if isinstance(metadata, Mapping):
        license_text = license_text or _norm(metadata.get("license"))
        access = _norm(metadata.get("access_tier") or metadata.get("access"))
        if access in {
            "open_access",
            "registered_access",
            "controlled_access",
            "restricted_access",
        }:
            return access
    if source in CONTROLLED_ACCESS_SOURCES or "controlled" in license_text:
        return "controlled_access"
    if source in RESTRICTED_ACCESS_SOURCES or "restricted" in license_text:
        return "restricted_access"
    if source in REGISTERED_ACCESS_SOURCES or "registered" in license_text:
        return "registered_access"
    if source in OPEN_ACCESS_SOURCES:
        return "open_access"
    if any(token in license_text for token in ("cc0", "cc_by", "public_domain", "open")):
        return "open_access"
    return "unknown_access"


def infer_analysis_levels(record: Mapping[str, Any]) -> list[str]:
    """Infer analysis-level strata represented by a dataset."""

    modalities = {_norm(item["value_id"]) for item in _label_items(record, "modalities")}
    scales = {_norm(item["value_id"]) for item in _items_for_dimension(record, "recording_scales")}
    standards = {_norm(item["value_id"]) for item in _label_items(record, "data_standards")}
    flags = _record_flags(record)
    behaviors = record.get("behaviors") or record.get("behavioral_events") or []
    tasks = record.get("tasks") or []
    levels: set[str] = set()

    if flags.get("has_raw_data") or "raw_extracellular_voltage" in scales:
        levels.add("raw_signal")
    if {"single_unit_spikes", "multi_unit_activity"} & scales:
        levels.add("unit_activity")
        levels.add("spike_sorted")
    if {"local_field_potential", "eeg_sensor_timeseries", "ecog_surface_potential"} & scales:
        levels.add("mesoscale_field_potential")
    if {
        "calcium_imaging",
        "calcium_roi_fluorescence",
        "fiber_photometry",
        "fiber_photometry_trace",
        "two_photon",
        "two_photon_calcium_imaging",
        "widefield_fluorescence",
    } & (modalities | scales):
        levels.add("population_dynamics")
    if {"fmri", "bold_voxel_timeseries", "structural_mri_voxel"} & (modalities | scales):
        levels.add("voxelwise_imaging")
    if {
        "single_cell_rnaseq",
        "spatial_transcriptomic_spot",
        "transcriptomic_cell_profile",
    } & (modalities | scales):
        levels.add("molecular_profile")
    if "connectomic_edge" in scales or "connectomics" in modalities:
        levels.add("circuit_structure")
    if flags.get("has_behavior") or behaviors or tasks:
        levels.add("behavior_correlation")
    if flags.get("has_processed_data"):
        levels.add("processed_derivative")
    if standards:
        levels.add("standardized_container")
    return sorted(levels) or ["unspecified_analysis_level"]


def infer_evidence_tier(record: Mapping[str, Any], label: Mapping[str, Any]) -> str:
    """Classify evidence strength without pretending silver labels are gold."""

    provenance = _norm(record.get("brain_regions_provenance"))
    evidence = _norm(label.get("evidence_text"))
    source_field = _norm(label.get("source_field"))
    if label.get("inferred"):
        return "inferred_metadata"
    if any(token in provenance for token in ("llm", "gemini", "silver")):
        return "silver_inferred"
    if any(token in evidence for token in ("llm", "gemini", "silver")):
        return "silver_inferred"
    if "human" in provenance and "gold" in provenance:
        return "human_gold"
    if any(token in source_field for token in ("electrode", "nwb", "bids", "metadata")):
        return "structured_metadata"
    if source_field:
        return "declared_metadata"
    return "unqualified_label"


def propagate_confidence(node_confidence: float, edge_confidence: float) -> float:
    """Propagate confidence across a graph edge with multiplicative decay."""

    return max(0.0, min(float(node_confidence) * float(edge_confidence), 1.0))


def modality_compatibility(modality_a: str, modality_b: str) -> ModalityCompatibility:
    """Return known cross-modality comparability class."""

    a = _norm(modality_a)
    b = _norm(modality_b)
    if a == b:
        return ModalityCompatibility(a, b, "same_modality", 1.0)
    key = (a, b)
    reverse_key = (b, a)
    sorted_key = tuple(sorted((a, b)))
    compatibility = (
        MODALITY_COMPATIBILITY.get(key)
        or MODALITY_COMPATIBILITY.get(reverse_key)
        or MODALITY_COMPATIBILITY.get(sorted_key)
        or "unknown_compatibility"
    )
    confidence = 0.9 if compatibility != "unknown_compatibility" else 0.25
    return ModalityCompatibility(a, b, compatibility, confidence)


def _entry_id(dataset_id: str, dimension: str, value_id: str, snapshot_id: str) -> str:
    return f"coverage:{snapshot_id}:{_norm(dataset_id)}:{dimension}:{_norm(value_id)}"


def _state_id(dataset_id: str, dimension: str, snapshot_id: str) -> str:
    return f"coverage_state:{snapshot_id}:{_norm(dataset_id)}:{dimension}"


def _record_id(record: Mapping[str, Any]) -> str:
    return str(record.get("dataset_id") or f"dataset:{record.get('source')}:{record.get('source_id')}")


def build_coverage_entries(
    records: Iterable[Any],
    *,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    ingested_at: str | None = None,
) -> list[CoverageEntry]:
    """Build coverage entries from normalized records or plain mappings."""

    timestamp = ingested_at or _now_iso()
    entries: list[CoverageEntry] = []
    for raw_record in records:
        record = _record_mapping(raw_record)
        if not _is_dataset_mapping(record):
            continue
        dataset_id = _record_id(record)
        source = str(record.get("source") or "unknown")
        source_id = str(record.get("source_id") or dataset_id)
        first_seen = str(record.get("created_at") or timestamp)
        access_tier = infer_access_tier(record)
        analysis_levels = infer_analysis_levels(record)
        primary_analysis_level = analysis_levels[0]

        for field_name in LABEL_FIELDS:
            for item in _items_for_dimension(record, field_name):
                evidence_tier = infer_evidence_tier(record, item)
                confidence = float(item["confidence"])
                if evidence_tier == "silver_inferred":
                    confidence = min(confidence, 0.7)
                entries.append(
                    CoverageEntry(
                        entry_id=_entry_id(
                            dataset_id,
                            field_name,
                            str(item["value_id"]),
                            snapshot_id,
                        ),
                        dataset_id=dataset_id,
                        source=source,
                        source_id=source_id,
                        dimension=field_name,
                        value_id=str(item["value_id"]),
                        label=str(item["label"]),
                        confidence=confidence,
                        evidence_tier=evidence_tier,
                        access_tier=access_tier,
                        analysis_level=primary_analysis_level,
                        source_field=item.get("source_field"),
                        evidence_text=item.get("evidence_text"),
                        first_seen=first_seen,
                        ingested_at=timestamp,
                        snapshot_id=snapshot_id,
                    )
                )

        for analysis_level in analysis_levels:
            entries.append(
                CoverageEntry(
                    entry_id=_entry_id(dataset_id, "analysis_levels", analysis_level, snapshot_id),
                    dataset_id=dataset_id,
                    source=source,
                    source_id=source_id,
                    dimension="analysis_levels",
                    value_id=analysis_level,
                    label=analysis_level,
                    confidence=0.8,
                    evidence_tier="inferred_metadata",
                    access_tier=access_tier,
                    analysis_level=analysis_level,
                    first_seen=first_seen,
                    ingested_at=timestamp,
                    snapshot_id=snapshot_id,
                )
            )
        entries.append(
            CoverageEntry(
                entry_id=_entry_id(dataset_id, "access_tiers", access_tier, snapshot_id),
                dataset_id=dataset_id,
                source=source,
                source_id=source_id,
                dimension="access_tiers",
                value_id=access_tier,
                label=access_tier,
                confidence=0.75 if access_tier != "unknown_access" else 0.35,
                evidence_tier="inferred_metadata",
                access_tier=access_tier,
                analysis_level=primary_analysis_level,
                first_seen=first_seen,
                ingested_at=timestamp,
                snapshot_id=snapshot_id,
            )
        )
    return entries


def _item_coverage_state(record: Mapping[str, Any], item: Mapping[str, Any]) -> str:
    evidence_tier = infer_evidence_tier(record, item)
    source_field = _norm(item.get("source_field"))
    evidence = _norm(item.get("evidence_text"))
    if evidence_tier == "structured_metadata" or any(
        token in source_field
        for token in ("bids", "nwb", "electrode", "events", "file", "hed")
    ):
        return "derived_from_file"
    if evidence_tier == "silver_inferred":
        return "inferred_silver"
    if evidence_tier == "inferred_metadata":
        if source_field == "modalities" or "source_default" in evidence:
            return "derived_from_source_default"
        return "inferred_silver"
    return "observed"


def _best_item_state(record: Mapping[str, Any], items: list[dict[str, Any]]) -> tuple[str, str]:
    priority = {
        "observed": 4,
        "derived_from_file": 3,
        "derived_from_source_default": 2,
        "inferred_silver": 1,
    }
    states = [_item_coverage_state(record, item) for item in items]
    state = max(states, key=lambda candidate: priority[candidate])
    if state == "observed":
        reason = "coverage value is present in normalized dataset metadata"
    elif state == "derived_from_file":
        reason = "coverage value was derived from structured file/container metadata"
    elif state == "derived_from_source_default":
        reason = "coverage value was inferred from source or modality defaults"
    else:
        reason = "coverage value is inferred and needs curator review before gold use"
    return state, reason


def _not_applicable_reason(record: Mapping[str, Any], dimension: str) -> str | None:
    if dimension not in {"tasks", "behavioral_events"}:
        return None
    flags = _record_flags(record)
    if flags.get("has_behavior") is False:
        return "record declares no behavioral component"
    source = _norm(record.get("source"))
    if dimension == "behavioral_events" and source == "neurovault":
        return "derived neuroimaging map archive without event-table sampling"
    if source in STATIC_BEHAVIOR_NA_SOURCES:
        return "static anatomical/molecular archive without behavioral sampling"
    modalities = {_norm(item["value_id"]) for item in _label_items(record, "modalities")}
    if modalities and modalities <= STATIC_BEHAVIOR_NA_MODALITIES:
        return "static anatomical/molecular modalities without behavioral sampling"
    return None


def _missing_dimension_state(
    record: Mapping[str, Any],
    dimension: str,
    access_tier: str,
) -> tuple[str, float, str, str | None]:
    not_applicable = _not_applicable_reason(record, dimension)
    if not_applicable:
        return "not_applicable", 0.9, not_applicable, None
    if access_tier in {"controlled_access", "restricted_access"}:
        return (
            "restricted_unavailable",
            0.8,
            "metadata may be unavailable because the dataset is not openly reusable",
            None,
        )
    source = _norm(record.get("source"))
    source_reason = SOURCE_DEFAULT_STATE_REASONS.get(source, {}).get(dimension)
    if source_reason:
        return "derived_from_source_default", 0.55, source_reason, "source"
    return "unknown_needs_review", 0.35, "no coverage value or applicability rule found", None


def _state_for_dimension(
    record: Mapping[str, Any],
    dimension: str,
    access_tier: str,
) -> tuple[str, float, str, list[str], str | None, str | None, str | None]:
    if dimension in LABEL_FIELDS:
        items = _items_for_dimension(record, dimension)
        if items:
            state, reason = _best_item_state(record, items)
            value_ids = sorted({str(item["value_id"]) for item in items})
            evidence_tiers = {infer_evidence_tier(record, item) for item in items}
            evidence_tier = sorted(evidence_tiers)[0] if len(evidence_tiers) == 1 else "mixed"
            confidence = max(float(item["confidence"]) for item in items)
            source_fields = sorted(
                {str(item["source_field"]) for item in items if item.get("source_field")}
            )
            evidence_texts = sorted(
                {str(item["evidence_text"]) for item in items if item.get("evidence_text")}
            )
            return (
                state,
                confidence,
                reason,
                value_ids,
                evidence_tier,
                ", ".join(source_fields) if source_fields else None,
                "; ".join(evidence_texts[:3]) if evidence_texts else None,
            )
        state, confidence, reason, source_field = _missing_dimension_state(
            record,
            dimension,
            access_tier,
        )
        return state, confidence, reason, [], None, source_field, None
    if dimension == "analysis_levels":
        analysis_levels = infer_analysis_levels(record)
        return (
            "derived_from_source_default",
            0.8,
            "analysis level inferred from modality, scale, standard, and usability flags",
            analysis_levels,
            "inferred_metadata",
            "analysis_level_rules",
            None,
        )
    if dimension == "access_tiers":
        if access_tier == "unknown_access":
            return (
                "unknown_needs_review",
                0.35,
                "no source, license, or access policy rule resolved the access tier",
                [access_tier],
                "inferred_metadata",
                "access_tier_rules",
                None,
            )
        return (
            "derived_from_source_default",
            0.75,
            "access tier inferred from source, license, or access policy metadata",
            [access_tier],
            "inferred_metadata",
            "access_tier_rules",
            None,
        )
    return "unknown_needs_review", 0.25, "unsupported coverage dimension", [], None, None, None


def build_coverage_state_entries(
    records: Iterable[Any],
    *,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    ingested_at: str | None = None,
) -> list[CoverageStateEntry]:
    """Build one coverage-state row for every dataset × coverage dimension."""

    timestamp = ingested_at or _now_iso()
    states: list[CoverageStateEntry] = []
    for raw_record in records:
        record = _record_mapping(raw_record)
        if not _is_dataset_mapping(record):
            continue
        dataset_id = _record_id(record)
        source = str(record.get("source") or "unknown")
        source_id = str(record.get("source_id") or dataset_id)
        first_seen = str(record.get("created_at") or timestamp)
        access_tier = infer_access_tier(record)
        for dimension in COVERAGE_DIMENSIONS:
            (
                coverage_state,
                confidence,
                reason,
                value_ids,
                evidence_tier,
                source_field,
                evidence_text,
            ) = _state_for_dimension(record, dimension, access_tier)
            states.append(
                CoverageStateEntry(
                    state_id=_state_id(dataset_id, dimension, snapshot_id),
                    dataset_id=dataset_id,
                    source=source,
                    source_id=source_id,
                    dimension=dimension,
                    coverage_state=coverage_state,
                    confidence=confidence,
                    access_tier=access_tier,
                    reason=reason,
                    value_ids=value_ids,
                    evidence_tier=evidence_tier,
                    source_field=source_field,
                    evidence_text=evidence_text,
                    first_seen=first_seen,
                    ingested_at=timestamp,
                    snapshot_id=snapshot_id,
                )
            )
    return states


def _completion_item_id(dataset_id: str, dimension: str, snapshot_id: str) -> str:
    return f"coverage_completion:{snapshot_id}:{_norm(dataset_id)}:{dimension}"


def _completion_action(source: str, dimension: str) -> str:
    source_actions = SOURCE_DIMENSION_COMPLETION_ACTIONS.get(_norm(source), {})
    return source_actions.get(dimension) or DIMENSION_COMPLETION_ACTIONS.get(
        dimension,
        "inspect source metadata and linked publication text",
    )


def _completion_priority(source: str, dimension: str, access_tier: str) -> int:
    priority = DIMENSION_PRIORITY.get(dimension, 50)
    if _norm(source) in SOURCE_DIMENSION_COMPLETION_ACTIONS:
        priority += 5
    if access_tier == "open_access":
        priority += 3
    elif access_tier in {"restricted_access", "controlled_access"}:
        priority -= 10
    return max(1, min(priority, 100))


def build_completion_worklist(
    records: Iterable[Any],
    *,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    states_to_include: tuple[str, ...] = ("unknown_needs_review",),
) -> list[CoverageCompletionItem]:
    """Build source-aware work items for unresolved coverage states."""

    record_list: list[dict[str, Any]] = []
    records_by_id: dict[str, dict[str, Any]] = {}
    for raw_record in records:
        record = _record_mapping(raw_record)
        if not _is_dataset_mapping(record):
            continue
        record_list.append(record)
        records_by_id[_record_id(record)] = record

    included_states = set(states_to_include)
    worklist: list[CoverageCompletionItem] = []
    for state in build_coverage_state_entries(record_list, snapshot_id=snapshot_id):
        if state.coverage_state not in included_states:
            continue
        record = records_by_id.get(state.dataset_id, {})
        source = str(record.get("source") or state.source)
        title = str(record.get("title") or state.dataset_id)
        reason = state.reason or "coverage state requires review"
        worklist.append(
            CoverageCompletionItem(
                item_id=_completion_item_id(state.dataset_id, state.dimension, snapshot_id),
                dataset_id=state.dataset_id,
                source=source,
                source_id=state.source_id,
                title=title,
                dimension=state.dimension,
                current_state=state.coverage_state,
                priority=_completion_priority(source, state.dimension, state.access_tier),
                recommended_action=_completion_action(source, state.dimension),
                reason=reason,
                access_tier=state.access_tier,
                snapshot_id=snapshot_id,
            )
        )
    worklist.sort(
        key=lambda item: (
            -item.priority,
            item.source,
            item.dimension,
            item.dataset_id,
        )
    )
    return worklist


def _source_state_coverage(
    records: list[dict[str, Any]],
    state_entries: list[CoverageStateEntry],
) -> list[dict[str, Any]]:
    dataset_counts = Counter(str(record.get("source") or "unknown") for record in records)
    source_dimension_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for state in state_entries:
        source_dimension_counts[state.source][state.dimension][state.coverage_state] += 1

    rows: list[dict[str, Any]] = []
    for source, dataset_count in dataset_counts.items():
        row: dict[str, Any] = {
            "source": source,
            "dataset_count": dataset_count,
        }
        unknown_total = 0
        not_applicable_total = 0
        actionable_total = 0
        state_total = 0
        for dimension in COVERAGE_DIMENSIONS:
            counter = source_dimension_counts[source].get(dimension, Counter())
            unknown = counter.get("unknown_needs_review", 0)
            not_applicable = counter.get("not_applicable", 0)
            total = sum(counter.values())
            actionable = total - unknown
            row[f"{dimension}_unknown"] = unknown
            row[f"{dimension}_not_applicable"] = not_applicable
            row[f"{dimension}_actionable_state_rate"] = round(
                actionable / total,
                4,
            ) if total else 0.0
            unknown_total += unknown
            not_applicable_total += not_applicable
            actionable_total += actionable
            state_total += total
        row["unknown_total"] = unknown_total
        row["not_applicable_total"] = not_applicable_total
        row["actionable_state_rate"] = round(
            actionable_total / state_total,
            4,
        ) if state_total else 0.0
        rows.append(row)
    rows.sort(key=lambda item: (-item["unknown_total"], -item["dataset_count"], item["source"]))
    return rows


def _completion_worklist_summary(
    worklist: list[CoverageCompletionItem],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[CoverageCompletionItem]] = defaultdict(list)
    for item in worklist:
        grouped[(item.source, item.dimension)].append(item)
    rows: list[dict[str, Any]] = []
    for (source, dimension), items in grouped.items():
        top = max(items, key=lambda item: item.priority)
        rows.append(
            {
                "source": source,
                "dimension": dimension,
                "count": len(items),
                "max_priority": top.priority,
                "recommended_action": top.recommended_action,
            }
        )
    rows.sort(
        key=lambda item: (
            -item["count"],
            -item["max_priority"],
            item["source"],
            item["dimension"],
        )
    )
    return rows[:limit]


def _values_by_dataset(entries: Iterable[CoverageEntry], dimension: str) -> dict[str, set[str]]:
    values: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        if entry.dimension == dimension and entry.confidence >= DEFAULT_MIN_CONFIDENCE:
            values[entry.dataset_id].add(entry.value_id)
    return values


def _pair_counts(
    entries: list[CoverageEntry],
    left_dimension: str,
    right_dimension: str,
) -> Counter[tuple[str, str]]:
    left = _values_by_dataset(entries, left_dimension)
    right = _values_by_dataset(entries, right_dimension)
    counts: Counter[tuple[str, str]] = Counter()
    for dataset_id, left_values in left.items():
        for left_value in left_values:
            for right_value in right.get(dataset_id, set()):
                counts[(left_value, right_value)] += 1
    return counts


def _dark_pairs(
    pair_counts: Counter[tuple[str, str]],
    left_counts: Counter[str],
    right_counts: Counter[str],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for left_value, left_count in left_counts.items():
        for right_value, right_count in right_counts.items():
            observed = pair_counts.get((left_value, right_value), 0)
            if observed > 0:
                continue
            opportunity = left_count + right_count
            candidates.append(
                {
                    "left": left_value,
                    "right": right_value,
                    "observed": observed,
                    "left_count": left_count,
                    "right_count": right_count,
                    "opportunity_score": opportunity,
                }
            )
    candidates.sort(key=lambda item: (-item["opportunity_score"], item["left"], item["right"]))
    return candidates[:limit]


def _thin_pairs(
    pair_counts: Counter[tuple[str, str]],
    left_counts: Counter[str],
    right_counts: Counter[str],
    *,
    limit: int,
    target_count: int = 3,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for (left_value, right_value), observed in pair_counts.items():
        if observed <= 0 or observed >= target_count:
            continue
        opportunity = left_counts.get(left_value, 0) + right_counts.get(right_value, 0) - observed
        candidates.append(
            {
                "left": left_value,
                "right": right_value,
                "observed": observed,
                "target_count": target_count,
                "left_count": left_counts.get(left_value, 0),
                "right_count": right_counts.get(right_value, 0),
                "opportunity_score": opportunity,
            }
        )
    candidates.sort(key=lambda item: (item["observed"], -item["opportunity_score"], item["left"], item["right"]))
    return candidates[:limit]


def _source_coverage(
    records: list[dict[str, Any]],
    entries: list[CoverageEntry],
) -> list[dict[str, Any]]:
    source_dataset_counts = Counter(str(record.get("source") or "unknown") for record in records)
    present_by_source: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    low_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        if entry.dimension not in LABEL_FIELDS:
            continue
        if entry.confidence >= DEFAULT_MIN_CONFIDENCE:
            present_by_source[entry.source][entry.dimension].add(entry.dataset_id)
        if entry.confidence < DEFAULT_MIN_CONFIDENCE or entry.evidence_tier == "silver_inferred":
            low_by_source[entry.source][entry.dimension] += 1

    rows: list[dict[str, Any]] = []
    for source, dataset_count in source_dataset_counts.items():
        row: dict[str, Any] = {
            "source": source,
            "dataset_count": dataset_count,
        }
        missing_total = 0
        for dimension in LABEL_FIELDS:
            present = len(present_by_source[source].get(dimension, set()))
            missing = max(dataset_count - present, 0)
            missing_total += missing
            row[f"{dimension}_present"] = present
            row[f"{dimension}_missing"] = missing
            row[f"{dimension}_coverage_rate"] = round(present / dataset_count, 4)
            if low_by_source[source].get(dimension):
                row[f"{dimension}_review_needed"] = low_by_source[source][dimension]
        row["missing_total"] = missing_total
        rows.append(row)
    rows.sort(key=lambda item: (-item["missing_total"], -item["dataset_count"], item["source"]))
    return rows


def build_gap_report(
    records: Iterable[Any],
    *,
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    dark_pair_limit: int = 20,
) -> CoverageGapReport:
    """Build an honest corpus coverage gap report."""

    record_list = []
    for record in records:
        mapped = _record_mapping(record)
        if _is_dataset_mapping(mapped):
            record_list.append(mapped)
    entries = build_coverage_entries(record_list, snapshot_id=snapshot_id)
    state_entries = build_coverage_state_entries(record_list, snapshot_id=snapshot_id)
    completion_worklist = build_completion_worklist(record_list, snapshot_id=snapshot_id)
    dimension_counts: Counter[str] = Counter()
    value_counts: dict[str, Counter[str]] = {dimension: Counter() for dimension in COVERAGE_DIMENSIONS}
    low_confidence_counts: Counter[str] = Counter()
    open_access_dimension_counts: Counter[str] = Counter()
    evidence_tier_counts: Counter[str] = Counter()
    inferred_dimension_counts: Counter[str] = Counter()
    coverage_state_counts: dict[str, Counter[str]] = {
        dimension: Counter() for dimension in COVERAGE_DIMENSIONS
    }
    for entry in entries:
        dimension_counts[entry.dimension] += 1
        value_counts.setdefault(entry.dimension, Counter())[entry.value_id] += 1
        evidence_tier_counts[entry.evidence_tier] += 1
        if entry.evidence_tier == "inferred_metadata":
            inferred_dimension_counts[entry.dimension] += 1
        if entry.confidence < min_confidence or entry.evidence_tier == "silver_inferred":
            low_confidence_counts[entry.dimension] += 1
        if entry.access_tier == "open_access":
            open_access_dimension_counts[entry.dimension] += 1
    for state_entry in state_entries:
        coverage_state_counts[state_entry.dimension][state_entry.coverage_state] += 1

    missing_dimension_counts: Counter[str] = Counter()
    for record in record_list:
        for field_name in LABEL_FIELDS:
            if not _items_for_dimension(record, field_name):
                missing_dimension_counts[field_name] += 1

    species_counts = value_counts.get("species", Counter())
    region_counts = value_counts.get("brain_regions", Counter())
    modality_counts = value_counts.get("modalities", Counter())

    species_region_pairs = _pair_counts(entries, "species", "brain_regions")
    species_modality_pairs = _pair_counts(entries, "species", "modalities")
    region_modality_pairs = _pair_counts(entries, "brain_regions", "modalities")
    source_coverage = _source_coverage(record_list, entries)
    source_state_coverage = _source_state_coverage(record_list, state_entries)

    coverage_rates = {
        dimension: round(
            (len(record_list) - missing_dimension_counts.get(dimension, 0)) / len(record_list),
            4,
        )
        for dimension in LABEL_FIELDS
    } if record_list else {}
    field_state_coverage_rates = {
        dimension: round(sum(counter.values()) / len(record_list), 4)
        for dimension, counter in coverage_state_counts.items()
    } if record_list else {}
    actionable_state_rates = {
        dimension: round(
            (
                sum(counter.values())
                - counter.get("unknown_needs_review", 0)
            )
            / len(record_list),
            4,
        )
        for dimension, counter in coverage_state_counts.items()
    } if record_list else {}
    not_applicable_counts = {
        dimension: counter.get("not_applicable", 0)
        for dimension, counter in sorted(coverage_state_counts.items())
        if counter.get("not_applicable", 0)
    }
    unknown_state_counts = {
        dimension: counter.get("unknown_needs_review", 0)
        for dimension, counter in sorted(coverage_state_counts.items())
        if counter.get("unknown_needs_review", 0)
    }
    recommendations = _recommendations(
        dataset_count=len(record_list),
        missing=missing_dimension_counts,
        unknown_states=Counter(unknown_state_counts),
        not_applicable=Counter(not_applicable_counts),
        access_tier_counts=value_counts.get("access_tiers", Counter()),
        low_confidence=low_confidence_counts,
        source_coverage=source_coverage,
    )
    return CoverageGapReport(
        generated_at=_now_iso(),
        snapshot_id=snapshot_id,
        dataset_count=len(record_list),
        entry_count=len(entries),
        dimension_counts=dict(sorted(dimension_counts.items())),
        value_counts={
            dimension: dict(counter.most_common(50))
            for dimension, counter in sorted(value_counts.items())
            if counter
        },
        missing_dimension_counts=dict(sorted(missing_dimension_counts.items())),
        low_confidence_counts=dict(sorted(low_confidence_counts.items())),
        access_tier_counts=dict(value_counts.get("access_tiers", Counter()).most_common()),
        analysis_level_counts=dict(
            value_counts.get("analysis_levels", Counter()).most_common()
        ),
        open_access_dimension_counts=dict(sorted(open_access_dimension_counts.items())),
        coverage_rates=coverage_rates,
        field_state_coverage_rates=field_state_coverage_rates,
        actionable_state_rates=actionable_state_rates,
        coverage_state_counts={
            dimension: dict(counter.most_common())
            for dimension, counter in sorted(coverage_state_counts.items())
            if counter
        },
        not_applicable_counts=not_applicable_counts,
        unknown_state_counts=unknown_state_counts,
        evidence_tier_counts=dict(evidence_tier_counts.most_common()),
        inferred_dimension_counts=dict(sorted(inferred_dimension_counts.items())),
        source_state_coverage=source_state_coverage,
        completion_worklist_summary=_completion_worklist_summary(completion_worklist),
        source_coverage=source_coverage,
        thin_species_region_pairs=_thin_pairs(
            species_region_pairs,
            species_counts,
            region_counts,
            limit=dark_pair_limit,
        ),
        thin_species_modality_pairs=_thin_pairs(
            species_modality_pairs,
            species_counts,
            modality_counts,
            limit=dark_pair_limit,
        ),
        thin_region_modality_pairs=_thin_pairs(
            region_modality_pairs,
            region_counts,
            modality_counts,
            limit=dark_pair_limit,
        ),
        dark_species_region_pairs=_dark_pairs(
            species_region_pairs,
            species_counts,
            region_counts,
            limit=dark_pair_limit,
        ),
        dark_species_modality_pairs=_dark_pairs(
            species_modality_pairs,
            species_counts,
            modality_counts,
            limit=dark_pair_limit,
        ),
        dark_region_modality_pairs=_dark_pairs(
            region_modality_pairs,
            region_counts,
            modality_counts,
            limit=dark_pair_limit,
        ),
        recommendations=recommendations,
    )


def _recommendations(
    *,
    dataset_count: int,
    missing: Counter[str],
    unknown_states: Counter[str],
    not_applicable: Counter[str],
    access_tier_counts: Counter[str],
    low_confidence: Counter[str],
    source_coverage: list[dict[str, Any]],
) -> list[str]:
    recommendations: list[str] = []
    if not dataset_count:
        return ["No datasets were available for coverage analysis."]
    for dimension, unknown_count in unknown_states.most_common():
        rate = unknown_count / dataset_count
        if rate >= 0.1:
            recommendations.append(
                f"Resolve unknown {dimension} states: {unknown_count}/{dataset_count} "
                "dataset slots still need source/file/paper enrichment."
            )
    for dimension, missing_count in missing.most_common():
        unknown_count = unknown_states.get(dimension, 0)
        if unknown_count:
            continue
        rate = missing_count / dataset_count
        if rate >= 0.5:
            recommendations.append(
                f"Prioritize {dimension}: missing on {missing_count}/{dataset_count} datasets "
                f"after excluding {not_applicable.get(dimension, 0)} explicit N/A slots."
            )
    if access_tier_counts.get("unknown_access", 0):
        recommendations.append(
            "Normalize licenses/access metadata so open-access gaps are actionable."
        )
    for row in source_coverage[:5]:
        if row["dataset_count"] < 25:
            continue
        worst_dimension = max(
            LABEL_FIELDS,
            key=lambda dimension: row.get(f"{dimension}_missing", 0),
        )
        if _source_dimension_probably_not_applicable(str(row["source"]), worst_dimension):
            continue
        missing_count = row.get(f"{worst_dimension}_missing", 0)
        if missing_count:
            recommendations.append(
                f"Backfill {worst_dimension} for `{row['source']}` "
                f"({missing_count}/{row['dataset_count']} missing)."
            )
    for dimension, count in low_confidence.most_common():
        if count:
            recommendations.append(
                f"Review {dimension} silver/low-confidence labels before treating coverage as gold."
            )
    if not recommendations:
        recommendations.append("No critical metadata gap exceeded the default thresholds.")
    return recommendations


def _source_dimension_probably_not_applicable(source: str, dimension: str) -> bool:
    source_id = _norm(source)
    if dimension in {"tasks", "behavioral_events"} and source_id in STATIC_BEHAVIOR_NA_SOURCES:
        return True
    return dimension == "behavioral_events" and source_id == "neurovault"


def render_gap_report_markdown(report: CoverageGapReport) -> str:
    """Render coverage report for human review."""

    lines = [
        "# Coverage Ledger Gap Report",
        "",
        f"- Snapshot: `{report.snapshot_id}`",
        f"- Generated: {report.generated_at}",
        f"- Datasets: {report.dataset_count}",
        f"- Coverage entries: {report.entry_count}",
        "",
        "## Executive Review",
        "",
        *(_executive_review_lines(report)),
        "",
        "## Dimension Coverage",
        "",
        "| Dimension | Value Coverage | State Coverage | Actionable State | Entries | Missing Values | Unknown State | N/A | Review Needed | Inferred Entries | Open Access Entries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dimension in COVERAGE_DIMENSIONS:
        lines.append(
            "| "
            + " | ".join(
                [
                    dimension,
                    _pct(report.coverage_rates.get(dimension, 1.0)),
                    _pct(report.field_state_coverage_rates.get(dimension, 0.0)),
                    _pct(report.actionable_state_rates.get(dimension, 0.0)),
                    str(report.dimension_counts.get(dimension, 0)),
                    str(report.missing_dimension_counts.get(dimension, 0)),
                    str(report.unknown_state_counts.get(dimension, 0)),
                    str(report.not_applicable_counts.get(dimension, 0)),
                    str(report.low_confidence_counts.get(dimension, 0)),
                    str(report.inferred_dimension_counts.get(dimension, 0)),
                    str(report.open_access_dimension_counts.get(dimension, 0)),
                ]
            )
            + " |"
        )

    lines.extend(["", "## Coverage States", ""])
    lines.extend(_coverage_state_lines(report.coverage_state_counts))
    lines.extend(["", "## Evidence Tiers", ""])
    lines.extend(_bullet_counts(report.evidence_tier_counts))
    lines.extend(["", "## Access Tiers", ""])
    lines.extend(_bullet_counts(report.access_tier_counts))
    lines.extend(["", "## Analysis Levels", ""])
    lines.extend(_bullet_counts(report.analysis_level_counts))
    lines.extend(["", "## Source Gap Hotspots", ""])
    lines.extend(_source_coverage_lines(report.source_coverage[:15]))
    lines.extend(["", "## Source State Hotspots", ""])
    lines.extend(_source_state_coverage_lines(report.source_state_coverage[:15]))
    lines.extend(["", "## Completion Worklist Summary", ""])
    lines.extend(_completion_summary_lines(report.completion_worklist_summary))
    lines.extend(["", "## Top Covered Values", ""])
    lines.extend(_top_value_lines(report.value_counts))
    lines.extend(["", "## Thin Species-Region Pairs", ""])
    lines.extend(_thin_pair_lines(report.thin_species_region_pairs))
    lines.extend(["", "## Thin Species-Modality Pairs", ""])
    lines.extend(_thin_pair_lines(report.thin_species_modality_pairs))
    lines.extend(["", "## Thin Region-Modality Pairs", ""])
    lines.extend(_thin_pair_lines(report.thin_region_modality_pairs))
    lines.extend(["", "## Largest Dark Species-Region Pairs", ""])
    lines.append(
        "These are candidate acquisition/enrichment gaps based on marginal coverage; "
        "they are not yet filtered by species-specific anatomical validity."
    )
    lines.append("")
    lines.extend(_dark_pair_lines(report.dark_species_region_pairs))
    lines.extend(["", "## Largest Dark Species-Modality Pairs", ""])
    lines.append(
        "These pairs need curator review because some modality/species combinations are "
        "scientifically impossible or intentionally absent."
    )
    lines.append("")
    lines.extend(_dark_pair_lines(report.dark_species_modality_pairs))
    lines.extend(["", "## Largest Dark Region-Modality Pairs", ""])
    lines.append(
        "These mostly identify where atlas-level regions and modality families do not "
        "co-occur in the current corpus."
    )
    lines.append("")
    lines.extend(_dark_pair_lines(report.dark_region_modality_pairs))
    lines.extend(["", "## Recommendations", ""])
    lines.extend(f"- {item}" for item in report.recommendations)
    return "\n".join(lines).rstrip() + "\n"


def _bullet_counts(counts: Mapping[str, int]) -> list[str]:
    if not counts:
        return ["- No entries."]
    return [f"- {key}: {value}" for key, value in counts.items()]


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _executive_review_lines(report: CoverageGapReport) -> list[str]:
    lines = [
        f"- Structured species coverage: {_pct(report.coverage_rates.get('species', 0.0))}",
        f"- Structured brain-region coverage: {_pct(report.coverage_rates.get('brain_regions', 0.0))}",
        f"- Structured modality coverage: {_pct(report.coverage_rates.get('modalities', 0.0))}",
        f"- Recording-scale coverage after conservative backfill: {_pct(report.coverage_rates.get('recording_scales', 0.0))}",
        "- State coverage: "
        + ", ".join(
            f"{dimension}={_pct(report.field_state_coverage_rates.get(dimension, 0.0))}"
            for dimension in LABEL_FIELDS
        ),
    ]
    unknown_total = sum(report.unknown_state_counts.values())
    if unknown_total:
        lines.append(
            f"- Unknown state slots still needing source/file/paper enrichment: {unknown_total}"
        )
    if report.inferred_dimension_counts.get("recording_scales"):
        lines.append(
            "- Recording-scale coverage is mostly inferred from modality and text; "
            "treat it as sortable silver metadata until file-level validation."
        )
    if report.missing_dimension_counts.get("behavioral_events", 0) > report.dataset_count * 0.75:
        lines.append(
            "- Behavioral-event coverage remains sparse; HED/event-table enrichment should be a priority."
        )
    return lines


def _coverage_state_lines(counts: Mapping[str, Mapping[str, int]]) -> list[str]:
    if not counts:
        return ["- No state rows."]
    lines = [
        "| Dimension | Observed | File-Derived | Source Default | Inferred Silver | Not Applicable | Restricted | Unknown |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dimension in COVERAGE_DIMENSIONS:
        row = counts.get(dimension, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    dimension,
                    str(row.get("observed", 0)),
                    str(row.get("derived_from_file", 0)),
                    str(row.get("derived_from_source_default", 0)),
                    str(row.get("inferred_silver", 0)),
                    str(row.get("not_applicable", 0)),
                    str(row.get("restricted_unavailable", 0)),
                    str(row.get("unknown_needs_review", 0)),
                ]
            )
            + " |"
        )
    return lines


def _source_coverage_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No source coverage rows."]
    lines = [
        "| Source | Datasets | Species | Regions | Modalities | Scales | Tasks | Events |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source"]),
                    str(row["dataset_count"]),
                    _pct(row.get("species_coverage_rate", 0.0)),
                    _pct(row.get("brain_regions_coverage_rate", 0.0)),
                    _pct(row.get("modalities_coverage_rate", 0.0)),
                    _pct(row.get("recording_scales_coverage_rate", 0.0)),
                    _pct(row.get("tasks_coverage_rate", 0.0)),
                    _pct(row.get("behavioral_events_coverage_rate", 0.0)),
                ]
            )
            + " |"
        )
    return lines


def _source_state_coverage_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["No source state rows."]
    lines = [
        "| Source | Datasets | Actionable State | Unknown Total | Species Unknown | Regions Unknown | Modalities Unknown | Scales Unknown | Tasks Unknown | Events Unknown | N/A Total |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source"]),
                    str(row["dataset_count"]),
                    _pct(row.get("actionable_state_rate", 0.0)),
                    str(row.get("unknown_total", 0)),
                    str(row.get("species_unknown", 0)),
                    str(row.get("brain_regions_unknown", 0)),
                    str(row.get("modalities_unknown", 0)),
                    str(row.get("recording_scales_unknown", 0)),
                    str(row.get("tasks_unknown", 0)),
                    str(row.get("behavioral_events_unknown", 0)),
                    str(row.get("not_applicable_total", 0)),
                ]
            )
            + " |"
        )
    return lines


def _completion_summary_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- No unresolved completion items."]
    lines = [
        "| Source | Dimension | Items | Max Priority | Recommended Action |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["source"]),
                    str(row["dimension"]),
                    str(row["count"]),
                    str(row["max_priority"]),
                    str(row["recommended_action"]),
                ]
            )
            + " |"
        )
    return lines


def _top_value_lines(value_counts: Mapping[str, Mapping[str, int]]) -> list[str]:
    lines: list[str] = []
    for dimension in ("species", "brain_regions", "modalities", "recording_scales", "tasks"):
        counts = value_counts.get(dimension, {})
        if not counts:
            continue
        top = ", ".join(f"{key} ({value})" for key, value in list(counts.items())[:10])
        lines.append(f"- {dimension}: {top}")
    if not lines:
        lines.append("- No covered values.")
    return lines


def _thin_pair_lines(pairs: list[dict[str, Any]]) -> list[str]:
    if not pairs:
        return ["- No thin pairs found."]
    return [
        (
            "- {left} × {right}: observed={observed}/{target_count}, "
            "opportunity={opportunity_score}"
        ).format(**pair)
        for pair in pairs
    ]


def _dark_pair_lines(pairs: list[dict[str, Any]]) -> list[str]:
    if not pairs:
        return ["- No dark pairs found."]
    return [
        (
            "- {left} × {right}: observed={observed}, "
            "opportunity={opportunity_score}"
        ).format(**pair)
        for pair in pairs
    ]


class CoverageLedger:
    """Convenience wrapper for building and writing coverage artifacts."""

    def __init__(
        self,
        records: Iterable[Any],
        *,
        snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    ) -> None:
        self.records = [_record_mapping(record) for record in records]
        self.snapshot_id = snapshot_id

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        snapshot_id: str = DEFAULT_SNAPSHOT_ID,
    ) -> CoverageLedger:
        return cls(load_dataset_mappings(path), snapshot_id=snapshot_id)

    def entries(self) -> list[CoverageEntry]:
        return build_coverage_entries(self.records, snapshot_id=self.snapshot_id)

    def states(self) -> list[CoverageStateEntry]:
        return build_coverage_state_entries(self.records, snapshot_id=self.snapshot_id)

    def completion_worklist(self) -> list[CoverageCompletionItem]:
        return build_completion_worklist(self.records, snapshot_id=self.snapshot_id)

    def gap_report(self) -> CoverageGapReport:
        return build_gap_report(self.records, snapshot_id=self.snapshot_id)

    def write(self, out_dir: str | Path) -> dict[str, Path]:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        entries = self.entries()
        states = self.states()
        worklist = self.completion_worklist()
        report = self.gap_report()
        entries_path = output / "coverage_entries.jsonl"
        states_path = output / "coverage_states.jsonl"
        worklist_path = output / "coverage_completion_worklist.jsonl"
        report_json_path = output / "coverage_gap_report.json"
        report_md_path = output / "coverage_gap_report.md"
        snapshot_path = output / "coverage_snapshot.json"
        with entries_path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry.model_dump(), sort_keys=True))
                handle.write("\n")
        with states_path.open("w", encoding="utf-8") as handle:
            for state in states:
                handle.write(json.dumps(state.model_dump(), sort_keys=True))
                handle.write("\n")
        with worklist_path.open("w", encoding="utf-8") as handle:
            for item in worklist:
                handle.write(json.dumps(item.model_dump(), sort_keys=True))
                handle.write("\n")
        report_json_path.write_text(
            json.dumps(report.model_dump(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        report_md_path.write_text(render_gap_report_markdown(report), encoding="utf-8")
        snapshot_path.write_text(
            json.dumps(
                {
                    "snapshot_id": self.snapshot_id,
                    "generated_at": report.generated_at,
                    "dataset_count": report.dataset_count,
                    "entry_count": report.entry_count,
                    "dimension_counts": report.dimension_counts,
                    "access_tier_counts": report.access_tier_counts,
                    "analysis_level_counts": report.analysis_level_counts,
                    "coverage_state_counts": report.coverage_state_counts,
                    "field_state_coverage_rates": report.field_state_coverage_rates,
                    "actionable_state_rates": report.actionable_state_rates,
                    "completion_worklist_items": len(worklist),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "entries": entries_path,
            "states": states_path,
            "completion_worklist": worklist_path,
            "gap_report_json": report_json_path,
            "gap_report_md": report_md_path,
            "snapshot": snapshot_path,
        }


def _cmd_report(args: argparse.Namespace) -> int:
    ledger = CoverageLedger.from_path(args.input, snapshot_id=args.snapshot_id)
    paths = ledger.write(args.out)
    print(json.dumps({key: str(path) for key, path in paths.items()}, indent=2))
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    ledger = CoverageLedger.from_path(args.input, snapshot_id=args.snapshot_id)
    entries = [
        entry
        for entry in ledger.entries()
        if (args.dimension is None or entry.dimension == args.dimension)
        and (args.value is None or entry.value_id == _norm(args.value))
    ]
    print(json.dumps([entry.model_dump() for entry in entries[: args.limit]], indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.coverage")
    subparsers = parser.add_subparsers(dest="command", required=True)

    report_parser = subparsers.add_parser("report", help="Build coverage ledger artifacts.")
    report_parser.add_argument("--input", required=True, type=Path)
    report_parser.add_argument("--out", required=True, type=Path)
    report_parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    report_parser.set_defaults(func=_cmd_report)

    query_parser = subparsers.add_parser("query", help="Print matching coverage entries.")
    query_parser.add_argument("--input", required=True, type=Path)
    query_parser.add_argument("--snapshot-id", default=DEFAULT_SNAPSHOT_ID)
    query_parser.add_argument("--dimension")
    query_parser.add_argument("--value")
    query_parser.add_argument("--limit", type=int, default=20)
    query_parser.set_defaults(func=_cmd_query)

    args = parser.parse_args(argv)
    return int(args.func(args))
