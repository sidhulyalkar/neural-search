"""Deterministic scientific label extraction from metadata and text."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from neural_search.ontology import (
    LabelMatch,
    match_behavior_labels,
    match_tasks,
    normalize_text,
)
from neural_search.schemas import ExtractionResult, LabelEvidence

MODALITY_SYNONYMS: dict[str, list[str]] = {
    "calcium_imaging": ["calcium imaging", "two photon", "2 photon", "2p", "gcamp"],
    "extracellular_ephys": ["extracellular", "spike sorting", "electrophysiology", "ephys"],
    "neuropixels": ["neuropixels", "neuropixel"],
    "eeg": ["eeg", "electroencephalography"],
    "ecog": ["ecog", "electrocorticography"],
    "ieeg": ["ieeg", "intracranial eeg", "seeg"],
    "meg": ["meg", "magnetoencephalography"],
    "fmri": ["fmri", "functional mri", "bold"],
    "lfp": ["lfp", "local field potential"],
    "fiber_photometry": ["fiber photometry", "photometry"],
    "behavior_video": ["behavior video", "behaviour video", "video tracking", "camera"],
    "pose_tracking": ["pose tracking", "deeplabcut", "sleap", "kinematics"],
    "emg": ["emg", "electromyography"],
    "audio": ["audio", "sound recording", "microphone"],
}

SPECIES_SYNONYMS: dict[str, list[str]] = {
    "mouse": ["mouse", "mice", "mus musculus"],
    "rat": ["rat", "rats", "rattus"],
    "human": ["human", "humans", "participant", "subject"],
    "macaque": ["macaque", "monkey", "non human primate", "nhp"],
    "zebrafish": ["zebrafish", "danio rerio"],
}

DATA_STANDARD_SYNONYMS: dict[str, list[str]] = {
    "NWB": [".nwb", "nwb", "neurodata without borders"],
    "BIDS": ["bids", "dataset_description.json", "participants.tsv", "events.tsv"],
}

BRAIN_REGION_SYNONYMS: dict[str, list[str]] = {
    "mPFC": ["mpfc", "medial prefrontal"],
    "ACC": ["acc", "anterior cingulate"],
    "OFC": ["ofc", "orbitofrontal"],
    "striatum": ["striatum", "striatal"],
    "motor_cortex": ["motor cortex", "m1", "primary motor"],
    "visual_cortex": ["visual cortex", "v1", "v2", "v4"],
    "hippocampus": ["hippocampus", "hippocampal", "ca1", "ca3", "dentate"],
    "amygdala": ["amygdala", "amygdalar"],
    "thalamus": ["thalamus", "thalamic"],
    "parietal_cortex": ["parietal cortex", "ppc"],
    "somatosensory_cortex": ["somatosensory cortex", "s1", "barrel cortex"],
}


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_stringify(item)}" for key, item in value.items())
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return " ".join(_stringify(item) for item in value)
    return str(value)


def _to_label(match: LabelMatch) -> LabelEvidence:
    return LabelEvidence(
        id=match.id,
        label=match.label,
        confidence=match.confidence,
        evidence=match.evidence,
        category=match.category,
    )


def _match_dictionary(
    text: str, entries: dict[str, list[str]], category: str
) -> list[LabelEvidence]:
    normalized = normalize_text(text)
    matches: list[LabelEvidence] = []
    for label_id, synonyms in entries.items():
        best: LabelEvidence | None = None
        for synonym in [label_id, *synonyms]:
            normalized_synonym = normalize_text(synonym)
            if not normalized_synonym:
                continue
            if re.search(rf"(?<!\w){re.escape(normalized_synonym)}(?!\w)", normalized):
                candidate = LabelEvidence(
                    id=label_id,
                    label=label_id,
                    confidence=0.92 if synonym != label_id else 0.95,
                    evidence=synonym,
                    category=category,
                )
                if best is None or candidate.confidence > best.confidence:
                    best = candidate
        if best:
            matches.append(best)
    return sorted(matches, key=lambda item: item.confidence, reverse=True)


def _missing_fields(
    title: str | None,
    description: str | None,
    source_metadata: Mapping[str, Any] | None,
    result: ExtractionResult,
) -> list[str]:
    missing: list[str] = []
    metadata = source_metadata or {}
    if not title:
        missing.append("title")
    if not description:
        missing.append("description")
    if not metadata.get("license") and not metadata.get("license_url"):
        missing.append("license")
    if not result.species:
        missing.append("species")
    if not result.modalities:
        missing.append("modalities")
    if not result.tasks:
        missing.append("tasks")
    if not result.behaviors:
        missing.append("behaviors")
    return missing


def extract_dataset_labels(
    title: str | None = None,
    description: str | None = None,
    file_paths: Iterable[str | Path] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    linked_paper_abstracts: Iterable[str] | None = None,
) -> ExtractionResult:
    """Extract deterministic labels from dataset metadata and linked paper text."""

    file_text = " ".join(str(path) for path in (file_paths or []))
    abstract_text = " ".join(linked_paper_abstracts or [])
    metadata_text = _stringify(source_metadata or {})
    combined = " ".join(
        part for part in [title or "", description or "", file_text, metadata_text, abstract_text] if part
    )

    result = ExtractionResult(
        tasks=[_to_label(match) for match in match_tasks(combined)],
        behaviors=[_to_label(match) for match in match_behavior_labels(combined)],
        modalities=_match_dictionary(combined, MODALITY_SYNONYMS, "modality"),
        brain_regions=_match_dictionary(combined, BRAIN_REGION_SYNONYMS, "brain_region"),
        species=_match_dictionary(combined, SPECIES_SYNONYMS, "species"),
        data_standards=_match_dictionary(combined, DATA_STANDARD_SYNONYMS, "data_standard"),
    )
    result.missing_fields = _missing_fields(title, description, source_metadata, result)
    return result
