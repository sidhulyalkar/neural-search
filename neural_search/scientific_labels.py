"""Conservative scientific label extraction for normalized corpus records."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from neural_search.normalized import make_evidence_label_id
from neural_search.schemas import (
    EvidenceLabel,
    NormalizedDatasetRecord,
    NormalizedPaperRecord,
)

EXTRACTOR_NAME = "rule_based_scientific_label_extractor"
EXTRACTOR_VERSION = "v0.3.0"


@dataclass(frozen=True)
class LabelRule:
    label_type: str
    label_id: str
    label: str
    terms: tuple[str, ...]


RULES: tuple[LabelRule, ...] = (
    LabelRule("species", "mouse", "mouse", ("mouse", "mice", "mus musculus")),
    LabelRule("species", "rat", "rat", ("rat", "rats", "rattus norvegicus")),
    LabelRule(
        "species",
        "human",
        "human",
        ("human", "humans", "participant", "participants", "patient", "patients", "subject", "subjects"),
    ),
    LabelRule(
        "species",
        "nonhuman_primate",
        "non-human primate",
        ("macaque", "monkey", "non-human primate", "non human primate", "nhp", "rhesus"),
    ),
    LabelRule("species", "zebrafish", "zebrafish", ("zebrafish",)),
    LabelRule("species", "drosophila", "drosophila", ("drosophila", "fly")),
    LabelRule(
        "modality",
        "calcium_imaging",
        "calcium imaging",
        ("calcium imaging", "two-photon", "two photon", "2p", "miniscope", "one-photon", "one photon"),
    ),
    LabelRule(
        "modality",
        "electrophysiology",
        "electrophysiology",
        ("electrophysiology", "extracellular", "spike sorting", "single unit", "multiunit"),
    ),
    LabelRule("modality", "neuropixels", "Neuropixels", ("neuropixels",)),
    LabelRule("modality", "lfp", "LFP", ("lfp", "local field potential")),
    LabelRule("modality", "eeg", "EEG", ("eeg",)),
    LabelRule(
        "modality",
        "ecog",
        "ECoG/iEEG",
        ("ecog", "electrocorticography", "ieeg", "intracranial eeg"),
    ),
    LabelRule("modality", "fmri", "fMRI", ("fmri", "bold")),
    LabelRule("modality", "fiber_photometry", "fiber photometry", ("fiber photometry", "photometry")),
    LabelRule(
        "modality",
        "behavior_tracking",
        "behavior tracking",
        ("behavior", "behavioral", "video tracking", "pose estimation", "deeplabcut", "facemap"),
    ),
    LabelRule("modality", "eye_tracking", "eye tracking", ("eye tracking", "pupil", "pupillometry")),
    LabelRule("modality", "emg", "EMG", ("emg",)),
    LabelRule("brain_region", "visual_cortex", "visual cortex", ("v1", "primary visual cortex", "visual cortex")),
    LabelRule("brain_region", "motor_cortex", "motor cortex", ("m1", "motor cortex", "primary motor cortex")),
    LabelRule("brain_region", "orbitofrontal_cortex", "orbitofrontal cortex", ("ofc", "orbitofrontal cortex")),
    LabelRule("brain_region", "prefrontal_cortex", "prefrontal cortex", ("pfc", "prefrontal cortex")),
    LabelRule("brain_region", "hippocampus", "hippocampus", ("hippocampus", "ca1", "ca3", "dentate gyrus")),
    LabelRule(
        "brain_region",
        "striatum",
        "striatum",
        ("striatum", "dorsal striatum", "ventral striatum", "nucleus accumbens"),
    ),
    LabelRule("brain_region", "thalamus", "thalamus", ("thalamus",)),
    LabelRule("brain_region", "amygdala", "amygdala", ("amygdala",)),
    LabelRule("brain_region", "auditory_cortex", "auditory cortex", ("auditory cortex",)),
    LabelRule("brain_region", "somatosensory_cortex", "somatosensory cortex", ("somatosensory cortex", "s1")),
    LabelRule("brain_region", "basal_ganglia", "basal ganglia", ("basal ganglia",)),
    LabelRule("task", "go_nogo", "go/no-go", ("go/no-go", "go nogo", "nogo", "response inhibition")),
    LabelRule("task", "reversal_learning", "reversal learning", ("reversal learning", "probabilistic reversal")),
    LabelRule("task", "delay_discounting", "delay discounting", ("delay discounting", "temporal discounting")),
    LabelRule("task", "motor_imagery", "motor imagery", ("motor imagery",)),
    LabelRule("task", "reaching", "reaching", ("reaching", "reach-to-grasp", "reach to grasp", "center-out reaching", "center out reaching")),
    LabelRule(
        "task",
        "visual_decision_making",
        "visual decision making",
        ("visual decision", "visual discrimination", "orientation discrimination"),
    ),
    LabelRule("task", "speech_production", "speech production", ("speech production", "articulation", "phoneme", "syllable")),
    LabelRule("task", "auditory_task", "auditory task", ("auditory task", "auditory discrimination")),
    LabelRule("task", "seizure_monitoring", "seizure monitoring", ("seizure monitoring", "seizure detection")),
    LabelRule("task", "sleep_staging", "sleep staging", ("sleep staging", "sleep task")),
    LabelRule("task", "spatial_navigation", "spatial navigation", ("navigation", "spatial navigation", "virtual reality navigation")),
    LabelRule("task", "fear_conditioning", "fear conditioning", ("fear conditioning",)),
    LabelRule("task", "foraging", "foraging", ("foraging",)),
    LabelRule("task", "reward_learning", "reward learning", ("reward learning", "reinforcement learning task")),
    LabelRule("behavioral_event", "lick", "lick", ("lick", "licking")),
    LabelRule("behavioral_event", "reward", "reward", ("reward", "water reward", "sucrose")),
    LabelRule("behavioral_event", "choice", "choice", ("choice", "decision")),
    LabelRule("behavioral_event", "response", "response", ("response", "lever press", "button press", "key press")),
    LabelRule("behavioral_event", "stimulus_onset", "stimulus onset", ("stimulus onset", "cue onset", "cue")),
    LabelRule("behavioral_event", "trial_start", "trial start", ("trial start",)),
    LabelRule("behavioral_event", "trial_end", "trial end", ("trial end",)),
    LabelRule("behavioral_event", "movement_onset", "movement onset", ("movement onset",)),
    LabelRule("behavioral_event", "reach_onset", "reach onset", ("reach onset",)),
    LabelRule("behavioral_event", "speech_onset", "speech onset", ("speech onset", "voice onset")),
    LabelRule("behavioral_event", "seizure_onset", "seizure onset", ("seizure onset",)),
    LabelRule("behavioral_event", "sleep_stage", "sleep stage", ("sleep stage",)),
    LabelRule(
        "behavioral_event",
        "trial_outcome",
        "trial outcome",
        ("error", "false alarm", "miss", "hit", "correct rejection", "trial outcome"),
    ),
    LabelRule("analysis_goal", "decoding", "decoding", ("decoding", "classifier", "classification")),
    LabelRule("analysis_goal", "encoding_modeling", "encoding modeling", ("encoding model", "encoding")),
    LabelRule(
        "analysis_goal",
        "q_learning_modeling",
        "Q-learning modeling",
        ("q-learning", "q learning", "reinforcement learning model", "value update", "reward prediction error"),
    ),
    LabelRule("analysis_goal", "state_space_modeling", "state-space modeling", ("state space", "latent dynamics", "dynamical system")),
    LabelRule("analysis_goal", "event_aligned_analysis", "event-aligned analysis", ("event aligned", "peri-event", "peri event", "psth", "trial aligned")),
    LabelRule("analysis_goal", "functional_connectivity", "functional connectivity", ("functional connectivity", "coherence", "granger")),
    LabelRule("analysis_goal", "representational_similarity_analysis", "representational similarity analysis", ("representational similarity", "rsa")),
    LabelRule("analysis_goal", "brain_behavior_alignment", "brain-behavior alignment", ("brain behavior", "neural-behavioral", "neural behavioral", "behavior prediction")),
    LabelRule("analysis_goal", "seizure_detection", "seizure detection", ("seizure detection",)),
    LabelRule("analysis_goal", "sleep_stage_classification", "sleep stage classification", ("sleep classification", "sleep staging")),
    LabelRule("analysis_goal", "bci_decoding", "BCI decoding", ("bci", "brain computer interface")),
    LabelRule("modeling_method", "q_learning", "Q-learning", ("q-learning", "q learning")),
    LabelRule("modeling_method", "state_space_model", "state-space model", ("state space", "latent dynamics", "dynamical system")),
    LabelRule("modeling_method", "classifier", "classifier", ("classifier", "classification")),
    LabelRule("modeling_method", "encoding_model", "encoding model", ("encoding model",)),
    LabelRule("modeling_method", "granger_causality", "Granger causality", ("granger",)),
    LabelRule("data_standard", "nwb", "NWB", ("nwb", "neurodata without borders")),
    LabelRule("data_standard", "bids", "BIDS", ("bids", "brain imaging data structure")),
    LabelRule("data_standard", "dandi", "DANDI", ("dandiset", "dandi")),
    LabelRule("data_standard", "openneuro", "OpenNeuro", ("openneuro",)),
    LabelRule("file_format", "zarr", "Zarr", ("zarr",)),
    LabelRule("file_format", "hdf5", "HDF5", ("hdf5", "h5")),
    LabelRule("file_format", "matlab", "MATLAB", ("mat", "matlab")),
    LabelRule("file_format", "csv", "CSV", ("csv",)),
    LabelRule("file_format", "tsv", "TSV", ("tsv",)),
    LabelRule("file_format", "parquet", "Parquet", ("parquet",)),
    LabelRule("file_format", "json", "JSON", ("json",)),
    LabelRule("stimulus_type", "visual_stimulus", "visual stimulus", ("visual stimulus", "drifting grating", "natural movie", "image", "scene")),
    LabelRule("stimulus_type", "auditory_stimulus", "auditory stimulus", ("auditory stimulus", "tone", "sound", "speech audio")),
    LabelRule("stimulus_type", "tactile_stimulus", "tactile stimulus", ("tactile", "whisker", "somatosensory")),
    LabelRule("stimulus_type", "olfactory_stimulus", "olfactory stimulus", ("reward cue", "odor cue", "olfactory")),
    LabelRule("stimulus_type", "stimulation", "stimulation", ("electrical stimulation", "optogenetic stimulation", "optogenetics")),
    LabelRule("subject_state", "awake", "awake", ("awake",)),
    LabelRule("subject_state", "anesthetized", "anesthetized", ("anesthetized", "anesthesia")),
    LabelRule("subject_state", "freely_moving", "freely moving", ("freely moving",)),
    LabelRule("subject_state", "head_fixed", "head-fixed", ("head-fixed", "head fixed")),
    LabelRule("subject_state", "sleep", "sleep", ("sleep", "rem", "non-rem", "non rem", "nrem")),
    LabelRule("subject_state", "behaving", "behaving", ("active behavior", "behaving")),
    LabelRule("subject_state", "resting_state", "resting state", ("resting state", "rest")),
    LabelRule("disease_state", "epilepsy", "epilepsy", ("epilepsy", "seizure")),
    LabelRule("clinical_condition", "epilepsy", "epilepsy", ("epilepsy", "seizure")),
    LabelRule("disease_state", "parkinsons", "Parkinson's disease", ("parkinson", "parkinson's")),
    LabelRule("clinical_condition", "parkinsons", "Parkinson's disease", ("parkinson", "parkinson's")),
    LabelRule("disease_state", "stroke", "stroke", ("stroke",)),
    LabelRule("clinical_condition", "stroke", "stroke", ("stroke",)),
    LabelRule("disease_state", "als", "ALS", ("als",)),
    LabelRule("clinical_condition", "als", "ALS", ("als",)),
    LabelRule("disease_state", "traumatic_brain_injury", "traumatic brain injury", ("traumatic brain injury", "tbi")),
    LabelRule("clinical_condition", "traumatic_brain_injury", "traumatic brain injury", ("traumatic brain injury", "tbi")),
    LabelRule("disease_state", "depression", "depression", ("depression",)),
    LabelRule("clinical_condition", "depression", "depression", ("depression",)),
    LabelRule("disease_state", "autism", "autism", ("autism", "asd")),
    LabelRule("clinical_condition", "autism", "autism", ("autism", "asd")),
    LabelRule("disease_state", "alzheimers", "Alzheimer's disease", ("alzheimer",)),
    LabelRule("clinical_condition", "alzheimers", "Alzheimer's disease", ("alzheimer",)),
)

DATASET_LABEL_FIELDS = (
    "species",
    "modalities",
    "brain_regions",
    "tasks",
    "behavioral_events",
    "analysis_goals",
    "data_standards",
    "file_formats",
)
EXTRA_LABEL_TYPES = (
    "modeling_method",
    "disease_state",
    "clinical_condition",
    "stimulus_type",
    "recording_context",
    "subject_state",
)
ALL_LABEL_TYPES = (*DATASET_LABEL_FIELDS, *EXTRA_LABEL_TYPES)


def _field_name_for_label_type(label_type: str) -> str | None:
    if label_type in DATASET_LABEL_FIELDS:
        return label_type
    return None


def _aliases_for_existing(label: EvidenceLabel) -> set[str]:
    return {
        label.id,
        label.label,
        label.id.removeprefix(f"label:{label.label_type}:"),
    }


def _source_items(record: NormalizedDatasetRecord | NormalizedPaperRecord) -> list[tuple[str, str, float]]:
    items: list[tuple[str, str, float]] = []
    for field, confidence in (("source", 0.95), ("source_id", 0.95), ("title", 0.78)):
        value = getattr(record, field, None)
        if value:
            items.append((field, str(value), confidence))
    if isinstance(record, NormalizedDatasetRecord):
        if record.description:
            items.append(("description", record.description, 0.68))
        if record.url:
            items.append(("url", record.url, 0.62))
        if record.raw_payload_path:
            items.append(("raw_payload_path", record.raw_payload_path, 0.62))
    else:
        if record.abstract:
            items.append(("abstract", record.abstract, 0.68))
        if record.doi:
            items.append(("doi", record.doi, 0.62))
        if record.url:
            items.append(("url", record.url, 0.62))
        if record.raw_payload_path:
            items.append(("raw_payload_path", record.raw_payload_path, 0.62))
    return items


def _pattern(term: str) -> re.Pattern[str]:
    escaped = re.escape(term.casefold())
    prefix = r"(?<![A-Za-z0-9])" if term[:1].isalnum() else ""
    suffix = r"(?![A-Za-z0-9])" if term[-1:].isalnum() else ""
    return re.compile(prefix + escaped + suffix)


def _find_evidence(text: str, term: str) -> str | None:
    match = _pattern(term).search(text.casefold())
    if not match:
        return None
    start = max(match.start() - 40, 0)
    end = min(match.end() + 60, len(text))
    return text[start:end].strip()


def _label_from_rule(
    rule: LabelRule,
    term: str,
    evidence_text: str,
    source_field: str,
    source_value: str,
    confidence: float,
) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(rule.label_type, rule.label_id),
        label=rule.label,
        label_type=rule.label_type,
        confidence=confidence,
        evidence_text=evidence_text or term,
        source_field=source_field,
        source_value=source_value,
        extractor_name=EXTRACTOR_NAME,
        extractor_version=EXTRACTOR_VERSION,
    )


def _merge(labels: Iterable[EvidenceLabel]) -> list[EvidenceLabel]:
    by_key: dict[tuple[str, str], EvidenceLabel] = {}
    for label in labels:
        key = (label.label_type, label.id)
        if key not in by_key or label.confidence > by_key[key].confidence:
            by_key[key] = label
    return sorted(by_key.values(), key=lambda item: (item.label_type, item.id))


def extract_scientific_labels(
    record: NormalizedDatasetRecord | NormalizedPaperRecord,
    *,
    include_existing: bool = True,
) -> list[EvidenceLabel]:
    """Extract conservative evidence-backed labels from a normalized record."""

    labels: list[EvidenceLabel] = []
    if include_existing:
        if isinstance(record, NormalizedPaperRecord):
            labels.extend(record.extracted_labels)
        else:
            for field in DATASET_LABEL_FIELDS:
                labels.extend(getattr(record, field, []))

    for source_field, source_value, base_confidence in _source_items(record):
        for rule in RULES:
            for term in sorted(rule.terms, key=len, reverse=True):
                evidence = _find_evidence(source_value, term)
                if evidence is None:
                    continue
                confidence = base_confidence
                if source_field in {"source", "source_id"}:
                    confidence = max(confidence, 0.95)
                labels.append(
                    _label_from_rule(
                        rule,
                        term,
                        evidence,
                        source_field,
                        source_value,
                        confidence,
                    )
                )
                break
    return _merge(labels)


def labels_by_type(labels: Iterable[EvidenceLabel]) -> dict[str, list[EvidenceLabel]]:
    grouped = {label_type: [] for label_type in ALL_LABEL_TYPES}
    for label in _merge(labels):
        grouped.setdefault(label.label_type, []).append(label)
    return grouped


def enrich_record_with_scientific_labels(
    record: NormalizedDatasetRecord | NormalizedPaperRecord,
) -> NormalizedDatasetRecord | NormalizedPaperRecord:
    """Return a copy of a normalized record with extracted labels attached."""

    labels = extract_scientific_labels(record)
    if isinstance(record, NormalizedPaperRecord):
        return record.model_copy(update={"extracted_labels": labels}, deep=True)

    grouped = labels_by_type(labels)
    updates: dict[str, Any] = {
        field: grouped.get(field, [])
        for field in DATASET_LABEL_FIELDS
    }
    # Extra label families are preserved in analysis_goals so downstream reports
    # can see them without changing the normalized dataset schema shape.
    extras = [
        label
        for label_type in EXTRA_LABEL_TYPES
        for label in grouped.get(label_type, [])
    ]
    updates["analysis_goals"] = _merge([*updates.get("analysis_goals", []), *extras])
    return record.model_copy(update=updates, deep=True)


def label_ids(labels: Iterable[EvidenceLabel]) -> set[str]:
    """Return canonical and compatibility aliases for label checks."""

    values: set[str] = set()
    for label in labels:
        values.update(value.casefold() for value in _aliases_for_existing(label) if value)
    return values
