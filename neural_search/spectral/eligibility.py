"""Conservative, metadata-only eligibility detection for aperiodic spectral
parameterization (FOOOF/specparam/IRASA-style 1/f reanalysis).

This module never inspects raw signal data — it only reasons about the
normalized labels and usability flags already attached to a
``NormalizedDatasetRecord``. Computing or validating an actual fit requires
``neural_search.spectral.psd`` and a backend from
``neural_search.spectral.specparam_backend`` / ``irasa_backend``.
"""

from __future__ import annotations

from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord
from neural_search.scientific_labels import label_ids
from neural_search.spectral.schemas import (
    DEFAULT_DETECTOR_NAME,
    DEFAULT_DETECTOR_VERSION,
    AperiodicEligibility,
)

# Modalities with sufficient temporal resolution to support aperiodic /
# periodic spectral parameterization of continuous neural activity. Includes
# common raw/display synonyms (not just the canonical taxonomy ids used by
# neural_search.analysis_affordances.NEURAL_MODALITIES) since metadata in
# the wild uses many spellings for the same recording technique.
APERIODIC_COMPATIBLE_MODALITIES = {
    "electrophysiology",
    "extracellular_ephys",
    "intracellular_ephys",
    "eeg",
    "scalp_eeg",
    "ecog",
    "ieeg",
    "lfp",
    "neuropixels",
    "meg",
    "intracranial_eeg",
    "seeg",
    "utah_array",
    "tetrode",
}

# Modalities that explicitly rule out aperiodic spectral reanalysis even
# though they may co-occur with otherwise-plausible metadata.
FMRI_MODALITIES = {"fmri"}
BEHAVIOR_ONLY_MODALITIES = {"behavior_tracking"}
ANATOMICAL_MODALITIES = {"structural_mri", "histology", "anatomy", "diffusion_mri", "dti"}

SAMPLING_RATE_HINT_STANDARDS = {"nwb", "bids"}
SAMPLING_RATE_HINT_FORMATS = {"edf", "brainvision", "nev", "ns5", "nwb"}
CHANNEL_METADATA_HINT_STANDARDS = {"nwb"}
CHANNEL_METADATA_HINT_TERMS = ("channel", "electrode", "probe")
MISSING_SAMPLING_RATE_TERMS = ("sampling_rate", "sample_rate", "sampling rate")
MISSING_CHANNEL_TERMS = ("channel", "electrode", "probe")


def _ids(labels: list[EvidenceLabel]) -> set[str]:
    values = label_ids(labels)
    return {value.removeprefix("label:").split(":")[-1] for value in values} | values


def _all_labels(record: NormalizedDatasetRecord) -> list[EvidenceLabel]:
    labels: list[EvidenceLabel] = []
    for field in ("species", "modalities", "brain_regions", "tasks", "data_standards", "file_formats"):
        labels.extend(getattr(record, field, []))
    return labels


def _by_type(record: NormalizedDatasetRecord, label_type: str) -> set[str]:
    return _ids([label for label in _all_labels(record) if label.label_type == label_type])


def _flag(record: NormalizedDatasetRecord, name: str) -> bool | None:
    return getattr(record.usability_flags, name)


def _missing_lower(record: NormalizedDatasetRecord) -> str:
    return " ".join(record.missing_fields).casefold()


def _has_terms(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def detect_aperiodic_eligibility(record: NormalizedDatasetRecord) -> AperiodicEligibility:
    """Estimate, from metadata alone, whether a dataset can support
    aperiodic spectral parameterization (FOOOF/specparam/IRASA)."""

    modalities = _by_type(record, "modality")
    standards = _by_type(record, "data_standard")
    file_formats = _by_type(record, "file_format")
    missing = _missing_lower(record)

    # `_ids` returns both compact ids ("fmri") and full label ids
    # ("label:modality:fmri") for compatibility with analysis_affordances.py
    # intersection checks; subset checks below need only the compact form.
    compact_modalities = {value for value in modalities if ":" not in value}

    compatible_modality = bool(modalities & APERIODIC_COMPATIBLE_MODALITIES)
    fmri_only = bool(compact_modalities) and compact_modalities <= FMRI_MODALITIES
    behavior_only = bool(compact_modalities) and compact_modalities <= BEHAVIOR_ONLY_MODALITIES
    anatomical_only = bool(compact_modalities) and compact_modalities <= ANATOMICAL_MODALITIES
    no_modality_evidence = not modalities

    has_raw_or_continuous = _flag(record, "has_raw_data") is True or _flag(record, "has_neural_data") is True
    has_processed = _flag(record, "has_processed_data") is True
    metadata_incomplete = bool(record.missing_fields)

    sampling_rate_likely_available = (
        not _has_terms(missing, MISSING_SAMPLING_RATE_TERMS)
        and (
            has_raw_or_continuous
            or bool(standards & SAMPLING_RATE_HINT_STANDARDS)
            or bool(file_formats & SAMPLING_RATE_HINT_FORMATS)
        )
    )
    channel_metadata_present = not _has_terms(missing, MISSING_CHANNEL_TERMS) and (
        bool(standards & CHANNEL_METADATA_HINT_STANDARDS)
        or _has_terms((record.description or "").casefold(), CHANNEL_METADATA_HINT_TERMS)
    )

    evidence = sorted(modalities & APERIODIC_COMPATIBLE_MODALITIES)
    present: list[str] = []
    helpful: list[str] = []
    missing_fields: list[str] = []
    reasons: list[str] = []

    if not compatible_modality:
        if fmri_only:
            reasons.append("Only fMRI modality detected; BOLD temporal resolution cannot support aperiodic spectral parameterization.")
            return _result(record, "unsupported", 0.92, evidence=sorted(modalities), missing=["electrophysiology_or_eeg_or_meg_modality"], reasons=reasons)
        if behavior_only:
            reasons.append("Only behavior-tracking modality detected; no neural signal is present.")
            return _result(record, "unsupported", 0.92, evidence=sorted(modalities), missing=["neural_modality"], reasons=reasons)
        if anatomical_only:
            reasons.append("Only anatomical/structural modality detected; no continuous neural time series is present.")
            return _result(record, "unsupported", 0.92, evidence=sorted(modalities), missing=["continuous_neural_signal"], reasons=reasons)
        if no_modality_evidence:
            reasons.append("No modality evidence recorded; eligibility cannot be determined.")
            return _result(record, "unknown", 0.3, missing=["modality_metadata"], reasons=reasons)
        reasons.append("No electrophysiology/EEG/MEG-compatible modality detected.")
        return _result(record, "unsupported", 0.7, evidence=sorted(modalities), missing=["electrophysiology_or_eeg_or_meg_modality"], reasons=reasons)

    present.append("compatible_modality")
    if has_raw_or_continuous and sampling_rate_likely_available and channel_metadata_present:
        helpful.extend(["sampling_rate_metadata", "channel_or_probe_metadata"])
        reasons.append("Compatible modality with raw/continuous neural data, likely sampling rate, and channel/probe metadata.")
        return _result(
            record,
            "high",
            0.85,
            evidence=evidence,
            present=present + ["raw_or_continuous_neural_data"],
            helpful=helpful,
            reasons=reasons,
            compatible_modality=True,
            likely_continuous_signal=True,
            sampling_rate_likely_available=True,
            channel_or_probe_metadata_present=True,
        )

    if has_raw_or_continuous or has_processed or metadata_incomplete:
        if has_raw_or_continuous:
            present.append("raw_or_continuous_neural_data")
        if has_processed:
            present.append("processed_neural_data")
        if not sampling_rate_likely_available:
            missing_fields.append("sampling_rate")
        if not channel_metadata_present:
            missing_fields.append("channel_or_probe_metadata")
        reasons.append("Compatible modality with processed data and/or incomplete supporting metadata.")
        return _result(
            record,
            "medium",
            0.55,
            evidence=evidence,
            present=present,
            missing=missing_fields,
            reasons=reasons,
            compatible_modality=True,
            likely_continuous_signal=has_raw_or_continuous,
            sampling_rate_likely_available=sampling_rate_likely_available,
            channel_or_probe_metadata_present=channel_metadata_present,
        )

    reasons.append("Compatible modality detected but data access (raw vs. processed) is unclear.")
    return _result(
        record,
        "low",
        0.3,
        evidence=evidence,
        present=present,
        missing=["raw_or_processed_data_clarity", "sampling_rate", "channel_or_probe_metadata"],
        reasons=reasons,
        compatible_modality=True,
    )


def _result(
    record: NormalizedDatasetRecord,
    support_level: str,
    confidence: float,
    *,
    evidence: list[str] = (),
    present: list[str] = (),
    helpful: list[str] = (),
    missing: list[str] = (),
    reasons: list[str] = (),
    compatible_modality: bool = False,
    likely_continuous_signal: bool = False,
    sampling_rate_likely_available: bool = False,
    channel_or_probe_metadata_present: bool = False,
) -> AperiodicEligibility:
    return AperiodicEligibility(
        dataset_id=record.dataset_id,
        support_level=support_level,  # type: ignore[arg-type]
        confidence=confidence,
        compatible_modality=compatible_modality,
        likely_continuous_signal=likely_continuous_signal,
        sampling_rate_likely_available=sampling_rate_likely_available,
        channel_or_probe_metadata_present=channel_or_probe_metadata_present,
        required_fields_present=sorted(set(present)),
        helpful_fields_present=sorted(set(helpful)),
        missing_fields=sorted(set(missing)),
        evidence=sorted(set(evidence)),
        reasons=list(reasons),
        detector_name=DEFAULT_DETECTOR_NAME,
        detector_version=DEFAULT_DETECTOR_VERSION,
    )
