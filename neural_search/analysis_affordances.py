"""Rule-based analysis affordance detection for normalized datasets."""

from __future__ import annotations

from collections.abc import Iterable

from neural_search.schemas import (
    AnalysisAffordance,
    EvidenceLabel,
    NormalizedDatasetRecord,
)
from neural_search.scientific_labels import label_ids

DETECTOR_NAME = "rule_based_analysis_affordance_detector"
DETECTOR_VERSION = "v0.3.0"

AFFORDANCE_IDS = (
    "event_aligned_activity",
    "trial_averaged_response",
    "choice_decoding",
    "motor_decoding",
    "speech_decoding",
    "q_learning_modeling",
    "state_space_modeling",
    "cross_modal_prediction",
    "brain_behavior_alignment",
    "seizure_detection",
    "sleep_stage_classification",
    "fmri_glm_analysis",
    "functional_connectivity",
    "representational_similarity_analysis",
    "encoding_modeling",
    "bci_decoding",
    "latent_dynamics_modeling",
)

NEURAL_MODALITIES = {
    "calcium_imaging",
    "electrophysiology",
    "neuropixels",
    "lfp",
    "eeg",
    "ecog",
    "fmri",
    "fiber_photometry",
}

# Genomic modalities for transcriptomics and epigenomics
GENOMIC_MODALITIES = {
    "single_cell_rnaseq",
    "single_nucleus_rnaseq",
    "bulk_rnaseq",
    "spatial_transcriptomics",
    "merfish",
    "visium",
    "slide_seq",
    "single_cell_atacseq",
    "single_nucleus_atacseq",
    "multiome",
    "chip_seq",
    "methylation",
    "patch_seq",
    # Title case variants for matching
    "Single Cell Rnaseq",
    "Single Nucleus Rnaseq",
    "Spatial Transcriptomics",
    "Merfish",
    "Single Nucleus Atacseq",
    "Multiome",
}
EVENTS = {
    "stimulus_onset",
    "reward",
    "choice",
    "response",
    "trial_start",
    "trial_end",
    "movement_onset",
    "reach_onset",
    "speech_onset",
    "seizure_onset",
    "sleep_stage",
    "trial_outcome",
}


def _record_labels(record: NormalizedDatasetRecord) -> list[EvidenceLabel]:
    labels: list[EvidenceLabel] = []
    for field in (
        "species",
        "modalities",
        "brain_regions",
        "tasks",
        "behavioral_events",
        "analysis_goals",
        "data_standards",
        "file_formats",
    ):
        labels.extend(getattr(record, field, []))
    return labels


def _ids(labels: Iterable[EvidenceLabel]) -> set[str]:
    values = label_ids(labels)
    return {value.removeprefix("label:").split(":")[-1] for value in values} | values


def _by_type(record: NormalizedDatasetRecord, label_type: str) -> set[str]:
    return _ids(label for label in _record_labels(record) if label.label_type == label_type)


def _text(record: NormalizedDatasetRecord) -> str:
    return " ".join(
        str(value)
        for value in [record.title, record.description, record.url, record.raw_payload_path]
        if value
    ).casefold()


def _has_any(values: set[str], options: set[str]) -> bool:
    return bool(values & options)


def _flag(record: NormalizedDatasetRecord, name: str) -> bool | None:
    return getattr(record.usability_flags, name)


def _neural_data(record: NormalizedDatasetRecord, modalities: set[str]) -> bool:
    return _flag(record, "has_neural_data") is True or _has_any(modalities, NEURAL_MODALITIES)


def _behavior_data(record: NormalizedDatasetRecord, modalities: set[str], events: set[str]) -> bool:
    return (
        _flag(record, "has_behavior") is True
        or "behavior_tracking" in modalities
        or bool(events)
    )


def _continuous_behavior(record: NormalizedDatasetRecord, text: str) -> bool:
    return _flag(record, "has_continuous_behavior") is True or any(
        term in text for term in ("kinematic", "trajectory", "position", "pose")
    )


def _affordance(
    analysis_id: str,
    support_level: str,
    confidence: float,
    *,
    present: Iterable[str] = (),
    helpful: Iterable[str] = (),
    missing: Iterable[str] = (),
    evidence: Iterable[str] = (),
) -> AnalysisAffordance:
    return AnalysisAffordance(
        analysis_id=analysis_id,
        support_level=support_level,  # type: ignore[arg-type]
        confidence=confidence,
        required_fields_present=sorted(set(present)),
        helpful_fields_present=sorted(set(helpful)),
        missing_fields=sorted(set(missing)),
        evidence=sorted(set(evidence)),
        detector_name=DETECTOR_NAME,
        detector_version=DETECTOR_VERSION,
    )


def detect_analysis_affordances(record: NormalizedDatasetRecord) -> list[AnalysisAffordance]:
    """Estimate supported analyses from normalized labels and usability flags."""

    modalities = _by_type(record, "modality")
    tasks = _by_type(record, "task")
    events = _by_type(record, "behavioral_event")
    analyses = _by_type(record, "analysis_goal")
    regions = _by_type(record, "brain_region")
    disease = _by_type(record, "disease_state") | _by_type(record, "clinical_condition")
    stimuli = _by_type(record, "stimulus_type")
    subject_states = _by_type(record, "subject_state")
    text = _text(record)

    has_neural = _neural_data(record, modalities)
    has_behavior = _behavior_data(record, modalities, events)
    has_events = _flag(record, "has_event_timestamps") is True or bool(events & EVENTS)
    has_trials = _flag(record, "has_trials") is True
    continuous_behavior = _continuous_behavior(record, text)
    multiple_regions = len(regions) >= 2
    neural_evidence = sorted(modalities & NEURAL_MODALITIES) or (["has_neural_data"] if has_neural else [])
    results: list[AnalysisAffordance] = []

    if has_neural and _flag(record, "has_event_timestamps") is True:
        results.append(_affordance("event_aligned_activity", "high", 0.9, present=["neural_data", "event_timestamps"], evidence=neural_evidence + ["event_timestamps"]))
    elif has_neural and bool(events & {"stimulus_onset", "reward", "choice", "response"}):
        results.append(_affordance("event_aligned_activity", "medium", 0.72, present=["neural_data"], helpful=sorted(events & EVENTS), missing=["event_timestamps"], evidence=neural_evidence + sorted(events & EVENTS)))
    elif has_neural:
        results.append(_affordance("event_aligned_activity", "low", 0.42, present=["neural_data"], missing=["event_timestamps"], evidence=neural_evidence))
    else:
        results.append(_affordance("event_aligned_activity", "unsupported", 0.05, missing=["neural_data", "event_timestamps"]))

    if has_trials and has_neural:
        results.append(_affordance("trial_averaged_response", "high", 0.88, present=["trials", "neural_data"], evidence=neural_evidence + ["trials"]))
    elif has_neural and tasks:
        results.append(_affordance("trial_averaged_response", "medium", 0.68, present=["neural_data"], helpful=sorted(tasks), missing=["trials"], evidence=neural_evidence + sorted(tasks)))
    elif has_neural:
        results.append(_affordance("trial_averaged_response", "low", 0.4, present=["neural_data"], missing=["trials"], evidence=neural_evidence))
    else:
        results.append(_affordance("trial_averaged_response", "unsupported", 0.05, missing=["neural_data", "trials"]))

    if has_neural and bool(events & {"choice", "response"}):
        results.append(_affordance("choice_decoding", "high", 0.87, present=["neural_data", "choice_or_response"], evidence=neural_evidence + sorted(events & {"choice", "response"})))
    elif has_neural and has_behavior and bool(tasks & {"go_nogo", "visual_decision_making", "delay_discounting", "reversal_learning", "reward_learning"}):
        results.append(_affordance("choice_decoding", "medium", 0.7, present=["neural_data", "behavior"], helpful=sorted(tasks), missing=["choice_or_response"], evidence=neural_evidence + sorted(tasks)))
    elif bool(tasks & {"go_nogo", "visual_decision_making", "delay_discounting", "reversal_learning"}):
        results.append(_affordance("choice_decoding", "low", 0.38, helpful=sorted(tasks), missing=["neural_data", "choice_or_response"], evidence=sorted(tasks)))
    else:
        results.append(_affordance("choice_decoding", "unsupported", 0.05, missing=["neural_data", "choice_or_response"]))

    motor_behavior = continuous_behavior or bool(events & {"movement_onset", "reach_onset"})
    if has_neural and "behavior_tracking" in modalities and motor_behavior:
        results.append(_affordance("motor_decoding", "high", 0.88, present=["neural_data", "behavior_tracking", "movement_or_kinematics"], evidence=neural_evidence + sorted(events & {"movement_onset", "reach_onset"})))
    elif has_neural and "behavior_tracking" in modalities and (("motor_cortex" in regions) or ("reaching" in tasks)):
        results.append(_affordance("motor_decoding", "medium", 0.68, present=["neural_data", "behavior_tracking"], helpful=sorted((regions & {"motor_cortex"}) | (tasks & {"reaching"})), missing=["movement_or_kinematics"], evidence=neural_evidence))
    elif "motor_cortex" in regions:
        results.append(_affordance("motor_decoding", "low", 0.3, helpful=["motor_cortex"], missing=["behavior_tracking", "movement_or_kinematics"], evidence=["motor_cortex"]))
    else:
        results.append(_affordance("motor_decoding", "unsupported", 0.05, missing=["neural_data", "movement_or_kinematics"]))

    speech_neural = bool(modalities & {"ecog", "eeg", "electrophysiology", "lfp"})
    if speech_neural and bool(events & {"speech_onset"} | tasks & {"speech_production"}):
        results.append(_affordance("speech_decoding", "high", 0.86, present=["speech_neural_modality", "speech_labels"], evidence=sorted(modalities & {"ecog", "eeg", "electrophysiology", "lfp"}) + sorted(events & {"speech_onset"} | tasks & {"speech_production"})))
    elif has_neural and "speech_production" in tasks:
        results.append(_affordance("speech_decoding", "medium", 0.68, present=["neural_data"], helpful=["speech_production"], missing=["speech_events"], evidence=neural_evidence + ["speech_production"]))
    elif "auditory_cortex" in regions or "auditory_task" in tasks:
        results.append(_affordance("speech_decoding", "low", 0.25, helpful=sorted((regions & {"auditory_cortex"}) | (tasks & {"auditory_task"})), missing=["speech_events", "speech_task"], evidence=sorted((regions & {"auditory_cortex"}) | (tasks & {"auditory_task"}))))
    else:
        results.append(_affordance("speech_decoding", "unsupported", 0.05, missing=["speech_events", "neural_data"]))

    q_requirements = {"choice", "reward", "trial_outcome"}
    if q_requirements <= events:
        results.append(_affordance("q_learning_modeling", "high", 0.9, present=sorted(q_requirements), evidence=sorted(q_requirements)))
    elif bool(tasks & {"reversal_learning", "reward_learning"}) and {"choice", "reward"} <= events:
        results.append(_affordance("q_learning_modeling", "medium", 0.72, present=["choice", "reward"], helpful=sorted(tasks & {"reversal_learning", "reward_learning"}), missing=["trial_outcome"], evidence=sorted(events & {"choice", "reward"}) + sorted(tasks)))
    elif "reward" in events:
        results.append(_affordance("q_learning_modeling", "low", 0.28, present=["reward"], missing=["choice", "trial_outcome"], evidence=["reward"]))
    else:
        results.append(_affordance("q_learning_modeling", "unsupported", 0.05, missing=sorted(q_requirements)))

    if has_neural and (has_trials or has_events) and bool(modalities & {"electrophysiology", "neuropixels", "lfp", "calcium_imaging"}):
        results.append(_affordance("state_space_modeling", "high", 0.84, present=["neural_timeseries", "trials_or_events"], evidence=neural_evidence))
    elif has_neural:
        results.append(_affordance("state_space_modeling", "medium", 0.64, present=["neural_data"], missing=["trials_or_events"], evidence=neural_evidence))
    else:
        results.append(_affordance("state_space_modeling", "low", 0.2, missing=["neural_timeseries"], evidence=[]))

    if has_neural and ("behavior_tracking" in modalities or stimuli):
        results.append(_affordance("cross_modal_prediction", "high", 0.82, present=["neural_data", "second_modality"], evidence=neural_evidence + sorted(stimuli or {"behavior_tracking"})))
    elif len(modalities) >= 2:
        results.append(_affordance("cross_modal_prediction", "medium", 0.62, present=["multiple_modalities"], missing=["alignment_evidence"], evidence=sorted(modalities)))
    else:
        results.append(_affordance("cross_modal_prediction", "low", 0.25, missing=["second_modality"], evidence=sorted(modalities)))

    if has_neural and _flag(record, "has_behavior") is True:
        results.append(_affordance("brain_behavior_alignment", "high", 0.88, present=["neural_data", "behavior"], evidence=neural_evidence + ["has_behavior"]))
    elif has_neural and ("behavior_tracking" in modalities or events):
        results.append(_affordance("brain_behavior_alignment", "medium", 0.7, present=["neural_data"], helpful=sorted(events | (modalities & {"behavior_tracking"})), evidence=neural_evidence + sorted(events)))
    elif has_neural:
        results.append(_affordance("brain_behavior_alignment", "low", 0.35, present=["neural_data"], missing=["behavior"], evidence=neural_evidence))
    else:
        results.append(_affordance("brain_behavior_alignment", "unsupported", 0.05, missing=["neural_data", "behavior"]))

    if has_neural and ("seizure_monitoring" in tasks or "epilepsy" in disease or "seizure_detection" in analyses):
        results.append(_affordance("seizure_detection", "high", 0.86, present=["neural_data", "seizure_or_epilepsy_label"], evidence=neural_evidence + sorted(tasks & {"seizure_monitoring"} | disease & {"epilepsy"} | analyses & {"seizure_detection"})))
    elif "epilepsy" in disease or "seizure_monitoring" in tasks:
        results.append(_affordance("seizure_detection", "medium", 0.58, present=["seizure_or_epilepsy_label"], missing=["neural_data"], evidence=sorted(disease | tasks)))
    elif has_neural:
        results.append(_affordance("seizure_detection", "low", 0.15, present=["neural_data"], missing=["seizure_or_epilepsy_label"], evidence=neural_evidence))
    else:
        results.append(_affordance("seizure_detection", "unsupported", 0.05, missing=["neural_data", "seizure_or_epilepsy_label"]))

    if has_neural and ("sleep_stage" in events or "sleep_staging" in tasks):
        results.append(_affordance("sleep_stage_classification", "high", 0.86, present=["neural_data", "sleep_stage_labels"], evidence=neural_evidence + sorted(events & {"sleep_stage"} | tasks & {"sleep_staging"})))
    elif has_neural and "sleep" in subject_states:
        results.append(_affordance("sleep_stage_classification", "medium", 0.62, present=["neural_data", "sleep_state"], missing=["sleep_stage_labels"], evidence=neural_evidence + ["sleep"]))
    elif "resting_state" in subject_states:
        results.append(_affordance("sleep_stage_classification", "low", 0.12, helpful=["resting_state"], missing=["sleep_stage_labels"], evidence=["resting_state"]))
    else:
        results.append(_affordance("sleep_stage_classification", "unsupported", 0.05, missing=["sleep_stage_labels", "neural_data"]))

    standards = _by_type(record, "data_standard")
    file_formats = _by_type(record, "file_format")
    has_bids = "bids" in standards
    normalized_tasks = {
        task.removeprefix("label:").split(":")[-1].replace(" ", "_").casefold()
        for task in tasks
    }
    has_task_fmri = "fmri" in modalities and bool(normalized_tasks - {"resting_state"})
    has_glm_events = has_events or "events.tsv" in text or "events_tsv" in file_formats
    has_glm_derivatives = any(
        term in text
        for term in (
            "glm",
            "general linear model",
            "first level",
            "first-level",
            "contrast",
            "design matrix",
            "beta map",
            "statistical map",
            "derivative",
            "derivatives",
        )
    )
    if "fmri" in modalities and has_bids and has_task_fmri and (has_glm_events or has_glm_derivatives):
        results.append(_affordance("fmri_glm_analysis", "high", 0.88, present=["fmri", "bids", "task_or_conditions"], helpful=["events_or_derivatives"], evidence=["fmri", "bids"] + sorted(tasks)))
    elif "fmri" in modalities and has_bids and has_task_fmri:
        results.append(_affordance("fmri_glm_analysis", "medium", 0.68, present=["fmri", "bids"], helpful=sorted(tasks), missing=["events_or_contrast_metadata"], evidence=["fmri", "bids"] + sorted(tasks)))
    elif "fmri" in modalities:
        results.append(_affordance("fmri_glm_analysis", "low", 0.28, present=["fmri"], missing=["bids", "task_events_or_contrasts"], evidence=["fmri"]))
    else:
        results.append(_affordance("fmri_glm_analysis", "unsupported", 0.05, missing=["fmri", "bids", "task_events_or_contrasts"]))

    if (modalities & {"fmri", "eeg", "ecog", "lfp", "electrophysiology"}) and ("functional_connectivity" in analyses or multiple_regions):
        results.append(_affordance("functional_connectivity", "high", 0.82, present=["connectivity_suitable_modality"], helpful=sorted(analyses & {"functional_connectivity"} | regions), evidence=sorted(modalities & {"fmri", "eeg", "ecog", "lfp", "electrophysiology"})))
    elif has_neural and multiple_regions:
        results.append(_affordance("functional_connectivity", "medium", 0.62, present=["neural_data", "multiple_regions"], evidence=neural_evidence + sorted(regions)))
    elif has_neural:
        results.append(_affordance("functional_connectivity", "low", 0.28, present=["neural_data"], missing=["multiple_regions"], evidence=neural_evidence))
    else:
        results.append(_affordance("functional_connectivity", "unsupported", 0.05, missing=["neural_data", "multiple_regions"]))

    if has_neural and (stimuli or tasks):
        results.append(_affordance("representational_similarity_analysis", "high", 0.78, present=["neural_data", "conditions"], evidence=neural_evidence + sorted(stimuli | tasks)))
    elif has_neural:
        results.append(_affordance("representational_similarity_analysis", "low", 0.34, present=["neural_data"], missing=["stimulus_or_task_conditions"], evidence=neural_evidence))
    else:
        results.append(_affordance("representational_similarity_analysis", "unsupported", 0.05, missing=["neural_data", "conditions"]))

    if has_neural and (stimuli or has_behavior):
        results.append(_affordance("encoding_modeling", "high", 0.82, present=["neural_data", "stimulus_or_behavior_covariates"], evidence=neural_evidence + sorted(stimuli)))
    elif has_neural and (tasks or stimuli):
        results.append(_affordance("encoding_modeling", "medium", 0.62, present=["neural_data"], helpful=sorted(tasks | stimuli), missing=["explicit_covariates"], evidence=neural_evidence + sorted(tasks | stimuli)))
    elif has_neural:
        results.append(_affordance("encoding_modeling", "low", 0.3, present=["neural_data"], missing=["stimulus_or_behavior_covariates"], evidence=neural_evidence))
    else:
        results.append(_affordance("encoding_modeling", "unsupported", 0.05, missing=["neural_data", "covariates"]))

    bci_context = "bci_decoding" in analyses or "bci" in text or "brain computer interface" in text
    bci_labels = bool(tasks & {"motor_imagery", "speech_production"} or events & {"response", "speech_onset", "movement_onset", "reach_onset"})
    if has_neural and bci_context and bci_labels:
        results.append(_affordance("bci_decoding", "high", 0.86, present=["neural_data", "bci_context", "behavior_or_task_labels"], evidence=neural_evidence + sorted(tasks | events)))
    elif has_neural and bool(modalities & {"eeg", "ecog"}) and bool(tasks & {"motor_imagery", "speech_production"}):
        results.append(_affordance("bci_decoding", "medium", 0.66, present=["neural_data"], helpful=sorted(tasks & {"motor_imagery", "speech_production"}), missing=["explicit_bci_context"], evidence=neural_evidence + sorted(tasks)))
    elif modalities == {"eeg"} or ("eeg" in modalities and not bci_labels):
        results.append(_affordance("bci_decoding", "low", 0.18, helpful=["eeg"], missing=["bci_context", "behavior_or_task_labels"], evidence=["eeg"]))
    else:
        results.append(_affordance("bci_decoding", "unsupported", 0.05, missing=["neural_data", "bci_context"]))

    if has_neural and (has_trials or has_events):
        results.append(_affordance("latent_dynamics_modeling", "high", 0.82, present=["neural_timeseries", "trials_or_events"], evidence=neural_evidence))
    elif has_neural:
        results.append(_affordance("latent_dynamics_modeling", "medium", 0.6, present=["neural_data"], missing=["trials_or_events"], evidence=neural_evidence))
    else:
        results.append(_affordance("latent_dynamics_modeling", "low", 0.2, missing=["neural_timeseries"], evidence=[]))

    return sorted(results, key=lambda item: AFFORDANCE_IDS.index(item.analysis_id))


def enrich_record_with_affordances(record: NormalizedDatasetRecord) -> NormalizedDatasetRecord:
    """Return a dataset record with analysis affordances attached."""

    return record.model_copy(
        update={"analysis_affordances": detect_analysis_affordances(record)},
        deep=True,
    )
