"""Deterministic feature summaries for future latent-state search."""

from __future__ import annotations

from typing import Any

from neural_search.latent.embedding_schema import (
    FeatureSummary,
    FeatureType,
    SessionFeatures,
)
from neural_search.latent.tokenization import normalized_histogram, tokenize_labels

EVENT_VOCABULARY = [
    "trial",
    "stimulus",
    "cue",
    "choice",
    "response",
    "movement",
    "reward",
    "omission",
    "error",
    "correct",
]

TASK_STATE_VOCABULARY = [
    "go_nogo",
    "reversal_learning",
    "delay_discounting",
    "reaching",
    "center_out_reaching",
    "visual_decision_making",
    "motor_imagery",
    "seizure_monitoring",
]

MODALITY_VOCABULARY = [
    "extracellular_ephys",
    "neuropixels",
    "calcium_imaging",
    "fiber_photometry",
    "ecog",
    "ieeg",
    "eeg",
    "fmri",
]


def extract_session_features(
    dataset: dict[str, Any],
    session_data: dict[str, Any] | None = None,
) -> SessionFeatures:
    """Create deterministic session summaries from metadata and optional events."""

    session_data = session_data or {}
    dataset_id = str(dataset.get("source_id") or dataset.get("id") or "unknown")
    session_id = str(session_data.get("session_id") or "session_0")

    event_labels = [
        *dataset.get("behaviors", []),
        *session_data.get("events", []),
    ]
    modalities = [str(value) for value in dataset.get("modalities", [])]
    tasks = [str(value) for value in dataset.get("tasks", [])]
    neural_stats = session_data.get("neural_stats", {})
    qc = session_data.get("qc", {})

    features = [
        event_histogram_feature(event_labels),
        neural_summary_feature(modalities, neural_stats),
        behavior_transition_feature(event_labels),
        task_state_feature(tasks),
        session_qc_feature(dataset, qc),
    ]
    warnings = []
    if not event_labels:
        warnings.append("No behavior or event labels available for event summaries.")
    if not modalities:
        warnings.append("No modalities available for neural summary features.")

    return SessionFeatures(
        dataset_id=dataset_id,
        session_id=session_id,
        features=features,
        warnings=warnings,
    )


def event_histogram_feature(event_labels: list[str]) -> FeatureSummary:
    values = normalized_histogram(tokenize_labels(event_labels), EVENT_VOCABULARY)
    return FeatureSummary(
        feature_type=FeatureType.EVENT_HISTOGRAM,
        dimensions=len(values),
        values=values,
        metadata={"vocabulary": EVENT_VOCABULARY},
    )


def neural_summary_feature(
    modalities: list[str],
    neural_stats: dict[str, Any] | None = None,
) -> FeatureSummary:
    neural_stats = neural_stats or {}
    modality_values = normalized_histogram(tokenize_labels(modalities), MODALITY_VOCABULARY)
    numeric_values = [
        _scaled_float(neural_stats.get("unit_count"), 1000),
        _scaled_float(neural_stats.get("mean_firing_rate_hz"), 50),
        _scaled_float(neural_stats.get("roi_count"), 5000),
        _scaled_float(neural_stats.get("sampling_rate_hz"), 30000),
    ]
    values = [*modality_values, *numeric_values]
    return FeatureSummary(
        feature_type=FeatureType.NEURAL_SUMMARY_STATISTICS,
        dimensions=len(values),
        values=values,
        metadata={
            "modality_vocabulary": MODALITY_VOCABULARY,
            "numeric_features": [
                "unit_count",
                "mean_firing_rate_hz",
                "roi_count",
                "sampling_rate_hz",
            ],
        },
    )


def behavior_transition_feature(event_labels: list[str]) -> FeatureSummary:
    tokens = tokenize_labels(event_labels)
    transitions = list(zip(tokens, tokens[1:], strict=False))
    transition_names = [
        "cue_to_choice",
        "choice_to_reward",
        "choice_to_error",
        "reward_to_omission",
    ]
    counts = {
        "cue_to_choice": ("cue", "choice"),
        "choice_to_reward": ("choice", "reward"),
        "choice_to_error": ("choice", "error"),
        "reward_to_omission": ("reward", "omission"),
    }
    total = max(len(transitions), 1)
    values = [
        sum(1 for transition in transitions if transition == counts[name]) / total
        for name in transition_names
    ]
    return FeatureSummary(
        feature_type=FeatureType.BEHAVIOR_TRANSITION_SUMMARY,
        dimensions=len(values),
        values=values,
        metadata={"transition_names": transition_names},
    )


def task_state_feature(tasks: list[str]) -> FeatureSummary:
    values = normalized_histogram(tokenize_labels(tasks), TASK_STATE_VOCABULARY)
    return FeatureSummary(
        feature_type=FeatureType.TASK_STATE_LABELS,
        dimensions=len(values),
        values=values,
        metadata={"vocabulary": TASK_STATE_VOCABULARY},
    )


def session_qc_feature(dataset: dict[str, Any], qc: dict[str, Any] | None = None) -> FeatureSummary:
    qc = qc or {}
    values = [
        1.0 if dataset.get("has_trials") else 0.0,
        1.0 if dataset.get("has_behavior") else 0.0,
        1.0 if dataset.get("license") else 0.0,
        1.0 if dataset.get("data_standards") else 0.0,
        _scaled_float(qc.get("artifact_ratio"), 1.0),
        _scaled_float(qc.get("metadata_completeness"), 1.0),
    ]
    return FeatureSummary(
        feature_type=FeatureType.SESSION_QC_VECTOR,
        dimensions=len(values),
        values=values,
        metadata={
            "metric_names": [
                "has_trials",
                "has_behavior",
                "has_license",
                "has_data_standard",
                "artifact_ratio",
                "metadata_completeness",
            ]
        },
    )


def _scaled_float(value: Any, denominator: float) -> float:
    if value is None:
        return 0.0
    try:
        return max(0.0, min(float(value) / denominator, 1.0))
    except (TypeError, ValueError):
        return 0.0
