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


# ============================================================================
# NWB-based Feature Extraction (scaffolded for future implementation)
# ============================================================================


def extract_features_from_nwb_path(nwb_path: str) -> SessionFeatures | None:
    """Extract latent features from an NWB file.

    This function extracts neural and behavioral features from NWB files
    for neural-neural similarity search.

    Args:
        nwb_path: Path to NWB file

    Returns:
        SessionFeatures if extraction succeeds, None otherwise

    Note:
        Requires pynwb to be installed. Falls back to metadata-only
        extraction if NWB reading fails.
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError:
        return None

    features: list[FeatureSummary] = []
    warnings: list[str] = []

    try:
        with NWBHDF5IO(nwb_path, "r") as io:
            nwbfile = io.read()

            # Extract basic metadata
            session_id = nwbfile.session_id or "unknown"
            dataset_id = nwbfile.identifier or nwb_path

            # Extract neural statistics
            neural_stats = _extract_nwb_neural_stats(nwbfile)
            if neural_stats:
                features.append(neural_stats)
            else:
                warnings.append("Could not extract neural statistics from NWB")

            # Extract behavioral statistics
            behavior_stats = _extract_nwb_behavior_stats(nwbfile)
            if behavior_stats:
                features.append(behavior_stats)
            else:
                warnings.append("No behavioral data found in NWB")

            # Extract trial statistics
            trial_stats = _extract_nwb_trial_stats(nwbfile)
            if trial_stats:
                features.append(trial_stats)
            else:
                warnings.append("No trial data found in NWB")

            # Extract electrode information
            electrode_stats = _extract_nwb_electrode_stats(nwbfile)
            if electrode_stats:
                features.append(electrode_stats)

            return SessionFeatures(
                dataset_id=dataset_id,
                session_id=session_id,
                features=features,
                warnings=warnings,
            )

    except Exception as e:
        return SessionFeatures(
            dataset_id=nwb_path,
            session_id="unknown",
            features=[],
            warnings=[f"NWB extraction failed: {e}"],
        )


def _extract_nwb_neural_stats(nwbfile: Any) -> FeatureSummary | None:
    """Extract neural recording statistics from NWB file."""
    values = []
    metadata = {}

    # Try to get units (sorted spikes)
    units = getattr(nwbfile, "units", None)
    if units is not None:
        try:
            unit_count = len(units)
            values.append(_scaled_float(unit_count, 1000))
            metadata["unit_count"] = unit_count

            # Calculate mean firing rate if spike times available
            if "spike_times" in units.colnames:
                total_spikes = sum(len(units["spike_times"][i]) for i in range(unit_count))
                # Estimate session duration
                all_spikes = []
                for i in range(min(unit_count, 10)):  # Sample first 10 units
                    all_spikes.extend(units["spike_times"][i])
                if all_spikes:
                    duration = max(all_spikes) - min(all_spikes)
                    if duration > 0:
                        mean_fr = (total_spikes / unit_count) / duration if unit_count > 0 else 0
                        values.append(_scaled_float(mean_fr, 50))
                        metadata["mean_firing_rate_hz"] = mean_fr
        except Exception:
            pass

    # Try to get acquisition data
    acquisition = getattr(nwbfile, "acquisition", {})
    for _name, data in acquisition.items():
        if hasattr(data, "rate"):
            values.append(_scaled_float(data.rate, 30000))
            metadata["sampling_rate_hz"] = data.rate
            break

    if not values:
        return None

    return FeatureSummary(
        feature_type=FeatureType.NEURAL_SUMMARY_STATISTICS,
        dimensions=len(values),
        values=values,
        metadata=metadata,
    )


def _extract_nwb_behavior_stats(nwbfile: Any) -> FeatureSummary | None:
    """Extract behavioral statistics from NWB file."""
    values = []
    metadata = {}

    # Check for behavioral time series
    behavior = getattr(nwbfile, "processing", {}).get("behavior", None)
    if behavior is None:
        return None

    behavior_types = []
    for name, data in behavior.data_interfaces.items():
        behavior_types.append(name.lower())
        if hasattr(data, "data"):
            try:
                data_len = len(data.data)
                values.append(_scaled_float(data_len, 100000))
            except Exception:
                pass

    if behavior_types:
        # Create histogram over common behavior types
        vocab = ["position", "speed", "velocity", "pupil", "lick", "wheel", "running"]
        hist_values = normalized_histogram(behavior_types, vocab)
        values.extend(hist_values)
        metadata["behavior_types"] = behavior_types

    if not values:
        return None

    return FeatureSummary(
        feature_type=FeatureType.BEHAVIOR_TRANSITION_SUMMARY,
        dimensions=len(values),
        values=values,
        metadata=metadata,
    )


def _extract_nwb_trial_stats(nwbfile: Any) -> FeatureSummary | None:
    """Extract trial statistics from NWB file."""
    trials = getattr(nwbfile, "trials", None)
    if trials is None:
        return None

    values = []
    metadata = {}

    try:
        trial_count = len(trials)
        values.append(_scaled_float(trial_count, 1000))
        metadata["trial_count"] = trial_count

        # Trial duration statistics
        if "start_time" in trials.colnames and "stop_time" in trials.colnames:
            durations = [
                trials["stop_time"][i] - trials["start_time"][i]
                for i in range(min(trial_count, 100))
            ]
            if durations:
                mean_duration = sum(durations) / len(durations)
                values.append(_scaled_float(mean_duration, 30))  # 30 second max
                metadata["mean_trial_duration_s"] = mean_duration

        # Check for common trial columns
        common_columns = ["correct", "choice", "stimulus", "response_time"]
        for col in common_columns:
            has_col = 1.0 if col in trials.colnames else 0.0
            values.append(has_col)
            metadata[f"has_{col}"] = bool(has_col)

    except Exception:
        return None

    if not values:
        return None

    return FeatureSummary(
        feature_type=FeatureType.TASK_STATE_LABELS,
        dimensions=len(values),
        values=values,
        metadata=metadata,
    )


def _extract_nwb_electrode_stats(nwbfile: Any) -> FeatureSummary | None:
    """Extract electrode/recording location statistics from NWB file."""
    electrodes = getattr(nwbfile, "electrodes", None)
    if electrodes is None:
        return None

    values = []
    metadata = {}

    try:
        electrode_count = len(electrodes)
        values.append(_scaled_float(electrode_count, 1000))
        metadata["electrode_count"] = electrode_count

        # Extract brain regions
        if "location" in electrodes.colnames:
            regions = set()
            for i in range(electrode_count):
                loc = electrodes["location"][i]
                if loc:
                    regions.add(str(loc).lower())
            metadata["brain_regions"] = list(regions)
            values.append(_scaled_float(len(regions), 20))

    except Exception:
        return None

    if not values:
        return None

    return FeatureSummary(
        feature_type=FeatureType.NEURAL_SUMMARY_STATISTICS,
        dimensions=len(values),
        values=values,
        metadata=metadata,
    )
