"""Analysis Affordance Registry and Validation.

This module defines the requirements for each analysis affordance and
provides validation against dataset features.

An affordance represents what analyses a dataset can support. This is
the key differentiator for Neural Search - not just finding datasets
that *mention* an analysis, but finding datasets that actually *support* it.

Affordance validation answers:
> Does this dataset have the required experimental structure to support this analysis?

Not:
> Does the dataset description contain similar words?

Supported affordances:
- event_aligned_psth: Event-aligned peri-stimulus time histogram
- choice_decoding: Neural decoding of behavioral choices
- q_learning: Q-learning/reinforcement learning model fitting
- stimulus_response_modeling: Encoding model fitting
- behavioral_state_decoding: Decoding behavioral states from neural activity
- cross_area_interaction: Cross-brain-area analysis
- dimensionality_reduction: Population dimensionality analysis
- functional_connectivity: Connectivity analysis
- trial_aligned_calcium_analysis: Calcium imaging trial analysis
- pose_neural_correlation: Pose/video and neural correlation

Each affordance specifies:
- Required features that MUST be present
- Optional features that help but aren't required
- Negative conditions that rule out support
- Validation methods for different data formats
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from neural_search.core.dataset_card import (
    AffordanceRequirement,
    AffordanceValidationResult,
    ProvenanceEvidence,
)


class SupportLevel(StrEnum):
    """Level of support for an affordance."""

    HIGH = "high"        # All required features verified in structured data
    MEDIUM = "medium"    # Required features inferred from metadata
    LOW = "low"          # Only textual evidence
    UNSUPPORTED = "unsupported"  # Missing critical features
    UNKNOWN = "unknown"  # Cannot determine


class DataFormat(StrEnum):
    """Supported data formats for validation."""

    NWB = "nwb"
    BIDS = "bids"
    METADATA_ONLY = "metadata_only"


# =============================================================================
# Affordance Requirement Definitions
# =============================================================================

EVENT_ALIGNED_PSTH = AffordanceRequirement(
    affordance_id="event_aligned_psth",
    label="Event-aligned PSTH",
    description="Compute peri-stimulus time histograms aligned to behavioral events",
    required_features=[
        "spike_times_or_firing_rates",
        "event_timestamps",
        "neural_data",
    ],
    optional_features=[
        "trial_structure",
        "stimulus_identity",
        "multiple_conditions",
        "sorted_units",
    ],
    negative_conditions=[
        "only_summary_statistics",
        "no_temporal_alignment",
        "aggregate_only",
    ],
    validation_methods=["nwb_units_check", "nwb_trials_check", "bids_events_check"],
    min_trials=10,
    example_use_cases=[
        "Analyze neural response to stimulus onset",
        "Compare activity across trial conditions",
        "Measure response latency",
    ],
)

CHOICE_DECODING = AffordanceRequirement(
    affordance_id="choice_decoding",
    label="Choice decoding",
    description="Decode behavioral choices from neural population activity",
    required_features=[
        "neural_data",
        "choice_labels",
        "trial_structure",
    ],
    optional_features=[
        "response_time",
        "stimulus_identity",
        "multiple_sessions",
        "sorted_units",
        "continuous_neural_data",
    ],
    negative_conditions=[
        "no_choice_labels",
        "only_summary_statistics",
        "single_trial",
    ],
    validation_methods=["nwb_trials_column_check", "bids_events_column_check"],
    min_trials=50,
    min_subjects=1,
    example_use_cases=[
        "Predict left/right choice from neural activity",
        "Decode stimulus category from population response",
        "Identify decision-related neural signals",
    ],
)

Q_LEARNING = AffordanceRequirement(
    affordance_id="q_learning",
    label="Q-learning model fitting",
    description="Fit reinforcement learning models to behavioral data",
    required_features=[
        "trial_structure",
        "ordered_trials",
        "choice_sequence",
        "reward_signal",
        "outcome_labels",
    ],
    optional_features=[
        "reaction_time",
        "stimulus_identity",
        "block_label",
        "session_id",
        "neural_correlates",
    ],
    negative_conditions=[
        "only_summary_statistics",
        "no_trialwise_behavior",
        "shuffled_trials",
    ],
    validation_methods=["nwb_trials_column_check", "bids_events_column_check"],
    min_trials=100,
    example_use_cases=[
        "Estimate learning rate from behavior",
        "Model value representation",
        "Analyze reward prediction errors",
    ],
)

STIMULUS_RESPONSE_MODELING = AffordanceRequirement(
    affordance_id="stimulus_response_modeling",
    label="Stimulus-response modeling",
    description="Build encoding models relating stimuli to neural responses",
    required_features=[
        "neural_data",
        "stimulus_presentation",
        "stimulus_timing",
    ],
    optional_features=[
        "stimulus_parameters",
        "multiple_stimulus_types",
        "trial_structure",
        "receptive_field_mapping",
    ],
    negative_conditions=[
        "no_stimulus_info",
        "only_summary_statistics",
    ],
    validation_methods=["nwb_stimulus_check", "bids_events_check"],
    min_trials=20,
    example_use_cases=[
        "Map receptive fields",
        "Build tuning curves",
        "Predict neural response to novel stimuli",
    ],
)

BEHAVIORAL_STATE_DECODING = AffordanceRequirement(
    affordance_id="behavioral_state_decoding",
    label="Behavioral state decoding",
    description="Decode behavioral states (e.g., running, sleeping) from neural data",
    required_features=[
        "neural_data",
        "behavioral_state_labels",
        "temporal_alignment",
    ],
    optional_features=[
        "continuous_behavior",
        "video_data",
        "movement_tracking",
        "multiple_states",
    ],
    negative_conditions=[
        "single_state_only",
        "no_behavior_labels",
    ],
    validation_methods=["nwb_behavior_check", "bids_derivatives_check"],
    example_use_cases=[
        "Decode running vs stationary",
        "Identify sleep stages",
        "Classify behavioral epochs",
    ],
)

CROSS_AREA_INTERACTION = AffordanceRequirement(
    affordance_id="cross_area_interaction",
    label="Cross-area interaction analysis",
    description="Analyze interactions between brain regions",
    required_features=[
        "neural_data",
        "multiple_brain_regions",
        "simultaneous_recording",
    ],
    optional_features=[
        "trial_structure",
        "lfp_data",
        "spike_times",
        "electrode_positions",
    ],
    negative_conditions=[
        "single_region_only",
        "non_simultaneous",
    ],
    validation_methods=["nwb_electrodes_check", "bids_channels_check"],
    example_use_cases=[
        "Analyze cortico-cortical communication",
        "Study information flow between areas",
        "Compute cross-area coherence",
    ],
)

DIMENSIONALITY_REDUCTION = AffordanceRequirement(
    affordance_id="dimensionality_reduction",
    label="Population dimensionality analysis",
    description="Analyze low-dimensional structure in population activity",
    required_features=[
        "neural_population_data",
        "multiple_units_or_voxels",
    ],
    optional_features=[
        "trial_structure",
        "task_variables",
        "time_series",
        "sorted_units",
    ],
    negative_conditions=[
        "single_neuron_only",
        "only_summary_statistics",
    ],
    validation_methods=["nwb_units_count_check"],
    example_use_cases=[
        "Identify latent neural dimensions",
        "Analyze neural manifold structure",
        "Apply PCA/GPFA/LFADS",
    ],
)

FUNCTIONAL_CONNECTIVITY = AffordanceRequirement(
    affordance_id="functional_connectivity",
    label="Functional connectivity analysis",
    description="Analyze statistical dependencies between neural signals",
    required_features=[
        "multi_channel_neural_data",
        "continuous_or_trial_data",
    ],
    optional_features=[
        "rest_data",
        "task_data",
        "anatomical_info",
        "channel_locations",
    ],
    negative_conditions=[
        "single_channel_only",
        "only_event_counts",
    ],
    validation_methods=["nwb_electrodes_check", "bids_channels_check", "fmri_check"],
    example_use_cases=[
        "Compute correlation matrices",
        "Identify functional networks",
        "Compare task vs rest connectivity",
    ],
)

TRIAL_ALIGNED_CALCIUM = AffordanceRequirement(
    affordance_id="trial_aligned_calcium_analysis",
    label="Trial-aligned calcium imaging analysis",
    description="Analyze calcium signals aligned to behavioral trials",
    required_features=[
        "calcium_imaging_data",
        "roi_traces",
        "trial_structure",
    ],
    optional_features=[
        "deconvolved_spikes",
        "behavior_events",
        "stimulus_info",
        "roi_masks",
    ],
    negative_conditions=[
        "no_trial_structure",
        "raw_video_only",
    ],
    validation_methods=["nwb_ophys_check", "nwb_trials_check"],
    min_trials=10,
    example_use_cases=[
        "Analyze task-related calcium signals",
        "Compute trial-averaged responses",
        "Decode from calcium activity",
    ],
)

POSE_NEURAL_CORRELATION = AffordanceRequirement(
    affordance_id="pose_neural_correlation",
    label="Pose-neural correlation analysis",
    description="Correlate body pose/movement with neural activity",
    required_features=[
        "neural_data",
        "pose_tracking_data",
        "temporal_alignment",
    ],
    optional_features=[
        "video_data",
        "multiple_body_parts",
        "continuous_tracking",
        "trial_structure",
    ],
    negative_conditions=[
        "no_pose_data",
        "unaligned_data",
    ],
    validation_methods=["nwb_behavior_check", "bids_motion_check"],
    example_use_cases=[
        "Correlate reaching movements with motor cortex",
        "Analyze locomotion-related activity",
        "Study pose-neural relationships",
    ],
)


# =============================================================================
# Extended Affordances: Reusability-Focused Validators
# =============================================================================

DELAY_DISCOUNTING_MODELING = AffordanceRequirement(
    affordance_id="delay_discounting_modeling",
    label="Delay discounting model fitting",
    description="Fit temporal discounting / intertemporal choice models to behavioral data",
    required_features=[
        "trial_structure",
        "choice_sequence",
        "delay_duration_variable",
        "reward_magnitude_variable",
        "outcome_labels",
    ],
    optional_features=[
        "reaction_time",
        "immediate_reward_option",
        "delayed_reward_option",
        "subject_id",
        "session_id",
        "neural_correlates",
        "reward_timestamps",
    ],
    negative_conditions=[
        "no_trialwise_behavior",
        "no_choice_labels",
        "no_delay_variable",
        "motor_delay_only",
        "signal_propagation_delay_only",
    ],
    validation_methods=["nwb_trials_column_check", "bids_events_column_check"],
    min_trials=50,
    example_use_cases=[
        "Estimate temporal discounting rate (k parameter)",
        "Compare hyperbolic vs exponential discounting",
        "Analyze impulsivity across subjects",
        "Model value-based decision making",
    ],
)

MOTOR_DECODING = AffordanceRequirement(
    affordance_id="motor_decoding",
    label="Motor decoding",
    description="Decode motor actions or movement parameters from neural activity",
    required_features=[
        "neural_data",
        "motor_action_labels",
        "temporal_alignment",
    ],
    optional_features=[
        "movement_kinematics",
        "reaction_time",
        "movement_onset",
        "target_position",
        "hand_position",
        "velocity_data",
        "trial_structure",
        "sorted_units",
    ],
    negative_conditions=[
        "no_motor_data",
        "only_summary_statistics",
        "passive_viewing_only",
    ],
    validation_methods=["nwb_trials_column_check", "nwb_behavior_check"],
    min_trials=30,
    example_use_cases=[
        "Decode reach direction from motor cortex",
        "Predict movement onset from neural activity",
        "Build neural prosthetic decoder",
        "Analyze preparatory motor activity",
    ],
)

TRIAL_ALIGNED_NEURAL_ANALYSIS = AffordanceRequirement(
    affordance_id="trial_aligned_neural_analysis",
    label="Trial-aligned neural analysis",
    description="Analyze neural activity aligned to trial events with proper epoching",
    required_features=[
        "neural_data",
        "trial_structure",
        "event_timestamps",
        "trial_onset_times",
    ],
    optional_features=[
        "trial_outcome",
        "stimulus_onset",
        "response_onset",
        "multiple_conditions",
        "baseline_period",
        "sorted_units",
        "continuous_neural_data",
    ],
    negative_conditions=[
        "no_trial_structure",
        "no_temporal_alignment",
        "aggregate_only",
    ],
    validation_methods=["nwb_trials_check", "nwb_units_check", "bids_events_check"],
    min_trials=20,
    example_use_cases=[
        "Compute trial-averaged firing rates",
        "Analyze condition-specific neural responses",
        "Build peristimulus time histograms",
        "Study trial-to-trial variability",
    ],
)

CROSS_SESSION_GENERALIZATION = AffordanceRequirement(
    affordance_id="cross_session_generalization",
    label="Cross-session generalization analysis",
    description="Analyze neural representations across multiple recording sessions",
    required_features=[
        "neural_data",
        "multiple_sessions",
        "session_id",
        "consistent_task_structure",
    ],
    optional_features=[
        "unit_matching",
        "electrode_positions",
        "session_timestamps",
        "behavioral_performance",
        "chronic_recording",
        "trial_structure",
    ],
    negative_conditions=[
        "single_session_only",
        "no_session_labels",
    ],
    validation_methods=["nwb_sessions_check"],
    min_sessions=2,
    example_use_cases=[
        "Track neural representation stability over days",
        "Train decoder on one session, test on another",
        "Study learning-related neural changes",
        "Analyze chronic recording stability",
    ],
)


# =============================================================================
# Affordance Registry
# =============================================================================

AFFORDANCE_REGISTRY: dict[str, AffordanceRequirement] = {
    # Core affordances
    "event_aligned_psth": EVENT_ALIGNED_PSTH,
    "choice_decoding": CHOICE_DECODING,
    "q_learning": Q_LEARNING,
    "stimulus_response_modeling": STIMULUS_RESPONSE_MODELING,
    "behavioral_state_decoding": BEHAVIORAL_STATE_DECODING,
    "cross_area_interaction": CROSS_AREA_INTERACTION,
    "dimensionality_reduction": DIMENSIONALITY_REDUCTION,
    "functional_connectivity": FUNCTIONAL_CONNECTIVITY,
    "trial_aligned_calcium_analysis": TRIAL_ALIGNED_CALCIUM,
    "pose_neural_correlation": POSE_NEURAL_CORRELATION,
    # Extended affordances (reusability-focused)
    "delay_discounting_modeling": DELAY_DISCOUNTING_MODELING,
    "motor_decoding": MOTOR_DECODING,
    "trial_aligned_neural_analysis": TRIAL_ALIGNED_NEURAL_ANALYSIS,
    "cross_session_generalization": CROSS_SESSION_GENERALIZATION,
}


def get_affordance(affordance_id: str) -> AffordanceRequirement | None:
    """Get an affordance requirement by ID."""
    return AFFORDANCE_REGISTRY.get(affordance_id)


def list_affordances() -> list[str]:
    """List all registered affordance IDs."""
    return list(AFFORDANCE_REGISTRY.keys())


def get_all_affordances() -> list[AffordanceRequirement]:
    """Get all registered affordance requirements."""
    return list(AFFORDANCE_REGISTRY.values())


# =============================================================================
# Feature Detection
# =============================================================================

@dataclass
class DatasetFeatures:
    """Detected features of a dataset for affordance validation."""

    dataset_id: str

    # Neural data features
    has_neural_data: bool = False
    has_spike_times: bool = False
    has_firing_rates: bool = False
    has_lfp: bool = False
    has_calcium_imaging: bool = False
    has_roi_traces: bool = False
    has_fmri: bool = False
    has_eeg: bool = False
    has_ecog: bool = False
    unit_count: int = 0
    channel_count: int = 0

    # Trial/event features
    has_trial_structure: bool = False
    has_event_timestamps: bool = False
    trial_count: int = 0
    event_types: list[str] = field(default_factory=list)

    # Behavioral features
    has_behavior: bool = False
    has_choice_labels: bool = False
    has_reward_signal: bool = False
    has_outcome_labels: bool = False
    has_reaction_time: bool = False
    has_continuous_behavior: bool = False
    has_pose_tracking: bool = False
    has_behavioral_states: bool = False

    # Delay discounting specific
    has_delay_duration: bool = False
    has_reward_magnitude: bool = False
    has_intertemporal_choice: bool = False

    # Motor features
    has_motor_actions: bool = False
    has_movement_kinematics: bool = False
    has_movement_onset: bool = False

    # Session features
    session_count: int = 1
    has_session_labels: bool = False
    has_chronic_recording: bool = False

    # Trial onset features
    has_trial_onset_times: bool = False

    # Stimulus features
    has_stimulus_info: bool = False
    has_stimulus_timing: bool = False
    stimulus_types: list[str] = field(default_factory=list)

    # Recording features
    brain_regions: list[str] = field(default_factory=list)
    has_multiple_regions: bool = False
    has_simultaneous_recording: bool = False

    # Data format
    data_format: DataFormat = DataFormat.METADATA_ONLY
    data_standards: list[str] = field(default_factory=list)

    # Detection provenance
    detection_method: str = "metadata"
    detection_confidence: float = 0.5


def detect_features_from_metadata(
    dataset: dict[str, Any],
) -> DatasetFeatures:
    """Detect dataset features from metadata (without file inspection).

    This is the fast path - uses normalized metadata to infer features.
    For higher confidence, use file-based validation.
    """
    dataset_id = dataset.get("dataset_id", "unknown")

    features = DatasetFeatures(
        dataset_id=dataset_id,
        detection_method="metadata",
        detection_confidence=0.6,
    )

    # Extract modalities
    modalities = set()
    for m in dataset.get("modality", []) + dataset.get("modalities", []):
        if isinstance(m, str):
            modalities.add(m.lower())
        elif hasattr(m, "label"):
            modalities.add(m.label.lower())

    # Neural data detection
    neural_modalities = {
        "electrophysiology", "neuropixels", "tetrode", "ephys",
        "calcium_imaging", "two_photon", "2p", "gcamp",
        "fmri", "bold",
        "eeg", "meg", "ecog", "ieeg", "lfp",
        "fiber_photometry",
    }
    features.has_neural_data = bool(modalities & neural_modalities)
    features.has_spike_times = bool(modalities & {"electrophysiology", "neuropixels", "tetrode", "ephys"})
    features.has_calcium_imaging = bool(modalities & {"calcium_imaging", "two_photon", "2p", "gcamp"})
    features.has_fmri = "fmri" in modalities or "bold" in modalities
    features.has_eeg = "eeg" in modalities
    features.has_ecog = bool(modalities & {"ecog", "ieeg"})
    features.has_lfp = "lfp" in modalities

    # Behavioral features from events
    behavioral_events = set()
    for e in dataset.get("behavioral_events", []):
        if isinstance(e, str):
            behavioral_events.add(e.lower())
        elif hasattr(e, "label"):
            behavioral_events.add(e.label.lower())

    choice_events = {"choice", "response", "decision", "left", "right", "go", "nogo"}
    reward_events = {"reward", "outcome", "feedback", "correct", "error"}

    features.has_choice_labels = bool(behavioral_events & choice_events)
    features.has_reward_signal = bool(behavioral_events & reward_events)
    features.has_outcome_labels = bool(behavioral_events & reward_events)
    features.has_event_timestamps = len(behavioral_events) > 0
    features.event_types = list(behavioral_events)

    # Trial structure from usability flags
    usability = dataset.get("usability", {})
    if isinstance(usability, dict):
        features.has_trial_structure = usability.get("has_trials", False)
        features.has_behavior = usability.get("has_behavior", False)
        features.has_continuous_behavior = usability.get("has_continuous_behavior", False)
    else:
        features.has_trial_structure = getattr(usability, "has_trials", False)
        features.has_behavior = getattr(usability, "has_behavior", False)
        features.has_continuous_behavior = getattr(usability, "has_continuous_behavior", False)

    # Alternative: check for trial count
    if dataset.get("n_trials", 0) > 0:
        features.has_trial_structure = True
        features.trial_count = dataset.get("n_trials", 0)

    # Brain regions
    regions = []
    for r in dataset.get("brain_region", []) + dataset.get("brain_regions", []):
        if isinstance(r, str):
            regions.append(r.lower())
        elif hasattr(r, "label"):
            regions.append(r.label.lower())

    features.brain_regions = regions
    features.has_multiple_regions = len(set(regions)) > 1

    # Data standards
    standards = []
    for s in dataset.get("data_standards", []):
        if isinstance(s, str):
            standards.append(s.lower())
        elif hasattr(s, "label"):
            standards.append(s.label.lower())

    features.data_standards = standards
    if "nwb" in standards:
        features.data_format = DataFormat.NWB
    elif "bids" in standards:
        features.data_format = DataFormat.BIDS

    # Stimulus info from tasks/stimuli
    stimuli = []
    for s in dataset.get("stimuli", []):
        if isinstance(s, str):
            stimuli.append(s.lower())
        elif hasattr(s, "label"):
            stimuli.append(s.label.lower())

    features.stimulus_types = stimuli
    features.has_stimulus_info = len(stimuli) > 0

    # Infer stimulus timing from event timestamps
    features.has_stimulus_timing = "stimulus_onset" in behavioral_events or "stimulus" in behavioral_events

    return features


# =============================================================================
# Affordance Validation
# =============================================================================

def validate_affordance(
    affordance_id: str,
    features: DatasetFeatures,
) -> AffordanceValidationResult:
    """Validate whether a dataset supports an affordance.

    Args:
        affordance_id: ID of the affordance to validate
        features: Detected features of the dataset

    Returns:
        AffordanceValidationResult with support level and evidence
    """
    requirement = get_affordance(affordance_id)
    if requirement is None:
        return AffordanceValidationResult(
            dataset_id=features.dataset_id,
            affordance_id=affordance_id,
            supported=False,
            support_level="unknown",
            confidence=0.0,
            validation_notes=f"Unknown affordance: {affordance_id}",
        )

    # Check required features
    found_required = []
    missing_required = []

    feature_checks = _get_feature_checks()

    for req_feature in requirement.required_features:
        check_fn = feature_checks.get(req_feature)
        if check_fn and check_fn(features):
            found_required.append(req_feature)
        else:
            missing_required.append(req_feature)

    # Check optional features
    found_optional = []
    for opt_feature in requirement.optional_features:
        check_fn = feature_checks.get(opt_feature)
        if check_fn and check_fn(features):
            found_optional.append(opt_feature)

    # Check negative conditions
    negative_found = []
    negative_checks = _get_negative_checks()
    for neg_condition in requirement.negative_conditions:
        check_fn = negative_checks.get(neg_condition)
        if check_fn and check_fn(features):
            negative_found.append(neg_condition)

    # Determine support level
    if negative_found:
        support_level = SupportLevel.UNSUPPORTED
        supported = False
        confidence = 0.2
    elif not missing_required:
        # All required features found
        if features.detection_method == "file":
            support_level = SupportLevel.HIGH
            confidence = 0.95
        else:
            support_level = SupportLevel.MEDIUM
            confidence = 0.75
        supported = True
    elif len(found_required) >= len(requirement.required_features) * 0.5:
        # Some required features found
        support_level = SupportLevel.LOW
        supported = True
        confidence = 0.5
    else:
        support_level = SupportLevel.UNSUPPORTED
        supported = False
        confidence = 0.3

    # Build evidence
    evidence = []
    if found_required:
        evidence.append(
            ProvenanceEvidence(
                evidence_type="structured_metadata",
                source=features.detection_method,
                text=f"Found required features: {', '.join(found_required)}",
                confidence=features.detection_confidence,
                extractor="affordance_validator",
                extractor_version="v1.0.0",
            )
        )

    return AffordanceValidationResult(
        dataset_id=features.dataset_id,
        affordance_id=affordance_id,
        supported=supported,
        support_level=support_level.value,
        confidence=confidence,
        found_required_features=found_required,
        missing_required_features=missing_required,
        found_optional_features=found_optional,
        negative_conditions_found=negative_found,
        evidence=evidence,
        validation_method=features.detection_method,
        validation_notes=f"Detected via {features.detection_method}",
    )


def validate_all_affordances(
    features: DatasetFeatures,
) -> list[AffordanceValidationResult]:
    """Validate all affordances for a dataset."""
    results = []
    for affordance_id in list_affordances():
        result = validate_affordance(affordance_id, features)
        results.append(result)
    return results


def _get_feature_checks() -> dict[str, Any]:
    """Get feature check functions."""
    return {
        # Neural data
        "neural_data": lambda f: f.has_neural_data,
        "spike_times_or_firing_rates": lambda f: f.has_spike_times or f.has_firing_rates,
        "neural_population_data": lambda f: f.has_neural_data and f.unit_count > 1,
        "multiple_units_or_voxels": lambda f: f.unit_count > 1 or f.channel_count > 1 or f.has_fmri,
        "multi_channel_neural_data": lambda f: f.channel_count > 1 or f.has_fmri,
        "calcium_imaging_data": lambda f: f.has_calcium_imaging,
        "roi_traces": lambda f: f.has_roi_traces or f.has_calcium_imaging,

        # Trial/event
        "trial_structure": lambda f: f.has_trial_structure,
        "event_timestamps": lambda f: f.has_event_timestamps,
        "ordered_trials": lambda f: f.has_trial_structure,
        "continuous_or_trial_data": lambda f: f.has_trial_structure or f.has_continuous_behavior,

        # Behavioral
        "choice_labels": lambda f: f.has_choice_labels,
        "choice_sequence": lambda f: f.has_choice_labels and f.has_trial_structure,
        "reward_signal": lambda f: f.has_reward_signal,
        "outcome_labels": lambda f: f.has_outcome_labels,
        "behavioral_state_labels": lambda f: f.has_behavioral_states,
        "pose_tracking_data": lambda f: f.has_pose_tracking,
        "temporal_alignment": lambda f: f.has_event_timestamps,
        "reaction_time": lambda f: f.has_reaction_time,

        # Delay discounting specific
        "delay_duration_variable": lambda f: f.has_delay_duration or f.has_intertemporal_choice,
        "reward_magnitude_variable": lambda f: f.has_reward_magnitude or f.has_intertemporal_choice,
        "immediate_reward_option": lambda f: f.has_intertemporal_choice,
        "delayed_reward_option": lambda f: f.has_intertemporal_choice,

        # Motor features
        "motor_action_labels": lambda f: f.has_motor_actions,
        "movement_kinematics": lambda f: f.has_movement_kinematics,
        "movement_onset": lambda f: f.has_movement_onset,

        # Trial onset
        "trial_onset_times": lambda f: f.has_trial_onset_times or f.has_event_timestamps,

        # Session features
        "multiple_sessions": lambda f: f.session_count > 1,
        "session_id": lambda f: f.has_session_labels or f.session_count > 1,
        "consistent_task_structure": lambda f: f.has_trial_structure,  # Approximation

        # Stimulus
        "stimulus_presentation": lambda f: f.has_stimulus_info,
        "stimulus_timing": lambda f: f.has_stimulus_timing,
        "stimulus_onset": lambda f: f.has_stimulus_timing,
        "response_onset": lambda f: f.has_event_timestamps,

        # Recording
        "multiple_brain_regions": lambda f: f.has_multiple_regions,
        "simultaneous_recording": lambda f: f.has_simultaneous_recording or f.has_multiple_regions,

        # Optional features for completeness
        "subject_id": lambda _: True,  # Almost always present
        "sorted_units": lambda f: f.has_spike_times,
        "continuous_neural_data": lambda f: f.has_lfp or f.has_eeg or f.has_ecog,
        "multiple_conditions": lambda f: len(f.event_types) > 1,
        "baseline_period": lambda _: True,  # Assumed available with trial structure
        "trial_outcome": lambda f: f.has_outcome_labels,
        "neural_correlates": lambda f: f.has_neural_data,
        "chronic_recording": lambda f: f.has_chronic_recording,
        "electrode_positions": lambda f: f.has_multiple_regions,
        "unit_matching": lambda _: False,  # Rare, need explicit detection
        "session_timestamps": lambda f: f.has_session_labels,
        "behavioral_performance": lambda f: f.has_behavior,
        "reward_timestamps": lambda f: f.has_reward_signal and f.has_event_timestamps,
    }


def _get_negative_checks() -> dict[str, Any]:
    """Get negative condition check functions."""
    return {
        "only_summary_statistics": lambda f: not f.has_trial_structure and not f.has_event_timestamps,
        "no_temporal_alignment": lambda f: not f.has_event_timestamps,
        "aggregate_only": lambda f: not f.has_trial_structure,
        "no_choice_labels": lambda f: not f.has_choice_labels,
        "single_trial": lambda f: f.trial_count == 1,
        "no_trialwise_behavior": lambda f: not f.has_trial_structure or not f.has_behavior,
        "shuffled_trials": lambda _: False,  # Can't detect from metadata
        "no_stimulus_info": lambda f: not f.has_stimulus_info,
        "single_state_only": lambda _: False,  # Can't detect from metadata
        "no_behavior_labels": lambda f: not f.has_behavior,
        "single_region_only": lambda f: not f.has_multiple_regions,
        "non_simultaneous": lambda f: not f.has_simultaneous_recording and not f.has_multiple_regions,
        "single_neuron_only": lambda f: f.unit_count <= 1 and f.channel_count <= 1,
        "single_channel_only": lambda f: f.channel_count <= 1,
        "only_event_counts": lambda f: not f.has_spike_times and not f.has_lfp,
        "no_trial_structure": lambda f: not f.has_trial_structure,
        "raw_video_only": lambda f: not f.has_roi_traces and not f.has_calcium_imaging,
        "no_pose_data": lambda f: not f.has_pose_tracking,
        "unaligned_data": lambda f: not f.has_event_timestamps,

        # Delay discounting specific negatives
        "no_delay_variable": lambda f: not f.has_delay_duration and not f.has_intertemporal_choice,
        "motor_delay_only": lambda f: f.has_motor_actions and not f.has_intertemporal_choice,
        "signal_propagation_delay_only": lambda _: False,  # Detected via query sense disambiguation

        # Motor decoding negatives
        "no_motor_data": lambda f: not f.has_motor_actions and not f.has_movement_kinematics,
        "passive_viewing_only": lambda f: not f.has_behavior and not f.has_motor_actions,

        # Session negatives
        "single_session_only": lambda f: f.session_count <= 1,
        "no_session_labels": lambda f: not f.has_session_labels and f.session_count <= 1,
    }
