"""NWB file validator for affordance verification.

This module inspects NWB (Neurodata Without Borders) files to validate
whether datasets actually support the analysis affordances predicted
from metadata. It checks for required tables, columns, and data structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ValidationConfidence(Enum):
    """Confidence level of validation result."""

    HIGH = "high"  # Direct file inspection confirmed
    MEDIUM = "medium"  # Inferred from partial inspection
    LOW = "low"  # Only metadata available
    UNKNOWN = "unknown"  # Could not validate


@dataclass
class NWBFeatureCheck:
    """Result of checking a specific NWB feature."""

    feature_name: str
    present: bool
    confidence: ValidationConfidence
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class NWBValidationResult:
    """Complete validation result for an NWB file."""

    file_path: str
    dataset_id: str | None
    validation_timestamp: str
    validator_version: str = "1.0.0"

    # Core structure checks
    has_units_table: NWBFeatureCheck | None = None
    has_trials_table: NWBFeatureCheck | None = None
    has_electrodes: NWBFeatureCheck | None = None
    has_imaging_planes: NWBFeatureCheck | None = None
    has_behavioral_events: NWBFeatureCheck | None = None
    has_stimulus_info: NWBFeatureCheck | None = None
    has_processing_modules: NWBFeatureCheck | None = None

    # Derived feature checks
    feature_checks: dict[str, NWBFeatureCheck] = field(default_factory=dict)

    # Affordance validation results
    affordance_support: dict[str, bool] = field(default_factory=dict)
    affordance_evidence: dict[str, list[str]] = field(default_factory=dict)
    missing_requirements: dict[str, list[str]] = field(default_factory=dict)

    # Overall status
    validation_success: bool = True
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_path": self.file_path,
            "dataset_id": self.dataset_id,
            "validation_timestamp": self.validation_timestamp,
            "validator_version": self.validator_version,
            "affordance_support": self.affordance_support,
            "affordance_evidence": self.affordance_evidence,
            "missing_requirements": self.missing_requirements,
            "validation_success": self.validation_success,
            "validation_errors": self.validation_errors,
            "feature_checks": {
                k: {
                    "feature_name": v.feature_name,
                    "present": v.present,
                    "confidence": v.confidence.value,
                    "details": v.details,
                    "error": v.error,
                }
                for k, v in self.feature_checks.items()
            },
        }


class NWBValidator:
    """Validator for NWB files to check affordance support.

    This validator can work in two modes:
    1. Full inspection: Opens and inspects NWB file contents (requires pynwb)
    2. Metadata-only: Validates based on provided metadata dict (no file access)
    """

    # Required columns for various affordances
    TRIAL_COLUMNS_CHOICE = {"choice", "response", "decision", "action", "selected"}
    TRIAL_COLUMNS_REWARD = {"reward", "outcome", "feedback", "reward_magnitude"}
    TRIAL_COLUMNS_STIMULUS = {"stimulus", "stim_type", "stimulus_id", "stim_name"}
    TRIAL_COLUMNS_TIMING = {"start_time", "stop_time", "response_time", "reaction_time"}

    def __init__(self, use_pynwb: bool = True):
        """Initialize validator.

        Args:
            use_pynwb: Whether to use pynwb for full file inspection.
                       If False, only metadata-based validation is available.
        """
        self.use_pynwb = use_pynwb
        self._pynwb = None

        if use_pynwb:
            try:
                import pynwb

                self._pynwb = pynwb
            except ImportError:
                self.use_pynwb = False

    def validate_file(
        self,
        file_path: str | Path,
        dataset_id: str | None = None,
    ) -> NWBValidationResult:
        """Validate an NWB file for affordance support.

        Args:
            file_path: Path to NWB file
            dataset_id: Optional dataset identifier

        Returns:
            NWBValidationResult with all validation findings
        """
        result = NWBValidationResult(
            file_path=str(file_path),
            dataset_id=dataset_id,
            validation_timestamp=datetime.utcnow().isoformat(),
        )

        if not self.use_pynwb or self._pynwb is None:
            result.validation_success = False
            result.validation_errors.append("pynwb not available for file inspection")
            return result

        try:
            with self._pynwb.NWBHDF5IO(str(file_path), "r") as io:
                nwbfile = io.read()
                self._validate_nwb_contents(nwbfile, result)
        except Exception as e:
            result.validation_success = False
            result.validation_errors.append(f"Failed to read NWB file: {e}")

        # Compute affordance support based on feature checks
        self._compute_affordance_support(result)

        return result

    def validate_from_metadata(
        self,
        metadata: dict[str, Any],
        dataset_id: str | None = None,
    ) -> NWBValidationResult:
        """Validate affordance support from metadata dictionary.

        This is used when actual file access is not available, providing
        lower-confidence validation based on metadata fields.

        Args:
            metadata: Dictionary of NWB metadata
            dataset_id: Optional dataset identifier

        Returns:
            NWBValidationResult with metadata-based findings
        """
        result = NWBValidationResult(
            file_path="metadata_only",
            dataset_id=dataset_id,
            validation_timestamp=datetime.utcnow().isoformat(),
        )

        # Check for units table indicators
        has_units = any(
            k in metadata
            for k in ["units", "has_units", "n_units", "spike_times"]
        )
        result.has_units_table = NWBFeatureCheck(
            feature_name="units_table",
            present=has_units,
            confidence=ValidationConfidence.LOW,
            details={"source": "metadata"},
        )
        result.feature_checks["has_spike_times"] = NWBFeatureCheck(
            feature_name="spike_times",
            present=has_units,
            confidence=ValidationConfidence.LOW,
        )

        # Check for trials table indicators
        has_trials = any(
            k in metadata
            for k in ["trials", "has_trials", "n_trials", "trial_count"]
        )
        result.has_trials_table = NWBFeatureCheck(
            feature_name="trials_table",
            present=has_trials,
            confidence=ValidationConfidence.LOW,
            details={"source": "metadata"},
        )
        result.feature_checks["has_trial_structure"] = NWBFeatureCheck(
            feature_name="trial_structure",
            present=has_trials,
            confidence=ValidationConfidence.LOW,
        )

        # Check for behavioral event indicators
        has_behavior = any(
            k in metadata
            for k in ["behavior", "behavioral_events", "has_behavior", "events"]
        )
        result.has_behavioral_events = NWBFeatureCheck(
            feature_name="behavioral_events",
            present=has_behavior,
            confidence=ValidationConfidence.LOW,
            details={"source": "metadata"},
        )

        # Check for electrode/region info
        has_electrodes = any(
            k in metadata
            for k in ["electrodes", "brain_regions", "n_electrodes", "n_channels"]
        )
        result.has_electrodes = NWBFeatureCheck(
            feature_name="electrodes",
            present=has_electrodes,
            confidence=ValidationConfidence.LOW,
            details={"source": "metadata"},
        )

        # Check for imaging indicators
        has_imaging = any(
            k in metadata
            for k in ["imaging_planes", "rois", "n_rois", "calcium_imaging"]
        )
        result.has_imaging_planes = NWBFeatureCheck(
            feature_name="imaging_planes",
            present=has_imaging,
            confidence=ValidationConfidence.LOW,
            details={"source": "metadata"},
        )

        # Infer additional features from metadata
        self._infer_features_from_metadata(metadata, result)

        # Compute affordance support
        self._compute_affordance_support(result)

        return result

    def _validate_nwb_contents(
        self,
        nwbfile: Any,
        result: NWBValidationResult,
    ) -> None:
        """Validate contents of an opened NWB file."""
        # Check units table
        if hasattr(nwbfile, "units") and nwbfile.units is not None:
            units = nwbfile.units
            n_units = len(units) if hasattr(units, "__len__") else 0
            columns = list(units.colnames) if hasattr(units, "colnames") else []

            result.has_units_table = NWBFeatureCheck(
                feature_name="units_table",
                present=True,
                confidence=ValidationConfidence.HIGH,
                details={"n_units": n_units, "columns": columns},
            )

            # Check for spike times
            has_spike_times = "spike_times" in columns or n_units > 0
            result.feature_checks["has_spike_times"] = NWBFeatureCheck(
                feature_name="spike_times",
                present=has_spike_times,
                confidence=ValidationConfidence.HIGH,
                details={"n_units": n_units},
            )

            result.feature_checks["has_neural_data"] = NWBFeatureCheck(
                feature_name="neural_data",
                present=True,
                confidence=ValidationConfidence.HIGH,
            )

            # Check for population recording
            result.feature_checks["has_population_recording"] = NWBFeatureCheck(
                feature_name="population_recording",
                present=n_units >= 5,
                confidence=ValidationConfidence.HIGH,
                details={"n_units": n_units, "threshold": 5},
            )
        else:
            result.has_units_table = NWBFeatureCheck(
                feature_name="units_table",
                present=False,
                confidence=ValidationConfidence.HIGH,
            )

        # Check trials table
        if hasattr(nwbfile, "trials") and nwbfile.trials is not None:
            trials = nwbfile.trials
            n_trials = len(trials) if hasattr(trials, "__len__") else 0
            columns = list(trials.colnames) if hasattr(trials, "colnames") else []
            columns_lower = {c.lower() for c in columns}

            result.has_trials_table = NWBFeatureCheck(
                feature_name="trials_table",
                present=True,
                confidence=ValidationConfidence.HIGH,
                details={"n_trials": n_trials, "columns": columns},
            )

            result.feature_checks["has_trial_structure"] = NWBFeatureCheck(
                feature_name="trial_structure",
                present=True,
                confidence=ValidationConfidence.HIGH,
                details={"n_trials": n_trials},
            )

            # Check for choice columns
            has_choice = bool(columns_lower & self.TRIAL_COLUMNS_CHOICE)
            result.feature_checks["has_choice_labels"] = NWBFeatureCheck(
                feature_name="choice_labels",
                present=has_choice,
                confidence=ValidationConfidence.HIGH,
                details={"found_columns": list(columns_lower & self.TRIAL_COLUMNS_CHOICE)},
            )

            result.feature_checks["has_choice_sequence"] = NWBFeatureCheck(
                feature_name="choice_sequence",
                present=has_choice and n_trials > 1,
                confidence=ValidationConfidence.HIGH,
            )

            # Check for reward columns
            has_reward = bool(columns_lower & self.TRIAL_COLUMNS_REWARD)
            result.feature_checks["has_reward_signal"] = NWBFeatureCheck(
                feature_name="reward_signal",
                present=has_reward,
                confidence=ValidationConfidence.HIGH,
                details={"found_columns": list(columns_lower & self.TRIAL_COLUMNS_REWARD)},
            )

            # Check for stimulus columns
            has_stimulus = bool(columns_lower & self.TRIAL_COLUMNS_STIMULUS)
            result.feature_checks["has_stimulus_info"] = NWBFeatureCheck(
                feature_name="stimulus_info",
                present=has_stimulus,
                confidence=ValidationConfidence.HIGH,
                details={"found_columns": list(columns_lower & self.TRIAL_COLUMNS_STIMULUS)},
            )

            # Check for event timestamps
            has_timing = bool(columns_lower & self.TRIAL_COLUMNS_TIMING)
            result.feature_checks["has_event_timestamps"] = NWBFeatureCheck(
                feature_name="event_timestamps",
                present=has_timing,
                confidence=ValidationConfidence.HIGH,
                details={"found_columns": list(columns_lower & self.TRIAL_COLUMNS_TIMING)},
            )
        else:
            result.has_trials_table = NWBFeatureCheck(
                feature_name="trials_table",
                present=False,
                confidence=ValidationConfidence.HIGH,
            )

        # Check electrodes
        if hasattr(nwbfile, "electrodes") and nwbfile.electrodes is not None:
            electrodes = nwbfile.electrodes
            n_electrodes = len(electrodes) if hasattr(electrodes, "__len__") else 0
            columns = list(electrodes.colnames) if hasattr(electrodes, "colnames") else []

            result.has_electrodes = NWBFeatureCheck(
                feature_name="electrodes",
                present=True,
                confidence=ValidationConfidence.HIGH,
                details={"n_electrodes": n_electrodes, "columns": columns},
            )

            # Check for brain region info
            has_regions = "location" in columns or "brain_region" in columns
            regions = set()
            if has_regions and hasattr(electrodes, "location"):
                try:
                    regions = set(electrodes.location[:])
                except Exception:
                    pass

            result.feature_checks["has_multiple_brain_regions"] = NWBFeatureCheck(
                feature_name="multiple_brain_regions",
                present=len(regions) > 1,
                confidence=ValidationConfidence.HIGH,
                details={"n_regions": len(regions), "regions": list(regions)[:10]},
            )
        else:
            result.has_electrodes = NWBFeatureCheck(
                feature_name="electrodes",
                present=False,
                confidence=ValidationConfidence.HIGH,
            )

        # Check for imaging planes (calcium imaging)
        has_ophys = (
            hasattr(nwbfile, "imaging_planes")
            and nwbfile.imaging_planes is not None
            and len(nwbfile.imaging_planes) > 0
        )
        result.has_imaging_planes = NWBFeatureCheck(
            feature_name="imaging_planes",
            present=has_ophys,
            confidence=ValidationConfidence.HIGH,
        )

        if has_ophys:
            result.feature_checks["has_calcium_imaging"] = NWBFeatureCheck(
                feature_name="calcium_imaging",
                present=True,
                confidence=ValidationConfidence.HIGH,
            )

        # Check for ROI traces in processing modules
        has_roi_traces = False
        if hasattr(nwbfile, "processing") and nwbfile.processing:
            for module_name in nwbfile.processing:
                module = nwbfile.processing[module_name]
                if hasattr(module, "data_interfaces"):
                    for interface_name in module.data_interfaces:
                        if "fluorescence" in interface_name.lower() or "dff" in interface_name.lower():
                            has_roi_traces = True
                            break

        result.feature_checks["has_roi_traces"] = NWBFeatureCheck(
            feature_name="roi_traces",
            present=has_roi_traces,
            confidence=ValidationConfidence.HIGH,
        )

        # Check for behavioral time series
        has_behavior = False
        if hasattr(nwbfile, "processing") and nwbfile.processing:
            for module_name in nwbfile.processing:
                if "behavior" in module_name.lower():
                    has_behavior = True
                    break

        result.has_behavioral_events = NWBFeatureCheck(
            feature_name="behavioral_events",
            present=has_behavior,
            confidence=ValidationConfidence.HIGH,
        )

        result.feature_checks["has_behavior"] = NWBFeatureCheck(
            feature_name="behavior",
            present=has_behavior,
            confidence=ValidationConfidence.HIGH,
        )

        # Check for stimulus presentations
        has_stimulus = (
            hasattr(nwbfile, "stimulus")
            and nwbfile.stimulus is not None
            and len(nwbfile.stimulus) > 0
        )
        result.has_stimulus_info = NWBFeatureCheck(
            feature_name="stimulus_info",
            present=has_stimulus,
            confidence=ValidationConfidence.HIGH,
        )

        if has_stimulus:
            result.feature_checks["has_stimulus_timing"] = NWBFeatureCheck(
                feature_name="stimulus_timing",
                present=True,
                confidence=ValidationConfidence.HIGH,
            )

    def _infer_features_from_metadata(
        self,
        metadata: dict[str, Any],
        result: NWBValidationResult,
    ) -> None:
        """Infer additional features from metadata fields."""
        # Neural data indicators
        has_neural = any(
            k in metadata
            for k in [
                "units", "electrodes", "n_units", "n_channels",
                "ephys", "electrophysiology", "spike_times", "neural"
            ]
        )
        result.feature_checks["has_neural_data"] = NWBFeatureCheck(
            feature_name="neural_data",
            present=has_neural,
            confidence=ValidationConfidence.LOW,
        )

        # Choice-related features
        has_choice = any(
            k in str(metadata).lower()
            for k in ["choice", "decision", "response", "action"]
        )
        result.feature_checks["has_choice_labels"] = NWBFeatureCheck(
            feature_name="choice_labels",
            present=has_choice,
            confidence=ValidationConfidence.LOW,
        )

        # Reward features
        has_reward = any(
            k in str(metadata).lower()
            for k in ["reward", "outcome", "feedback"]
        )
        result.feature_checks["has_reward_signal"] = NWBFeatureCheck(
            feature_name="reward_signal",
            present=has_reward,
            confidence=ValidationConfidence.LOW,
        )

        # Multiple brain regions
        regions = metadata.get("brain_regions", [])
        if isinstance(regions, str):
            regions = [regions]
        result.feature_checks["has_multiple_brain_regions"] = NWBFeatureCheck(
            feature_name="multiple_brain_regions",
            present=len(regions) > 1,
            confidence=ValidationConfidence.LOW,
            details={"n_regions": len(regions)},
        )

        # Population recording
        n_units = metadata.get("n_units", 0)
        result.feature_checks["has_population_recording"] = NWBFeatureCheck(
            feature_name="population_recording",
            present=n_units >= 5 if isinstance(n_units, int) else False,
            confidence=ValidationConfidence.LOW,
        )

    def _compute_affordance_support(self, result: NWBValidationResult) -> None:
        """Compute affordance support based on feature checks."""
        from neural_search.affordances.registry import AFFORDANCE_REGISTRY

        for affordance_id, requirement in AFFORDANCE_REGISTRY.items():
            required = requirement.required_features
            negative = requirement.negative_conditions

            # Check required features
            missing = []
            found = []
            for feature in required:
                check = result.feature_checks.get(f"has_{feature}")
                if check is None:
                    # Try without has_ prefix
                    check = result.feature_checks.get(feature)

                if check is not None and check.present:
                    found.append(feature)
                else:
                    missing.append(feature)

            # Check negative conditions
            negative_triggered = []
            for condition in negative:
                check = result.feature_checks.get(condition)
                if check is not None and check.present:
                    negative_triggered.append(condition)

            # Determine support
            supported = len(missing) == 0 and len(negative_triggered) == 0

            result.affordance_support[affordance_id] = supported
            result.affordance_evidence[affordance_id] = found
            if missing or negative_triggered:
                result.missing_requirements[affordance_id] = missing + [
                    f"negative:{c}" for c in negative_triggered
                ]


def validate_nwb_affordances(
    file_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    dataset_id: str | None = None,
) -> NWBValidationResult:
    """Convenience function to validate NWB affordances.

    Args:
        file_path: Path to NWB file (for full inspection)
        metadata: Metadata dictionary (for metadata-only validation)
        dataset_id: Optional dataset identifier

    Returns:
        NWBValidationResult with validation findings

    Raises:
        ValueError: If neither file_path nor metadata is provided
    """
    validator = NWBValidator()

    if file_path is not None:
        return validator.validate_file(file_path, dataset_id)
    elif metadata is not None:
        return validator.validate_from_metadata(metadata, dataset_id)
    else:
        raise ValueError("Either file_path or metadata must be provided")
