"""BIDS dataset validator for affordance verification.

This module inspects BIDS (Brain Imaging Data Structure) datasets to validate
whether they support the analysis affordances predicted from metadata.
It checks for required files, directory structure, and metadata completeness.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class BIDSFeatureCheck:
    """Result of checking a specific BIDS feature."""

    feature_name: str
    present: bool
    confidence: str = "high"  # high, medium, low
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class BIDSValidationResult:
    """Complete validation result for a BIDS dataset."""

    dataset_path: str
    dataset_id: str | None
    validation_timestamp: str
    validator_version: str = "1.0.0"

    # Dataset structure checks
    has_participants_tsv: BIDSFeatureCheck | None = None
    has_dataset_description: BIDSFeatureCheck | None = None
    has_events_files: BIDSFeatureCheck | None = None
    has_task_labels: BIDSFeatureCheck | None = None

    # Modality-specific checks
    modalities_found: list[str] = field(default_factory=list)
    has_func: bool = False
    has_anat: bool = False
    has_dwi: bool = False
    has_eeg: bool = False
    has_meg: bool = False
    has_ieeg: bool = False
    has_pet: bool = False

    # Feature checks
    feature_checks: dict[str, BIDSFeatureCheck] = field(default_factory=dict)

    # Affordance validation results
    affordance_support: dict[str, bool] = field(default_factory=dict)
    affordance_evidence: dict[str, list[str]] = field(default_factory=dict)
    missing_requirements: dict[str, list[str]] = field(default_factory=dict)

    # Statistics
    n_subjects: int = 0
    n_sessions: int = 0
    tasks: list[str] = field(default_factory=list)
    event_columns: list[str] = field(default_factory=list)

    # Overall status
    validation_success: bool = True
    validation_errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset_path": self.dataset_path,
            "dataset_id": self.dataset_id,
            "validation_timestamp": self.validation_timestamp,
            "validator_version": self.validator_version,
            "modalities_found": self.modalities_found,
            "n_subjects": self.n_subjects,
            "n_sessions": self.n_sessions,
            "tasks": self.tasks,
            "event_columns": self.event_columns,
            "affordance_support": self.affordance_support,
            "affordance_evidence": self.affordance_evidence,
            "missing_requirements": self.missing_requirements,
            "validation_success": self.validation_success,
            "validation_errors": self.validation_errors,
            "feature_checks": {
                k: {
                    "feature_name": v.feature_name,
                    "present": v.present,
                    "confidence": v.confidence,
                    "details": v.details,
                    "error": v.error,
                }
                for k, v in self.feature_checks.items()
            },
        }


class BIDSValidator:
    """Validator for BIDS datasets to check affordance support.

    This validator can work in two modes:
    1. Full inspection: Scans BIDS directory structure and files
    2. Metadata-only: Validates based on provided metadata dict
    """

    # BIDS modality directories
    MODALITY_DIRS = {
        "func": "fmri",
        "anat": "anatomical_mri",
        "dwi": "diffusion_mri",
        "eeg": "eeg",
        "meg": "meg",
        "ieeg": "ieeg",
        "pet": "pet",
        "beh": "behavior",
        "perf": "perfusion",
    }

    # Event columns that indicate various features
    EVENT_COLUMNS_TRIAL = {"trial_type", "trial_number", "trial_index"}
    EVENT_COLUMNS_STIMULUS = {"stimulus", "stim_file", "stim_type", "stimulus_id"}
    EVENT_COLUMNS_RESPONSE = {"response", "response_time", "rt", "reaction_time"}
    EVENT_COLUMNS_CONDITION = {"condition", "trial_type", "task_type"}

    def __init__(self):
        """Initialize validator."""
        pass

    def validate_dataset(
        self,
        dataset_path: str | Path,
        dataset_id: str | None = None,
    ) -> BIDSValidationResult:
        """Validate a BIDS dataset for affordance support.

        Args:
            dataset_path: Path to BIDS dataset root directory
            dataset_id: Optional dataset identifier

        Returns:
            BIDSValidationResult with all validation findings
        """
        dataset_path = Path(dataset_path)
        result = BIDSValidationResult(
            dataset_path=str(dataset_path),
            dataset_id=dataset_id,
            validation_timestamp=datetime.utcnow().isoformat(),
        )

        if not dataset_path.exists():
            result.validation_success = False
            result.validation_errors.append(f"Dataset path does not exist: {dataset_path}")
            return result

        # Check dataset_description.json
        self._check_dataset_description(dataset_path, result)

        # Check participants.tsv
        self._check_participants(dataset_path, result)

        # Scan for subjects and sessions
        self._scan_subjects_sessions(dataset_path, result)

        # Check for modalities
        self._check_modalities(dataset_path, result)

        # Check for events files
        self._check_events_files(dataset_path, result)

        # Check for task labels
        self._check_task_labels(dataset_path, result)

        # Compute derived features
        self._compute_derived_features(result)

        # Compute affordance support
        self._compute_affordance_support(result)

        return result

    def validate_from_metadata(
        self,
        metadata: dict[str, Any],
        dataset_id: str | None = None,
    ) -> BIDSValidationResult:
        """Validate affordance support from metadata dictionary.

        Args:
            metadata: Dictionary of BIDS metadata
            dataset_id: Optional dataset identifier

        Returns:
            BIDSValidationResult with metadata-based findings
        """
        result = BIDSValidationResult(
            dataset_path="metadata_only",
            dataset_id=dataset_id,
            validation_timestamp=datetime.utcnow().isoformat(),
        )

        # Infer modalities
        modality = metadata.get("modality", [])
        if isinstance(modality, str):
            modality = [modality]

        for m in modality:
            m_lower = m.lower()
            if "fmri" in m_lower or "bold" in m_lower:
                result.has_func = True
                result.modalities_found.append("func")
            if "eeg" in m_lower:
                result.has_eeg = True
                result.modalities_found.append("eeg")
            if "meg" in m_lower:
                result.has_meg = True
                result.modalities_found.append("meg")
            if "pet" in m_lower:
                result.has_pet = True
                result.modalities_found.append("pet")
            if "dwi" in m_lower or "diffusion" in m_lower:
                result.has_dwi = True
                result.modalities_found.append("dwi")
            if "ieeg" in m_lower or "ecog" in m_lower:
                result.has_ieeg = True
                result.modalities_found.append("ieeg")

        # Infer subject count
        result.n_subjects = metadata.get("n_subjects", 0)

        # Infer task info
        tasks = metadata.get("tasks", metadata.get("task", []))
        if isinstance(tasks, str):
            tasks = [tasks]
        result.tasks = tasks

        # Check for neural data
        has_neural = any([
            result.has_func, result.has_eeg, result.has_meg,
            result.has_ieeg, result.has_pet
        ])
        result.feature_checks["has_neural_data"] = BIDSFeatureCheck(
            feature_name="neural_data",
            present=has_neural,
            confidence="low",
        )

        # Check for task structure
        has_task = len(result.tasks) > 0
        result.feature_checks["has_trial_structure"] = BIDSFeatureCheck(
            feature_name="trial_structure",
            present=has_task,
            confidence="low",
        )

        # Infer continuous data (fMRI, EEG, MEG)
        result.feature_checks["has_continuous_data"] = BIDSFeatureCheck(
            feature_name="continuous_data",
            present=has_neural,
            confidence="low",
        )

        # Infer multiple channels
        n_channels = metadata.get("n_channels", 0)
        result.feature_checks["has_multiple_channels"] = BIDSFeatureCheck(
            feature_name="multiple_channels",
            present=n_channels > 1 or result.has_eeg or result.has_meg,
            confidence="low",
        )

        # Compute affordance support
        self._compute_affordance_support(result)

        return result

    def _check_dataset_description(
        self,
        dataset_path: Path,
        result: BIDSValidationResult,
    ) -> None:
        """Check for dataset_description.json."""
        desc_file = dataset_path / "dataset_description.json"
        if desc_file.exists():
            try:
                with open(desc_file) as f:
                    desc = json.load(f)
                result.has_dataset_description = BIDSFeatureCheck(
                    feature_name="dataset_description",
                    present=True,
                    details={
                        "name": desc.get("Name", ""),
                        "bids_version": desc.get("BIDSVersion", ""),
                    },
                )
            except Exception as e:
                result.has_dataset_description = BIDSFeatureCheck(
                    feature_name="dataset_description",
                    present=True,
                    error=str(e),
                )
        else:
            result.has_dataset_description = BIDSFeatureCheck(
                feature_name="dataset_description",
                present=False,
            )

    def _check_participants(
        self,
        dataset_path: Path,
        result: BIDSValidationResult,
    ) -> None:
        """Check for participants.tsv."""
        participants_file = dataset_path / "participants.tsv"
        if participants_file.exists():
            try:
                with open(participants_file) as f:
                    header = f.readline().strip().split("\t")
                    n_lines = sum(1 for _ in f)

                result.has_participants_tsv = BIDSFeatureCheck(
                    feature_name="participants_tsv",
                    present=True,
                    details={
                        "columns": header,
                        "n_participants": n_lines,
                    },
                )
                result.n_subjects = n_lines
            except Exception as e:
                result.has_participants_tsv = BIDSFeatureCheck(
                    feature_name="participants_tsv",
                    present=True,
                    error=str(e),
                )
        else:
            result.has_participants_tsv = BIDSFeatureCheck(
                feature_name="participants_tsv",
                present=False,
            )

    def _scan_subjects_sessions(
        self,
        dataset_path: Path,
        result: BIDSValidationResult,
    ) -> None:
        """Scan for subject and session directories."""
        subjects = list(dataset_path.glob("sub-*"))
        result.n_subjects = max(result.n_subjects, len(subjects))

        sessions = set()
        for subj_dir in subjects[:10]:  # Sample first 10 subjects
            for ses_dir in subj_dir.glob("ses-*"):
                sessions.add(ses_dir.name)

        result.n_sessions = len(sessions)

        # Check for multiple sessions (useful for cross-session analysis)
        result.feature_checks["has_multiple_sessions"] = BIDSFeatureCheck(
            feature_name="multiple_sessions",
            present=result.n_sessions > 1,
            details={"n_sessions": result.n_sessions},
        )

    def _check_modalities(
        self,
        dataset_path: Path,
        result: BIDSValidationResult,
    ) -> None:
        """Check for BIDS modality directories."""
        # Check in subject directories
        subjects = list(dataset_path.glob("sub-*"))[:5]  # Sample

        modalities_found = set()
        for subj_dir in subjects:
            # Check direct modality dirs
            for mod_dir in self.MODALITY_DIRS:
                if (subj_dir / mod_dir).exists():
                    modalities_found.add(mod_dir)
                # Also check in sessions
                for ses_dir in subj_dir.glob("ses-*"):
                    if (ses_dir / mod_dir).exists():
                        modalities_found.add(mod_dir)

        result.modalities_found = list(modalities_found)
        result.has_func = "func" in modalities_found
        result.has_anat = "anat" in modalities_found
        result.has_dwi = "dwi" in modalities_found
        result.has_eeg = "eeg" in modalities_found
        result.has_meg = "meg" in modalities_found
        result.has_ieeg = "ieeg" in modalities_found
        result.has_pet = "pet" in modalities_found

        # Set neural data feature
        has_neural = any([
            result.has_func, result.has_eeg, result.has_meg,
            result.has_ieeg, result.has_pet
        ])
        result.feature_checks["has_neural_data"] = BIDSFeatureCheck(
            feature_name="neural_data",
            present=has_neural,
            details={"modalities": result.modalities_found},
        )

        # Set continuous data feature
        result.feature_checks["has_continuous_data"] = BIDSFeatureCheck(
            feature_name="continuous_data",
            present=has_neural,
        )

    def _check_events_files(
        self,
        dataset_path: Path,
        result: BIDSValidationResult,
    ) -> None:
        """Check for events.tsv files."""
        events_files = list(dataset_path.rglob("*_events.tsv"))

        if events_files:
            # Sample first few events files to get columns
            all_columns = set()
            for ef in events_files[:10]:
                try:
                    with open(ef) as f:
                        header = f.readline().strip().split("\t")
                        all_columns.update(header)
                except Exception:
                    continue

            result.event_columns = list(all_columns)
            columns_lower = {c.lower() for c in all_columns}

            result.has_events_files = BIDSFeatureCheck(
                feature_name="events_files",
                present=True,
                details={
                    "n_files": len(events_files),
                    "columns": result.event_columns,
                },
            )

            # Check for trial structure
            has_trial_cols = bool(columns_lower & {c.lower() for c in self.EVENT_COLUMNS_TRIAL})
            result.feature_checks["has_trial_structure"] = BIDSFeatureCheck(
                feature_name="trial_structure",
                present=True,  # events.tsv implies trial structure
                details={"trial_columns": list(columns_lower & {c.lower() for c in self.EVENT_COLUMNS_TRIAL})},
            )

            # Check for event timestamps
            has_timing = "onset" in columns_lower or "duration" in columns_lower
            result.feature_checks["has_event_timestamps"] = BIDSFeatureCheck(
                feature_name="event_timestamps",
                present=has_timing,
            )

            # Check for stimulus info
            has_stim = bool(columns_lower & {c.lower() for c in self.EVENT_COLUMNS_STIMULUS})
            result.feature_checks["has_stimulus_info"] = BIDSFeatureCheck(
                feature_name="stimulus_info",
                present=has_stim,
            )

            # Check for condition labels
            has_condition = bool(columns_lower & {c.lower() for c in self.EVENT_COLUMNS_CONDITION})
            result.feature_checks["has_condition_labels"] = BIDSFeatureCheck(
                feature_name="condition_labels",
                present=has_condition,
            )

            # Check for response/choice
            has_response = bool(columns_lower & {c.lower() for c in self.EVENT_COLUMNS_RESPONSE})
            result.feature_checks["has_choice_labels"] = BIDSFeatureCheck(
                feature_name="choice_labels",
                present=has_response,
            )
        else:
            result.has_events_files = BIDSFeatureCheck(
                feature_name="events_files",
                present=False,
            )
            result.feature_checks["has_trial_structure"] = BIDSFeatureCheck(
                feature_name="trial_structure",
                present=False,
            )

    def _check_task_labels(
        self,
        dataset_path: Path,
        result: BIDSValidationResult,
    ) -> None:
        """Check for task labels in filenames."""
        # Look for task-* pattern in filenames
        task_pattern = re.compile(r"task-([a-zA-Z0-9]+)")

        tasks = set()
        for f in dataset_path.rglob("*task-*"):
            match = task_pattern.search(f.name)
            if match:
                tasks.add(match.group(1))

        result.tasks = list(tasks)

        result.has_task_labels = BIDSFeatureCheck(
            feature_name="task_labels",
            present=len(tasks) > 0,
            details={"tasks": result.tasks},
        )

    def _compute_derived_features(self, result: BIDSValidationResult) -> None:
        """Compute derived features from basic checks."""
        # Multiple channels (for EEG/MEG datasets)
        if result.has_eeg or result.has_meg or result.has_ieeg:
            result.feature_checks["has_multiple_channels"] = BIDSFeatureCheck(
                feature_name="multiple_channels",
                present=True,
                details={"modality": "eeg/meg/ieeg"},
            )

        # Check for behavior feature
        has_behavior = (
            result.has_events_files is not None and result.has_events_files.present
        ) or "beh" in result.modalities_found

        result.feature_checks["has_behavior"] = BIDSFeatureCheck(
            feature_name="behavior",
            present=has_behavior,
        )

    def _compute_affordance_support(self, result: BIDSValidationResult) -> None:
        """Compute affordance support based on feature checks."""
        # Import here to avoid circular imports
        try:
            from neural_search.affordances.registry import AFFORDANCE_REGISTRY
        except ImportError:
            result.validation_errors.append("Could not import affordance registry")
            return

        for affordance_id, requirement in AFFORDANCE_REGISTRY.items():
            required = requirement.required_features
            negative = requirement.negative_conditions

            # Check required features
            missing = []
            found = []
            for feature in required:
                check = result.feature_checks.get(f"has_{feature}")
                if check is None:
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


def validate_bids_affordances(
    dataset_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
    dataset_id: str | None = None,
) -> BIDSValidationResult:
    """Convenience function to validate BIDS affordances.

    Args:
        dataset_path: Path to BIDS dataset (for full inspection)
        metadata: Metadata dictionary (for metadata-only validation)
        dataset_id: Optional dataset identifier

    Returns:
        BIDSValidationResult with validation findings

    Raises:
        ValueError: If neither dataset_path nor metadata is provided
    """
    validator = BIDSValidator()

    if dataset_path is not None:
        return validator.validate_dataset(dataset_path, dataset_id)
    elif metadata is not None:
        return validator.validate_from_metadata(metadata, dataset_id)
    else:
        raise ValueError("Either dataset_path or metadata must be provided")
