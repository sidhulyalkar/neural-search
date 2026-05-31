"""Neural Signature schema for content-derived dataset features.

This module defines NeuralSignatureV1, a schema for representing
content-derived features extracted directly from NWB files.

These signatures enable:
- Content-based similarity search ("find datasets with similar firing patterns")
- Dataset characterization beyond metadata
- Quality-based filtering

Usage:
    from neural_search.core.neural_signature import (
        NeuralSignatureV1,
        extract_signature_from_nwb,
    )

    # From NWB file
    signature = extract_signature_from_nwb("path/to/file.nwb")

    # Manual construction
    signature = NeuralSignatureV1(
        dataset_id="dandi:000003",
        modality="ephys",
        n_units=150,
        brain_regions=["M1", "PMd"],
    )
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RecordingModality(StrEnum):
    """Neural recording modality types."""

    EPHYS = "ephys"
    CALCIUM_IMAGING = "calcium_imaging"
    FMRI = "fmri"
    EEG = "eeg"
    MEG = "meg"
    NIRS = "nirs"
    ECOG = "ecog"
    LFP = "lfp"
    MULTI_MODAL = "multi_modal"
    UNKNOWN = "unknown"


class SignatureQuality(StrEnum):
    """Quality level of extracted signature."""

    HIGH = "high"       # Direct extraction from file
    MEDIUM = "medium"   # Partial extraction, some estimates
    LOW = "low"         # Mostly inferred from metadata


class FiringRateStats(BaseModel):
    """Firing rate statistics for ephys data."""

    mean_hz: float | None = None
    median_hz: float | None = None
    std_hz: float | None = None
    min_hz: float | None = None
    max_hz: float | None = None
    n_units_sampled: int | None = None


class ISIStats(BaseModel):
    """Inter-spike interval statistics."""

    mean_ms: float | None = None
    cv: float | None = None
    burst_fraction: float | None = None


class CalciumStats(BaseModel):
    """Calcium imaging statistics."""

    mean_snr: float | None = None
    active_roi_fraction: float | None = None
    mean_event_rate_hz: float | None = None


class TrialStats(BaseModel):
    """Trial structure statistics."""

    n_trials: int | None = None
    trial_duration_mean_s: float | None = None
    trial_duration_std_s: float | None = None
    n_event_types: int | None = None
    event_types: list[str] = Field(default_factory=list)


class NeuralSignatureV1(BaseModel):
    """Content-derived neural signature for a dataset.

    This schema captures features extracted directly from NWB files,
    enabling content-based similarity search and quality assessment.
    """

    # Identity
    dataset_id: str
    asset_id: str | None = None
    signature_id: str | None = None

    # Recording characteristics
    modality: RecordingModality = RecordingModality.UNKNOWN
    duration_seconds: float | None = None
    sampling_rate_hz: float | None = None

    # Neural population
    n_units: int | None = None
    n_rois: int | None = None
    n_channels: int | None = None
    n_electrodes: int | None = None

    # Anatomical coverage
    brain_regions: list[str] = Field(default_factory=list)
    n_brain_regions: int | None = None

    # Trial structure
    trial_stats: TrialStats | None = None

    # Modality-specific stats
    firing_rate_stats: FiringRateStats | None = None
    isi_stats: ISIStats | None = None
    calcium_stats: CalciumStats | None = None

    # Feature vector for similarity search
    feature_vector: list[float] = Field(default_factory=list)
    feature_names: list[str] = Field(default_factory=list)
    feature_version: str = "v1"

    # Quality and provenance
    quality: SignatureQuality = SignatureQuality.LOW
    extraction_version: str = "0.1.0"
    extracted_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_file_hash: str | None = None
    extractor_notes: list[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Generate signature_id if not provided."""
        if not self.signature_id:
            content = f"{self.dataset_id}:{self.asset_id or 'main'}:{self.extraction_version}"
            self.signature_id = hashlib.sha256(content.encode()).hexdigest()[:16]

        if self.brain_regions and self.n_brain_regions is None:
            self.n_brain_regions = len(self.brain_regions)

    def to_feature_dict(self) -> dict[str, float | None]:
        """Convert signature to a feature dictionary for ML."""
        features: dict[str, float | None] = {
            "duration_seconds": self.duration_seconds,
            "sampling_rate_hz": self.sampling_rate_hz,
            "n_units": float(self.n_units) if self.n_units else None,
            "n_rois": float(self.n_rois) if self.n_rois else None,
            "n_channels": float(self.n_channels) if self.n_channels else None,
            "n_brain_regions": float(self.n_brain_regions) if self.n_brain_regions else None,
        }

        if self.firing_rate_stats:
            features["firing_rate_mean"] = self.firing_rate_stats.mean_hz
            features["firing_rate_std"] = self.firing_rate_stats.std_hz

        if self.trial_stats:
            features["n_trials"] = float(self.trial_stats.n_trials) if self.trial_stats.n_trials else None
            features["n_event_types"] = float(self.trial_stats.n_event_types) if self.trial_stats.n_event_types else None

        return features

    def compute_similarity(self, other: NeuralSignatureV1) -> float:
        """Compute similarity to another signature (0-1 scale)."""
        if not self.feature_vector or not other.feature_vector:
            return 0.0

        if len(self.feature_vector) != len(other.feature_vector):
            return 0.0

        # Cosine similarity
        dot = sum(a * b for a, b in zip(self.feature_vector, other.feature_vector))
        norm_a = sum(a * a for a in self.feature_vector) ** 0.5
        norm_b = sum(b * b for b in other.feature_vector) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot / (norm_a * norm_b)


def extract_signature_from_metadata(
    dataset_id: str,
    metadata: dict[str, Any],
) -> NeuralSignatureV1:
    """Extract a signature from dataset metadata (without file access).

    Args:
        dataset_id: Dataset identifier
        metadata: Metadata dictionary with available fields

    Returns:
        NeuralSignatureV1 with metadata-derived fields
    """
    # Determine modality
    modality = RecordingModality.UNKNOWN
    if metadata.get("modality"):
        mod_str = str(metadata["modality"]).lower()
        if "ephys" in mod_str or "spike" in mod_str:
            modality = RecordingModality.EPHYS
        elif "calcium" in mod_str or "imaging" in mod_str:
            modality = RecordingModality.CALCIUM_IMAGING
        elif "fmri" in mod_str or "bold" in mod_str:
            modality = RecordingModality.FMRI
        elif "eeg" in mod_str:
            modality = RecordingModality.EEG
        elif "meg" in mod_str:
            modality = RecordingModality.MEG

    # Build trial stats if available
    trial_stats = None
    if metadata.get("n_trials"):
        trial_stats = TrialStats(
            n_trials=metadata.get("n_trials"),
            event_types=metadata.get("event_types", []),
        )

    return NeuralSignatureV1(
        dataset_id=dataset_id,
        modality=modality,
        duration_seconds=metadata.get("duration_seconds"),
        n_units=metadata.get("n_units"),
        n_rois=metadata.get("n_rois"),
        n_channels=metadata.get("n_channels"),
        brain_regions=metadata.get("brain_regions", []),
        trial_stats=trial_stats,
        quality=SignatureQuality.LOW,
        extractor_notes=["Extracted from metadata only, no file access"],
    )


def extract_signature_from_nwb(
    file_path: str,
    dataset_id: str | None = None,
) -> NeuralSignatureV1:
    """Extract a signature from an NWB file.

    Args:
        file_path: Path to NWB file
        dataset_id: Optional dataset ID (defaults to filename)

    Returns:
        NeuralSignatureV1 with extracted features

    Raises:
        ImportError: If pynwb is not installed
        FileNotFoundError: If file doesn't exist
    """
    try:
        from pynwb import NWBHDF5IO
    except ImportError as exc:
        raise ImportError(
            "Install pynwb to extract signatures from NWB files: "
            "pip install pynwb"
        ) from exc

    import hashlib
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NWB file not found: {file_path}")

    # Compute file hash
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read(65536)).hexdigest()[:16]

    if dataset_id is None:
        dataset_id = path.stem

    extractor_notes: list[str] = []

    with NWBHDF5IO(str(path), "r") as io:
        nwbfile = io.read()

        # Recording duration
        duration = None
        if hasattr(nwbfile, "session_start_time") and hasattr(nwbfile, "timestamps_reference_time"):
            pass  # Would compute from actual data

        # Units (spike-sorted neurons)
        n_units = None
        brain_regions: list[str] = []
        firing_stats = None

        if hasattr(nwbfile, "units") and nwbfile.units is not None:
            units_df = nwbfile.units.to_dataframe()
            n_units = len(units_df)

            # Extract brain regions
            if "location" in units_df.columns:
                brain_regions = list(units_df["location"].dropna().unique())

            extractor_notes.append(f"Extracted {n_units} units from units table")

        # ROIs (calcium imaging)
        n_rois = None
        if hasattr(nwbfile, "processing"):
            for mod_name, mod in nwbfile.processing.items():
                if hasattr(mod, "data_interfaces"):
                    for di_name, di in mod.data_interfaces.items():
                        if "PlaneSegmentation" in type(di).__name__:
                            if hasattr(di, "id"):
                                n_rois = len(di.id.data)
                                extractor_notes.append(f"Found {n_rois} ROIs")

        # Electrodes
        n_electrodes = None
        if hasattr(nwbfile, "electrodes") and nwbfile.electrodes is not None:
            n_electrodes = len(nwbfile.electrodes)
            if "location" in nwbfile.electrodes.colnames:
                elec_regions = list(
                    set(nwbfile.electrodes["location"].data[:])
                )
                brain_regions.extend(elec_regions)
            extractor_notes.append(f"Found {n_electrodes} electrodes")

        # Trials
        trial_stats = None
        if hasattr(nwbfile, "trials") and nwbfile.trials is not None:
            trials_df = nwbfile.trials.to_dataframe()
            n_trials = len(trials_df)

            event_types = []
            for col in trials_df.columns:
                if col not in ("start_time", "stop_time", "id"):
                    event_types.append(col)

            trial_stats = TrialStats(
                n_trials=n_trials,
                n_event_types=len(event_types),
                event_types=event_types[:10],  # Limit
            )
            extractor_notes.append(f"Found {n_trials} trials")

        # Deduplicate brain regions
        brain_regions = list(set(brain_regions))

        # Determine modality
        modality = RecordingModality.UNKNOWN
        if n_units and n_units > 0:
            modality = RecordingModality.EPHYS
        elif n_rois and n_rois > 0:
            modality = RecordingModality.CALCIUM_IMAGING

        return NeuralSignatureV1(
            dataset_id=dataset_id,
            modality=modality,
            duration_seconds=duration,
            n_units=n_units,
            n_rois=n_rois,
            n_electrodes=n_electrodes,
            brain_regions=brain_regions,
            trial_stats=trial_stats,
            firing_rate_stats=firing_stats,
            quality=SignatureQuality.HIGH,
            source_file_hash=file_hash,
            extractor_notes=extractor_notes,
        )
