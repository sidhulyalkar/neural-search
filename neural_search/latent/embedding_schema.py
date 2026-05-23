"""Schemas for experimental latent neural and behavioral state search."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class FeatureType(StrEnum):
    """Feature summary families used by the latent-search scaffold."""

    EVENT_HISTOGRAM = "event_histogram"
    NEURAL_SUMMARY_STATISTICS = "neural_summary_statistics"
    BEHAVIOR_TRANSITION_SUMMARY = "behavior_transition_summary"
    TASK_STATE_LABELS = "task_state_labels"
    SESSION_QC_VECTOR = "session_qc_vector"
    FIRING_RATE_HISTOGRAM = "firing_rate_histogram"
    SPIKE_TRAIN_STATISTICS = "spike_train_statistics"
    LFP_POWER_SPECTRUM = "lfp_power_spectrum"
    CALCIUM_TRACE_SUMMARY = "calcium_trace_summary"
    BEHAVIOR_TRANSITION_MATRIX = "behavior_transition_matrix"
    TASK_STATE_SEQUENCE = "task_state_sequence"
    TRIAL_OUTCOME_DISTRIBUTION = "trial_outcome_distribution"
    NEURAL_EMBEDDING = "neural_embedding"
    BEHAVIOR_EMBEDDING = "behavior_embedding"


@dataclass(frozen=True)
class FeatureSummary:
    """A fixed-length feature summary for one dataset/session view."""

    feature_type: FeatureType
    dimensions: int
    values: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    extraction_method: str = "metadata_summary"
    extraction_version: str = "0.1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_type": self.feature_type.value,
            "dimensions": self.dimensions,
            "values": self.values,
            "metadata": self.metadata,
            "extraction_method": self.extraction_method,
            "extraction_version": self.extraction_version,
        }


@dataclass
class SessionFeatures:
    """Latent feature summaries tied to a dataset session."""

    dataset_id: str
    session_id: str
    features: list[FeatureSummary] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    warnings: list[str] = field(default_factory=list)

    @property
    def has_neural_features(self) -> bool:
        neural_types = {
            FeatureType.NEURAL_SUMMARY_STATISTICS,
            FeatureType.FIRING_RATE_HISTOGRAM,
            FeatureType.SPIKE_TRAIN_STATISTICS,
            FeatureType.LFP_POWER_SPECTRUM,
            FeatureType.CALCIUM_TRACE_SUMMARY,
            FeatureType.NEURAL_EMBEDDING,
        }
        return any(feature.feature_type in neural_types for feature in self.features)

    @property
    def has_behavior_features(self) -> bool:
        behavior_types = {
            FeatureType.EVENT_HISTOGRAM,
            FeatureType.BEHAVIOR_TRANSITION_SUMMARY,
            FeatureType.TASK_STATE_LABELS,
            FeatureType.BEHAVIOR_TRANSITION_MATRIX,
            FeatureType.TASK_STATE_SEQUENCE,
            FeatureType.TRIAL_OUTCOME_DISTRIBUTION,
            FeatureType.BEHAVIOR_EMBEDDING,
        }
        return any(feature.feature_type in behavior_types for feature in self.features)

    def vector(self) -> list[float]:
        values: list[float] = []
        for feature in self.features:
            values.extend(feature.values)
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "features": [feature.to_dict() for feature in self.features],
            "extracted_at": self.extracted_at.isoformat(),
            "warnings": self.warnings,
            "has_neural_features": self.has_neural_features,
            "has_behavior_features": self.has_behavior_features,
        }


@dataclass
class LatentIndex:
    """Manifest for an in-memory or future vector-backed latent index."""

    name: str
    feature_type: FeatureType
    embedding_dim: int
    num_sessions: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    index_backend: str = "in_memory"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "feature_type": self.feature_type.value,
            "embedding_dim": self.embedding_dim,
            "num_sessions": self.num_sessions,
            "created_at": self.created_at.isoformat(),
            "index_backend": self.index_backend,
        }


@dataclass
class LatentSearchResult:
    """A session-level match from latent similarity search."""

    dataset_id: str
    session_id: str
    similarity_score: float
    matched_feature_types: list[FeatureType] = field(default_factory=list)
    why_similar: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "similarity_score": self.similarity_score,
            "matched_feature_types": [
                feature_type.value for feature_type in self.matched_feature_types
            ],
            "why_similar": self.why_similar,
        }
