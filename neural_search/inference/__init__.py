"""Provider-neutral inference infrastructure for Neural Search."""

from neural_search.inference.registry import InferenceRegistry
from neural_search.inference.schemas import (
    EmbeddingRequest,
    EmbeddingResult,
    InferenceCapability,
    InferenceMessage,
    InferenceRequest,
    InferenceResult,
    ModelProfile,
    ProviderSettings,
    RankedPassage,
    RerankRequest,
    RerankResult,
    RunManifest,
)
from neural_search.inference.service import InferenceService

__all__ = [
    "EmbeddingRequest",
    "EmbeddingResult",
    "InferenceCapability",
    "InferenceMessage",
    "InferenceRegistry",
    "InferenceRequest",
    "InferenceResult",
    "InferenceService",
    "ModelProfile",
    "ProviderSettings",
    "RankedPassage",
    "RerankRequest",
    "RerankResult",
    "RunManifest",
]
