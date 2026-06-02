"""Field-specific semantic scoring for main retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from neural_search.embeddings import (
    FieldEmbeddingRecord,
    HashingEmbeddingProvider,
    cosine_similarity,
    read_field_embedding_cache,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class EmbeddingProviderProtocol(Protocol):
    """Protocol for embedding providers."""

    def embed_text(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]: ...

DEFAULT_FIELD_WEIGHTS = {
    "title": 0.18,
    "description": 0.16,
    "tasks": 0.16,
    "behavioral_events": 0.14,
    "modalities": 0.12,
    "brain_regions": 0.10,
    "analysis_goals": 0.12,
    "combined_scientific_summary": 0.20,
}


@dataclass(frozen=True)
class FieldSemanticScore:
    """Field semantic score plus explainable top field matches."""

    score: float
    matched_fields: list[str]


class FieldSemanticIndex:
    """Small in-memory index over precomputed field embedding records."""

    provider: EmbeddingProviderProtocol | None

    def __init__(self, records: list[FieldEmbeddingRecord]) -> None:
        self.records = records
        self._by_record_id: dict[str, list[FieldEmbeddingRecord]] = {}
        for record in records:
            if record.record_type != "dataset":
                continue
            aliases = {
                record.record_id,
                record.record_id.split(":")[-1],
            }
            if record.record_id.startswith("dataset:"):
                aliases.add(":".join(record.record_id.split(":")[1:]))
            for alias in aliases:
                self._by_record_id.setdefault(alias, []).append(record)
        self.provider = _provider_for_cache(records)
        self._provider_type = records[0].provider_name if records else "unknown"

    def score_dataset(
        self,
        dataset_id: str,
        query: str,
        *,
        field_weights: dict[str, float] | None = None,
    ) -> FieldSemanticScore:
        if not self.provider:
            return FieldSemanticScore(score=0.0, matched_fields=[])
        records = self._by_record_id.get(dataset_id, [])
        if not records:
            return FieldSemanticScore(score=0.0, matched_fields=[])

        weights = {**DEFAULT_FIELD_WEIGHTS, **(field_weights or {})}
        query_vector = self.provider.embed_text(query)
        scored_fields: list[tuple[str, float, float]] = []
        weighted_total = 0.0
        weight_total = 0.0
        for record in records:
            weight = float(weights.get(record.field_name, 0.05))
            similarity = max(0.0, cosine_similarity(query_vector, record.embedding))
            weighted_total += similarity * weight
            weight_total += weight
            scored_fields.append((record.field_name, similarity, weight))
        if weight_total == 0:
            return FieldSemanticScore(score=0.0, matched_fields=[])

        scored_fields.sort(key=lambda item: (item[1] * item[2], item[1]), reverse=True)
        matched_fields = [
            field_name
            for field_name, similarity, _weight in scored_fields[:3]
            if similarity > 0
        ]
        return FieldSemanticScore(
            score=round(min(weighted_total / weight_total, 1.0), 4),
            matched_fields=matched_fields,
        )


def _provider_for_cache(
    records: list[FieldEmbeddingRecord],
) -> EmbeddingProviderProtocol | None:
    """Create an embedding provider matching the cache metadata.

    Supports both hashing (deterministic) and sentence-transformer (neural) providers.
    """
    if not records:
        return None
    first = records[0]

    if first.provider_name == "hashing":
        return HashingEmbeddingProvider(
            dimensions=first.dimension,
            normalize_embeddings=first.normalize,
        )

    if first.provider_name == "sentence-transformer":
        try:
            from neural_search.embeddings import SentenceTransformerEmbeddingProvider

            return SentenceTransformerEmbeddingProvider(
                model_name=first.model_name,
                normalize=first.normalize,
            )
        except RuntimeError:
            # sentence-transformers not installed, fall back gracefully
            return None

    if first.provider_name == "bge-large":
        try:
            from neural_search.embeddings.dense_provider import DenseEmbeddingProvider

            return DenseEmbeddingProvider(normalize=first.normalize)
        except RuntimeError:
            # sentence-transformers not installed
            return None

    return None


@lru_cache(maxsize=4)
def load_field_semantic_index(path: str | None) -> FieldSemanticIndex | None:
    """Load a field embedding cache if available and CI-safe to query."""

    if not path:
        return None
    cache_path = Path(path)
    if not cache_path.exists():
        return None
    records = read_field_embedding_cache(cache_path)
    index = FieldSemanticIndex(records)
    if index.provider is None:
        return None
    return index


def field_semantic_score_for_result(
    index: FieldSemanticIndex | None,
    dataset_id: str,
    query: str,
    config: dict[str, Any] | None = None,
) -> FieldSemanticScore:
    """Return a safe field-semantic score for a dataset result."""

    if index is None:
        return FieldSemanticScore(score=0.0, matched_fields=[])
    return index.score_dataset(
        dataset_id,
        query,
        field_weights=(config or {}).get("field_weights"),
    )
