"""Build multi-modal fingerprints for datasets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from neural_search.embeddings.fingerprint import DatasetFingerprint, write_fingerprints
from neural_search.embeddings.hashing import HashingEmbeddingProvider

if TYPE_CHECKING:
    from collections.abc import Iterable

    from neural_search.schemas import NormalizedDatasetRecord


class DatasetFingerprintBuilder:
    """Build multi-modal fingerprints for datasets.

    Combines text embeddings (from title/description) with concept
    embeddings (from tasks, modalities, regions) into a unified
    fingerprint for similarity search.
    """

    def __init__(
        self,
        text_model: str = "hashing",
        concept_dim: int = 64,
        combined_dim: int = 256,
        fusion_method: str = "concatenate",
    ):
        """Initialize fingerprint builder.

        Args:
            text_model: Model for text embeddings ("hashing" or model name)
            concept_dim: Dimension for concept embeddings
            combined_dim: Target dimension for combined embedding
            fusion_method: How to combine embeddings ("concatenate" or "weighted")
        """
        self.text_model = text_model
        self.concept_dim = concept_dim
        self.combined_dim = combined_dim
        self.fusion_method = fusion_method

        # Initialize providers
        if text_model == "hashing":
            self.text_encoder = HashingEmbeddingProvider(dimensions=combined_dim)
        else:
            try:
                from neural_search.embeddings import (
                    SentenceTransformerEmbeddingProvider,
                )
                self.text_encoder = SentenceTransformerEmbeddingProvider(text_model)
            except RuntimeError:
                # Fallback to hashing
                self.text_encoder = HashingEmbeddingProvider(dimensions=combined_dim)

        self.concept_encoder = HashingEmbeddingProvider(dimensions=concept_dim)

        self.model_version = f"fingerprint-{text_model}-{concept_dim}d"

    def _extract_labels(self, labels: list[Any]) -> list[str]:
        """Extract string labels from label objects."""
        result = []
        for label in labels:
            if hasattr(label, "label"):
                result.append(label.label)
            elif hasattr(label, "id"):
                result.append(label.id)
            elif isinstance(label, str):
                result.append(label)
        return result

    def _build_text(self, record: NormalizedDatasetRecord) -> str:
        """Build combined text for text embedding."""
        parts = [record.title]
        if record.description:
            parts.append(record.description)

        # Add task names
        task_labels = self._extract_labels(record.tasks)
        if task_labels:
            parts.append("Tasks: " + ", ".join(task_labels))

        # Add modality names
        modality_labels = self._extract_labels(record.modalities)
        if modality_labels:
            parts.append("Modalities: " + ", ".join(modality_labels))

        return " ".join(parts)

    def _fuse_embeddings(
        self,
        text_emb: list[float],
        task_emb: list[float],
        modality_emb: list[float],
        region_emb: list[float],
    ) -> list[float]:
        """Fuse component embeddings into combined representation."""
        if self.fusion_method == "concatenate":
            # Concatenate and project to target dimension
            combined = np.concatenate([
                np.array(text_emb),
                np.array(task_emb),
                np.array(modality_emb),
                np.array(region_emb),
            ])

            # Project to target dimension if needed
            if len(combined) > self.combined_dim:
                # Simple truncation (PCA would be better for production)
                combined = combined[:self.combined_dim]
            elif len(combined) < self.combined_dim:
                # Pad with zeros
                combined = np.pad(
                    combined,
                    (0, self.combined_dim - len(combined)),
                )

            # L2 normalize
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

            return combined.tolist()

        elif self.fusion_method == "weighted":
            # Weighted average of normalized embeddings
            weights = [0.4, 0.25, 0.20, 0.15]
            embeddings = [text_emb, task_emb, modality_emb, region_emb]

            # Ensure all embeddings have same dimension
            min_dim = min(len(e) for e in embeddings)
            truncated = [np.array(e[:min_dim]) for e in embeddings]

            combined = sum(w * e for w, e in zip(weights, truncated, strict=False))

            # L2 normalize
            norm = np.linalg.norm(combined)
            if norm > 0:
                combined = combined / norm

            return combined.tolist()

        return text_emb  # Fallback

    def build_fingerprint(
        self,
        record: NormalizedDatasetRecord,
    ) -> DatasetFingerprint:
        """Generate fingerprint for a single dataset.

        Args:
            record: Normalized dataset record

        Returns:
            DatasetFingerprint with all component embeddings
        """
        # Extract labels
        task_labels = self._extract_labels(record.tasks)
        modality_labels = self._extract_labels(record.modalities)
        region_labels = self._extract_labels(record.brain_regions)

        # Build text embedding
        text = self._build_text(record)
        text_embedding = self.text_encoder.embed_text(text)

        # Build concept embeddings
        task_text = " ".join(task_labels) if task_labels else "unknown"
        task_embedding = self.concept_encoder.embed_text(task_text)

        modality_text = " ".join(modality_labels) if modality_labels else "unknown"
        modality_embedding = self.concept_encoder.embed_text(modality_text)

        region_text = " ".join(region_labels) if region_labels else "unknown"
        region_embedding = self.concept_encoder.embed_text(region_text)

        # Fuse into combined embedding
        combined_embedding = self._fuse_embeddings(
            text_embedding,
            task_embedding,
            modality_embedding,
            region_embedding,
        )

        return DatasetFingerprint(
            dataset_id=record.dataset_id,
            text_embedding=text_embedding,
            task_embedding=task_embedding,
            modality_embedding=modality_embedding,
            region_embedding=region_embedding,
            combined_embedding=combined_embedding,
            model_version=self.model_version,
            title=record.title,
            source=record.source,
            task_labels=task_labels,
            modality_labels=modality_labels,
            region_labels=region_labels,
        )

    def build_fingerprints(
        self,
        records: Iterable[NormalizedDatasetRecord],
    ) -> list[DatasetFingerprint]:
        """Generate fingerprints for multiple datasets.

        Args:
            records: Iterable of normalized dataset records

        Returns:
            List of DatasetFingerprint
        """
        return [self.build_fingerprint(record) for record in records]


def build_fingerprints_from_corpus(
    corpus_path: str,
    output_path: str,
    text_model: str = "hashing",
) -> list[DatasetFingerprint]:
    """Build fingerprints from a normalized corpus file.

    Args:
        corpus_path: Path to normalized corpus JSONL
        output_path: Path for output fingerprints JSONL
        text_model: Model for text embeddings

    Returns:
        List of generated fingerprints
    """
    from neural_search.normalized import load_normalized_records
    from neural_search.schemas import NormalizedDatasetRecord

    records = load_normalized_records(corpus_path)
    datasets = [r for r in records if isinstance(r, NormalizedDatasetRecord)]

    builder = DatasetFingerprintBuilder(text_model=text_model)
    fingerprints = builder.build_fingerprints(datasets)

    write_fingerprints(fingerprints, output_path)

    return fingerprints
