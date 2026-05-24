"""In-memory fingerprint index for fast similarity search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from neural_search.embeddings.fingerprint import (
    DatasetFingerprint,
    FingerprintSimilarity,
    compute_fingerprint_similarity,
    read_fingerprints,
)


@dataclass
class SimilarDataset:
    """A similar dataset with similarity score."""

    dataset_id: str
    similarity: float
    title: str
    breakdown: FingerprintSimilarity


class FingerprintIndex:
    """In-memory index for fast nearest-neighbor fingerprint search.

    Provides efficient similarity search over precomputed
    dataset fingerprints using numpy vectorized operations.
    """

    def __init__(self, fingerprints: list[DatasetFingerprint]):
        """Initialize index from list of fingerprints.

        Args:
            fingerprints: List of DatasetFingerprint objects
        """
        self.fingerprints = {fp.dataset_id: fp for fp in fingerprints}
        self._ids = list(self.fingerprints.keys())
        self._build_index()

    def _build_index(self) -> None:
        """Build numpy arrays for fast similarity computation."""
        if not self._ids:
            self._combined_matrix = np.array([])
            self._text_matrix = np.array([])
            self._task_matrix = np.array([])
            self._modality_matrix = np.array([])
            self._region_matrix = np.array([])
            return

        # Stack embeddings into matrices for vectorized operations
        self._combined_matrix = np.array([
            self.fingerprints[id_].combined_embedding
            for id_ in self._ids
        ])
        self._text_matrix = np.array([
            self.fingerprints[id_].text_embedding
            for id_ in self._ids
        ])
        self._task_matrix = np.array([
            self.fingerprints[id_].task_embedding
            for id_ in self._ids
        ])
        self._modality_matrix = np.array([
            self.fingerprints[id_].modality_embedding
            for id_ in self._ids
        ])
        self._region_matrix = np.array([
            self.fingerprints[id_].region_embedding
            for id_ in self._ids
        ])

        # Normalize matrices for cosine similarity
        self._combined_matrix = self._normalize_matrix(self._combined_matrix)
        self._text_matrix = self._normalize_matrix(self._text_matrix)
        self._task_matrix = self._normalize_matrix(self._task_matrix)
        self._modality_matrix = self._normalize_matrix(self._modality_matrix)
        self._region_matrix = self._normalize_matrix(self._region_matrix)

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """L2 normalize rows of a matrix."""
        if matrix.size == 0:
            return matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return matrix / norms

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, dataset_id: str) -> bool:
        return dataset_id in self.fingerprints

    def get(self, dataset_id: str) -> DatasetFingerprint | None:
        """Get fingerprint by dataset ID."""
        return self.fingerprints.get(dataset_id)

    def find_similar(
        self,
        query_fingerprint: DatasetFingerprint,
        k: int = 10,
        min_similarity: float = 0.0,
        exclude_self: bool = True,
        similarity_type: Literal["combined", "weighted", "text"] = "combined",
    ) -> list[SimilarDataset]:
        """Find k most similar datasets to query.

        Args:
            query_fingerprint: Query fingerprint
            k: Maximum number of results
            min_similarity: Minimum similarity threshold
            exclude_self: Exclude the query dataset from results
            similarity_type: Which similarity to use for ranking

        Returns:
            List of SimilarDataset sorted by similarity descending
        """
        if len(self._ids) == 0:
            return []

        # Compute similarities using vectorized operations
        query_combined = np.array(query_fingerprint.combined_embedding)
        query_combined = query_combined / (np.linalg.norm(query_combined) or 1)

        combined_sims = self._combined_matrix @ query_combined

        # Compute component similarities if needed
        if similarity_type == "weighted":
            query_text = np.array(query_fingerprint.text_embedding)
            query_task = np.array(query_fingerprint.task_embedding)
            query_mod = np.array(query_fingerprint.modality_embedding)
            query_reg = np.array(query_fingerprint.region_embedding)

            # Normalize queries
            query_text = query_text / (np.linalg.norm(query_text) or 1)
            query_task = query_task / (np.linalg.norm(query_task) or 1)
            query_mod = query_mod / (np.linalg.norm(query_mod) or 1)
            query_reg = query_reg / (np.linalg.norm(query_reg) or 1)

            text_sims = self._text_matrix @ query_text
            task_sims = self._task_matrix @ query_task
            mod_sims = self._modality_matrix @ query_mod
            reg_sims = self._region_matrix @ query_reg

            # Weighted combination
            similarities = (
                0.35 * text_sims
                + 0.25 * task_sims
                + 0.20 * mod_sims
                + 0.20 * reg_sims
            )
        elif similarity_type == "text":
            query_text = np.array(query_fingerprint.text_embedding)
            query_text = query_text / (np.linalg.norm(query_text) or 1)
            similarities = self._text_matrix @ query_text
        else:
            similarities = combined_sims

        # Build results
        results = []
        for i, (dataset_id, sim) in enumerate(zip(self._ids, similarities)):
            if exclude_self and dataset_id == query_fingerprint.dataset_id:
                continue
            if sim < min_similarity:
                continue

            fp = self.fingerprints[dataset_id]
            breakdown = compute_fingerprint_similarity(query_fingerprint, fp)

            results.append(SimilarDataset(
                dataset_id=dataset_id,
                similarity=float(sim),
                title=fp.title,
                breakdown=breakdown,
            ))

        # Sort by similarity descending
        results.sort(key=lambda x: x.similarity, reverse=True)

        return results[:k]

    def find_similar_by_id(
        self,
        dataset_id: str,
        k: int = 10,
        min_similarity: float = 0.0,
        similarity_type: Literal["combined", "weighted", "text"] = "combined",
    ) -> list[SimilarDataset]:
        """Find similar datasets given a dataset ID.

        Args:
            dataset_id: Source dataset ID
            k: Maximum number of results
            min_similarity: Minimum similarity threshold
            similarity_type: Which similarity to use

        Returns:
            List of SimilarDataset (excluding the source)
        """
        query_fp = self.fingerprints.get(dataset_id)
        if query_fp is None:
            return []

        return self.find_similar(
            query_fp,
            k=k,
            min_similarity=min_similarity,
            exclude_self=True,
            similarity_type=similarity_type,
        )

    def batch_find_similar(
        self,
        dataset_ids: list[str],
        k: int = 5,
        min_similarity: float = 0.3,
    ) -> dict[str, list[SimilarDataset]]:
        """Find similar datasets for multiple sources.

        Args:
            dataset_ids: List of source dataset IDs
            k: Results per source
            min_similarity: Minimum threshold

        Returns:
            Dict mapping source ID to similar datasets
        """
        results = {}
        for dataset_id in dataset_ids:
            results[dataset_id] = self.find_similar_by_id(
                dataset_id,
                k=k,
                min_similarity=min_similarity,
            )
        return results


def load_fingerprint_index(path: str | Path) -> FingerprintIndex | None:
    """Load fingerprint index from JSONL file.

    Args:
        path: Path to fingerprints JSONL

    Returns:
        FingerprintIndex or None if file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        return None

    fingerprints = read_fingerprints(path)
    return FingerprintIndex(fingerprints)
