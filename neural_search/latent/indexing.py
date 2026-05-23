"""Indexing utilities for latent neural/behavioral embeddings.

This module provides placeholder implementations for building and
managing vector indices of latent features. Future implementations
will integrate with vector databases like pgvector, FAISS, or Pinecone.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from neural_search.latent.schema import (
    FeatureType,
    LatentIndex,
    SessionFeatures,
)


class LatentIndexManager:
    """Manager for latent feature indices.

    Placeholder implementation that stores features in memory.
    Future versions will integrate with vector databases.
    """

    def __init__(self) -> None:
        """Initialize the index manager."""
        self._indices: dict[str, LatentIndex] = {}
        self._features: dict[str, list[SessionFeatures]] = {}

    def create_index(
        self,
        name: str,
        feature_type: FeatureType,
        embedding_dim: int,
    ) -> LatentIndex:
        """Create a new latent feature index.

        Args:
            name: Unique name for the index
            feature_type: Type of features to index
            embedding_dim: Dimensionality of feature vectors

        Returns:
            Created LatentIndex object
        """
        index = LatentIndex(
            name=name,
            feature_type=feature_type,
            embedding_dim=embedding_dim,
            index_backend="in_memory_placeholder",
        )
        self._indices[name] = index
        self._features[name] = []
        return index

    def add_session(self, index_name: str, session_features: SessionFeatures) -> bool:
        """Add a session's features to an index.

        Args:
            index_name: Name of the target index
            session_features: Features to add

        Returns:
            True if successfully added
        """
        if index_name not in self._indices:
            return False

        self._features[index_name].append(session_features)
        self._indices[index_name].num_sessions += 1
        return True

    def get_index(self, name: str) -> LatentIndex | None:
        """Get an index by name."""
        return self._indices.get(name)

    def list_indices(self) -> list[LatentIndex]:
        """List all available indices."""
        return list(self._indices.values())

    def get_index_stats(self, name: str) -> dict[str, Any]:
        """Get statistics about an index.

        Args:
            name: Index name

        Returns:
            Dictionary of index statistics
        """
        index = self._indices.get(name)
        if index is None:
            return {"error": f"Index {name} not found"}

        sessions = self._features.get(name, [])

        return {
            "name": name,
            "feature_type": index.feature_type.value,
            "embedding_dim": index.embedding_dim,
            "num_sessions": len(sessions),
            "num_datasets": len({s.dataset_id for s in sessions}),
            "created_at": index.created_at.isoformat(),
            "backend": index.index_backend,
        }

    def delete_index(self, name: str) -> bool:
        """Delete an index.

        Args:
            name: Index name to delete

        Returns:
            True if deleted, False if not found
        """
        if name not in self._indices:
            return False

        del self._indices[name]
        del self._features[name]
        return True


def build_default_indices(
    sessions: list[SessionFeatures],
) -> dict[str, LatentIndex]:
    """Build a default set of indices from session features.

    Creates indices for the most common feature types found in the sessions.

    Args:
        sessions: List of session features to index

    Returns:
        Dictionary of created indices by name
    """
    manager = LatentIndexManager()
    created: dict[str, LatentIndex] = {}

    # Determine which feature types are present
    feature_dims: dict[FeatureType, int] = {}
    for session in sessions:
        for feature in session.features:
            if feature.feature_type not in feature_dims:
                feature_dims[feature.feature_type] = feature.dimensions

    # Create an index for each feature type
    for feature_type, dim in feature_dims.items():
        index_name = f"{feature_type.value}_index"
        index = manager.create_index(index_name, feature_type, dim)
        created[index_name] = index

        # Add all sessions with this feature type
        for session in sessions:
            if any(f.feature_type == feature_type for f in session.features):
                manager.add_session(index_name, session)

    return created


def export_index_manifest(indices: dict[str, LatentIndex]) -> dict[str, Any]:
    """Export a manifest of all indices for persistence.

    Args:
        indices: Dictionary of indices to export

    Returns:
        JSON-serializable manifest
    """
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "num_indices": len(indices),
        "indices": [idx.to_dict() for idx in indices.values()],
    }
