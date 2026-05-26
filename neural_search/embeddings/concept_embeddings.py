"""Dense embeddings for neuroscience concepts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neural_search.embeddings.index import cosine_similarity


@dataclass
class ConceptEmbedding:
    """Dense embedding for a neuroscience concept."""

    concept_id: str  # e.g., "task:reversal_learning"
    concept_type: str  # task, modality, behavior, analysis, region
    label: str
    embedding: list[float]  # 128D for major types, 64D for minor
    model_version: str
    aliases: list[str] = field(default_factory=list)
    parent_concepts: list[str] = field(default_factory=list)
    child_concepts: list[str] = field(default_factory=list)
    definition: str = ""
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "concept_id": self.concept_id,
            "concept_type": self.concept_type,
            "label": self.label,
            "embedding": self.embedding,
            "model_version": self.model_version,
            "aliases": self.aliases,
            "parent_concepts": self.parent_concepts,
            "child_concepts": self.child_concepts,
            "definition": self.definition,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConceptEmbedding:
        """Create from dictionary."""
        return cls(
            concept_id=data["concept_id"],
            concept_type=data["concept_type"],
            label=data["label"],
            embedding=data["embedding"],
            model_version=data.get("model_version", "unknown"),
            aliases=data.get("aliases", []),
            parent_concepts=data.get("parent_concepts", []),
            child_concepts=data.get("child_concepts", []),
            definition=data.get("definition", ""),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ConceptSimilarity:
    """Result of concept similarity search."""

    concept_id: str
    label: str
    concept_type: str
    similarity: float


class ConceptEmbeddingIndex:
    """Index for fast concept similarity lookup."""

    def __init__(self, embeddings: list[ConceptEmbedding]):
        """Initialize index from embeddings.

        Args:
            embeddings: List of concept embeddings
        """
        self.embeddings = embeddings
        self.by_id: dict[str, ConceptEmbedding] = {
            e.concept_id: e for e in embeddings
        }
        self.by_type: dict[str, list[ConceptEmbedding]] = {}
        self.by_label: dict[str, ConceptEmbedding] = {}

        # Index by type
        for emb in embeddings:
            if emb.concept_type not in self.by_type:
                self.by_type[emb.concept_type] = []
            self.by_type[emb.concept_type].append(emb)

            # Index by label (case-insensitive)
            self.by_label[emb.label.lower()] = emb
            for alias in emb.aliases:
                self.by_label[alias.lower()] = emb

        # Build embedding matrix per type
        self._embedding_matrices: dict[str, np.ndarray] = {}
        self._type_concept_ids: dict[str, list[str]] = {}

        for concept_type, type_embeddings in self.by_type.items():
            if type_embeddings:
                self._embedding_matrices[concept_type] = np.array(
                    [e.embedding for e in type_embeddings]
                )
                self._type_concept_ids[concept_type] = [
                    e.concept_id for e in type_embeddings
                ]

    def get(self, concept_id: str) -> ConceptEmbedding | None:
        """Get embedding by concept ID."""
        return self.by_id.get(concept_id)

    def get_by_label(self, label: str) -> ConceptEmbedding | None:
        """Get embedding by label or alias."""
        return self.by_label.get(label.lower())

    def get_embedding_vector(self, concept_id: str) -> np.ndarray | None:
        """Get embedding vector for a concept."""
        emb = self.by_id.get(concept_id)
        if emb is None:
            return None
        return np.array(emb.embedding)

    def find_similar(
        self,
        concept_id: str,
        concept_type: str | None = None,
        k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[ConceptSimilarity]:
        """Find concepts similar to the given concept.

        Args:
            concept_id: Source concept ID
            concept_type: Optional type filter
            k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar concepts sorted by similarity
        """
        source = self.by_id.get(concept_id)
        if source is None:
            return []

        query_vec = np.array(source.embedding)

        # Determine which types to search
        if concept_type is not None:
            search_types = [concept_type] if concept_type in self.by_type else []
        else:
            search_types = list(self.by_type.keys())

        results: list[ConceptSimilarity] = []

        for search_type in search_types:
            if search_type not in self._embedding_matrices:
                continue

            matrix = self._embedding_matrices[search_type]
            concept_ids = self._type_concept_ids[search_type]

            # Handle dimension mismatch
            if matrix.shape[1] != len(query_vec):
                continue

            # Compute similarities
            similarities = np.dot(matrix, query_vec) / (
                np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec) + 1e-9
            )

            for idx, sim in enumerate(similarities):
                cid = concept_ids[idx]
                if cid == concept_id:
                    continue  # Skip self
                if sim >= min_similarity:
                    emb = self.by_id[cid]
                    results.append(
                        ConceptSimilarity(
                            concept_id=cid,
                            label=emb.label,
                            concept_type=emb.concept_type,
                            similarity=float(sim),
                        )
                    )

        # Sort by similarity and limit
        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]

    def find_similar_to_vector(
        self,
        query_vec: np.ndarray,
        concept_type: str,
        k: int = 10,
        min_similarity: float = 0.5,
    ) -> list[ConceptSimilarity]:
        """Find concepts similar to a query vector.

        Args:
            query_vec: Query embedding vector
            concept_type: Type of concepts to search
            k: Number of results
            min_similarity: Minimum similarity threshold

        Returns:
            List of similar concepts
        """
        if concept_type not in self._embedding_matrices:
            return []

        matrix = self._embedding_matrices[concept_type]
        concept_ids = self._type_concept_ids[concept_type]

        # Handle dimension mismatch by truncating or padding
        if matrix.shape[1] != len(query_vec):
            if len(query_vec) > matrix.shape[1]:
                query_vec = query_vec[: matrix.shape[1]]
            else:
                query_vec = np.pad(
                    query_vec, (0, matrix.shape[1] - len(query_vec))
                )

        # Compute similarities
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-9:
            return []

        similarities = np.dot(matrix, query_vec) / (
            np.linalg.norm(matrix, axis=1) * query_norm + 1e-9
        )

        results: list[ConceptSimilarity] = []
        for idx, sim in enumerate(similarities):
            if sim >= min_similarity:
                cid = concept_ids[idx]
                emb = self.by_id[cid]
                results.append(
                    ConceptSimilarity(
                        concept_id=cid,
                        label=emb.label,
                        concept_type=emb.concept_type,
                        similarity=float(sim),
                    )
                )

        results.sort(key=lambda x: x.similarity, reverse=True)
        return results[:k]

    def compute_similarity(
        self,
        concept_a: str,
        concept_b: str,
    ) -> float:
        """Compute similarity between two concepts.

        Args:
            concept_a: First concept ID
            concept_b: Second concept ID

        Returns:
            Cosine similarity (0-1)
        """
        emb_a = self.by_id.get(concept_a)
        emb_b = self.by_id.get(concept_b)

        if emb_a is None or emb_b is None:
            return 0.0

        vec_a = emb_a.embedding
        vec_b = emb_b.embedding

        # Handle dimension mismatch
        if len(vec_a) != len(vec_b):
            return 0.0

        return cosine_similarity(vec_a, vec_b)

    @property
    def concept_types(self) -> list[str]:
        """Get all concept types in the index."""
        return list(self.by_type.keys())

    def __len__(self) -> int:
        """Return number of concepts."""
        return len(self.embeddings)


def read_concept_embeddings(path: str | Path) -> list[ConceptEmbedding]:
    """Read concept embeddings from JSONL file.

    Args:
        path: Path to JSONL file

    Returns:
        List of concept embeddings
    """
    path = Path(path)
    if not path.exists():
        return []

    embeddings = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                embeddings.append(ConceptEmbedding.from_dict(data))

    return embeddings


def write_concept_embeddings(
    embeddings: list[ConceptEmbedding],
    path: str | Path,
) -> None:
    """Write concept embeddings to JSONL file.

    Args:
        embeddings: List of embeddings
        path: Output path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for emb in embeddings:
            f.write(json.dumps(emb.to_dict()) + "\n")


def load_concept_index(path: str | Path) -> ConceptEmbeddingIndex:
    """Load concept embedding index from file.

    Args:
        path: Path to JSONL file

    Returns:
        ConceptEmbeddingIndex
    """
    embeddings = read_concept_embeddings(path)
    return ConceptEmbeddingIndex(embeddings)
