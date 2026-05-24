"""Multi-modal dataset fingerprints for similarity search."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DatasetFingerprint:
    """Multi-modal embedding for dataset similarity.

    Combines text, task, modality, and structural embeddings
    into a unified representation for fast similarity search.
    """

    dataset_id: str
    text_embedding: list[float]      # From title + description (384D typical)
    task_embedding: list[float]      # From task labels (64D)
    modality_embedding: list[float]  # From modality labels (64D)
    region_embedding: list[float]    # From brain region labels (64D)
    combined_embedding: list[float]  # Fused representation

    model_version: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    # Optional metadata
    title: str = ""
    source: str = ""
    task_labels: list[str] = field(default_factory=list)
    modality_labels: list[str] = field(default_factory=list)
    region_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "dataset_id": self.dataset_id,
            "text_embedding": self.text_embedding,
            "task_embedding": self.task_embedding,
            "modality_embedding": self.modality_embedding,
            "region_embedding": self.region_embedding,
            "combined_embedding": self.combined_embedding,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "title": self.title,
            "source": self.source,
            "task_labels": self.task_labels,
            "modality_labels": self.modality_labels,
            "region_labels": self.region_labels,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetFingerprint:
        """Create from dict."""
        return cls(
            dataset_id=data["dataset_id"],
            text_embedding=data["text_embedding"],
            task_embedding=data["task_embedding"],
            modality_embedding=data["modality_embedding"],
            region_embedding=data["region_embedding"],
            combined_embedding=data["combined_embedding"],
            model_version=data["model_version"],
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            title=data.get("title", ""),
            source=data.get("source", ""),
            task_labels=data.get("task_labels", []),
            modality_labels=data.get("modality_labels", []),
            region_labels=data.get("region_labels", []),
        )

    @property
    def text_dim(self) -> int:
        return len(self.text_embedding)

    @property
    def concept_dim(self) -> int:
        return len(self.task_embedding)

    @property
    def combined_dim(self) -> int:
        return len(self.combined_embedding)


def cosine_similarity_np(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


@dataclass
class FingerprintSimilarity:
    """Similarity breakdown between two fingerprints."""

    source_id: str
    target_id: str
    combined_similarity: float
    text_similarity: float
    task_similarity: float
    modality_similarity: float
    region_similarity: float

    @property
    def weighted_similarity(self) -> float:
        """Weighted combination of component similarities."""
        return (
            0.35 * self.text_similarity
            + 0.25 * self.task_similarity
            + 0.20 * self.modality_similarity
            + 0.20 * self.region_similarity
        )


def compute_fingerprint_similarity(
    source: DatasetFingerprint,
    target: DatasetFingerprint,
) -> FingerprintSimilarity:
    """Compute multi-modal similarity between fingerprints."""
    return FingerprintSimilarity(
        source_id=source.dataset_id,
        target_id=target.dataset_id,
        combined_similarity=cosine_similarity_np(
            source.combined_embedding, target.combined_embedding
        ),
        text_similarity=cosine_similarity_np(
            source.text_embedding, target.text_embedding
        ),
        task_similarity=cosine_similarity_np(
            source.task_embedding, target.task_embedding
        ),
        modality_similarity=cosine_similarity_np(
            source.modality_embedding, target.modality_embedding
        ),
        region_similarity=cosine_similarity_np(
            source.region_embedding, target.region_embedding
        ),
    )


def write_fingerprints(
    fingerprints: list[DatasetFingerprint],
    path: str | Path,
) -> Path:
    """Write fingerprints to JSONL file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for fp in fingerprints:
            f.write(json.dumps(fp.to_dict(), sort_keys=True))
            f.write("\n")

    return output


def read_fingerprints(path: str | Path) -> list[DatasetFingerprint]:
    """Read fingerprints from JSONL file."""
    fingerprints = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                fingerprints.append(DatasetFingerprint.from_dict(data))
    return fingerprints
