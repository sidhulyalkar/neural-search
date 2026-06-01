"""Enhanced semantic fingerprints with behavior and analysis dimensions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from neural_search.embeddings.concept_embeddings import ConceptEmbeddingIndex
from neural_search.embeddings.hashing import HashingEmbeddingProvider


@dataclass
class SemanticDatasetFingerprint:
    """Rich multi-modal fingerprint with behavioral + analysis dimensions.

    Extends the basic fingerprint with:
    - Behavioral event embedding (64D)
    - Analysis affordance embedding (64D)
    - Experimental design embedding (32D)
    - Embedding confidence score
    """

    # Required fields (no defaults) - must come first
    dataset_id: str
    text_embedding: list[float]      # Text (384D typical)
    task_embedding: list[float]      # Concepts (128D)
    modality_embedding: list[float]  # Modality (128D)
    behavior_embedding: list[float]  # Behavioral (64D)
    analysis_embedding: list[float]  # Analysis (64D)
    region_embedding: list[float]    # Brain regions (64D)
    design_embedding: list[float]    # Experimental design (32D)
    combined_embedding: list[float]  # Combined embedding

    # Optional fields with defaults
    behavior_labels: list[str] = field(default_factory=list)
    behavior_complexity: float = 0.5  # 0-1 score
    analysis_affordance_ids: list[str] = field(default_factory=list)
    region_labels: list[str] = field(default_factory=list)
    design_type: str = ""  # "2afc", "go_nogo", "free_behavior", etc.
    model_version: str = "semantic_v1"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    title: str = ""
    source: str = ""
    task_labels: list[str] = field(default_factory=list)
    modality_labels: list[str] = field(default_factory=list)
    embedding_confidence: float = 1.0  # How well-defined the dataset is
    missing_dimensions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "dataset_id": self.dataset_id,
            "text_embedding": self.text_embedding,
            "task_embedding": self.task_embedding,
            "modality_embedding": self.modality_embedding,
            "behavior_embedding": self.behavior_embedding,
            "behavior_labels": self.behavior_labels,
            "behavior_complexity": self.behavior_complexity,
            "analysis_embedding": self.analysis_embedding,
            "analysis_affordance_ids": self.analysis_affordance_ids,
            "region_embedding": self.region_embedding,
            "region_labels": self.region_labels,
            "design_embedding": self.design_embedding,
            "design_type": self.design_type,
            "combined_embedding": self.combined_embedding,
            "model_version": self.model_version,
            "created_at": self.created_at,
            "title": self.title,
            "source": self.source,
            "task_labels": self.task_labels,
            "modality_labels": self.modality_labels,
            "embedding_confidence": self.embedding_confidence,
            "missing_dimensions": self.missing_dimensions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticDatasetFingerprint:
        """Create from dict."""
        return cls(
            dataset_id=data["dataset_id"],
            text_embedding=data["text_embedding"],
            task_embedding=data["task_embedding"],
            modality_embedding=data["modality_embedding"],
            behavior_embedding=data.get("behavior_embedding", []),
            behavior_labels=data.get("behavior_labels", []),
            behavior_complexity=data.get("behavior_complexity", 0.5),
            analysis_embedding=data.get("analysis_embedding", []),
            analysis_affordance_ids=data.get("analysis_affordance_ids", []),
            region_embedding=data.get("region_embedding", []),
            region_labels=data.get("region_labels", []),
            design_embedding=data.get("design_embedding", []),
            design_type=data.get("design_type", ""),
            combined_embedding=data["combined_embedding"],
            model_version=data.get("model_version", "semantic_v1"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            title=data.get("title", ""),
            source=data.get("source", ""),
            task_labels=data.get("task_labels", []),
            modality_labels=data.get("modality_labels", []),
            embedding_confidence=data.get("embedding_confidence", 1.0),
            missing_dimensions=data.get("missing_dimensions", []),
        )

    @property
    def total_dimensions(self) -> int:
        """Total embedding dimensions."""
        return (
            len(self.text_embedding)
            + len(self.task_embedding)
            + len(self.modality_embedding)
            + len(self.behavior_embedding)
            + len(self.analysis_embedding)
            + len(self.region_embedding)
            + len(self.design_embedding)
        )


@dataclass
class SemanticSimilarity:
    """Detailed similarity breakdown between semantic fingerprints."""

    source_id: str
    target_id: str
    combined_similarity: float
    text_similarity: float
    task_similarity: float
    modality_similarity: float
    behavior_similarity: float
    analysis_similarity: float
    region_similarity: float
    design_similarity: float

    @property
    def weighted_similarity(self) -> float:
        """Weighted combination emphasizing task and behavior."""
        return (
            0.20 * self.text_similarity
            + 0.20 * self.task_similarity
            + 0.15 * self.modality_similarity
            + 0.15 * self.behavior_similarity
            + 0.15 * self.analysis_similarity
            + 0.10 * self.region_similarity
            + 0.05 * self.design_similarity
        )


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity."""
    if not a or not b or len(a) != len(b):
        return 0.0
    a_arr = np.array(a)
    b_arr = np.array(b)
    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def compute_semantic_similarity(
    source: SemanticDatasetFingerprint,
    target: SemanticDatasetFingerprint,
) -> SemanticSimilarity:
    """Compute multi-dimensional similarity between semantic fingerprints."""
    return SemanticSimilarity(
        source_id=source.dataset_id,
        target_id=target.dataset_id,
        combined_similarity=_cosine_sim(
            source.combined_embedding, target.combined_embedding
        ),
        text_similarity=_cosine_sim(source.text_embedding, target.text_embedding),
        task_similarity=_cosine_sim(source.task_embedding, target.task_embedding),
        modality_similarity=_cosine_sim(
            source.modality_embedding, target.modality_embedding
        ),
        behavior_similarity=_cosine_sim(
            source.behavior_embedding, target.behavior_embedding
        ),
        analysis_similarity=_cosine_sim(
            source.analysis_embedding, target.analysis_embedding
        ),
        region_similarity=_cosine_sim(
            source.region_embedding, target.region_embedding
        ),
        design_similarity=_cosine_sim(
            source.design_embedding, target.design_embedding
        ),
    )


class SemanticFingerprintBuilder:
    """Build semantic fingerprints with all dimensions."""

    def __init__(
        self,
        concept_index: ConceptEmbeddingIndex | None = None,
        text_dim: int = 256,
        concept_dim: int = 128,
        minor_dim: int = 64,
        design_dim: int = 32,
    ):
        """Initialize builder.

        Args:
            concept_index: Optional concept embedding index for semantic matching
            text_dim: Dimension for text embeddings
            concept_dim: Dimension for major concept embeddings (task, modality)
            minor_dim: Dimension for minor embeddings (behavior, analysis, region)
            design_dim: Dimension for design embeddings
        """
        self.concept_index = concept_index
        self.text_dim = text_dim
        self.concept_dim = concept_dim
        self.minor_dim = minor_dim
        self.design_dim = design_dim

        # Create hash providers for fallback
        self._text_hasher = HashingEmbeddingProvider(
            dimensions=text_dim, normalize_embeddings=True
        )
        self._concept_hasher = HashingEmbeddingProvider(
            dimensions=concept_dim, normalize_embeddings=True
        )
        self._minor_hasher = HashingEmbeddingProvider(
            dimensions=minor_dim, normalize_embeddings=True
        )
        self._design_hasher = HashingEmbeddingProvider(
            dimensions=design_dim, normalize_embeddings=True
        )

        # Behavior complexity patterns
        self._simple_behaviors = {"lick", "lever_press", "nose_poke", "running"}
        self._complex_behaviors = {
            "decision", "choice", "strategy", "planning",
            "working_memory", "attention", "inhibition"
        }

    def _embed_text(self, text: str) -> list[float]:
        """Embed text using hasher."""
        return self._text_hasher.embed_text(text)

    def _embed_concept(self, labels: list[str], concept_type: str) -> list[float]:
        """Embed concept labels, using index if available."""
        if not labels:
            return [0.0] * self.concept_dim

        # Try to use concept index for semantic embeddings
        if self.concept_index is not None:
            embeddings = []
            for label in labels:
                concept_id = f"{concept_type}:{label}"
                emb = self.concept_index.get(concept_id)
                if emb is None:
                    emb = self.concept_index.get_by_label(label)
                if emb is not None:
                    embeddings.append(emb.embedding)

            if embeddings:
                # Average the embeddings
                avg = np.mean(embeddings, axis=0)
                # Normalize
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg = avg / norm
                # Pad or truncate to concept_dim
                if len(avg) < self.concept_dim:
                    avg = np.pad(avg, (0, self.concept_dim - len(avg)))
                elif len(avg) > self.concept_dim:
                    avg = avg[:self.concept_dim]
                return avg.tolist()

        # Fallback to hashing
        text = " ".join(labels)
        return self._concept_hasher.embed_text(text)

    def _embed_minor(self, labels: list[str], concept_type: str) -> list[float]:
        """Embed minor concept labels (behavior, analysis, region)."""
        if not labels:
            return [0.0] * self.minor_dim

        # Try to use concept index
        if self.concept_index is not None:
            embeddings = []
            for label in labels:
                concept_id = f"{concept_type}:{label}"
                emb = self.concept_index.get(concept_id)
                if emb is None:
                    emb = self.concept_index.get_by_label(label)
                if emb is not None:
                    embeddings.append(emb.embedding)

            if embeddings:
                avg = np.mean(embeddings, axis=0)
                norm = np.linalg.norm(avg)
                if norm > 0:
                    avg = avg / norm
                # Adjust dimension
                if len(avg) < self.minor_dim:
                    avg = np.pad(avg, (0, self.minor_dim - len(avg)))
                elif len(avg) > self.minor_dim:
                    avg = avg[:self.minor_dim]
                return avg.tolist()

        # Fallback to hashing
        text = " ".join(labels)
        return self._minor_hasher.embed_text(text)

    def _compute_behavior_complexity(self, behaviors: list[str]) -> float:
        """Estimate behavioral complexity (0-1)."""
        if not behaviors:
            return 0.3  # Unknown complexity

        behavior_set = {b.lower().replace("_", " ") for b in behaviors}

        # Check for complex behaviors
        complex_count = sum(
            1 for b in self._complex_behaviors
            if any(b in beh for beh in behavior_set)
        )

        # Check for simple behaviors
        simple_count = sum(
            1 for b in self._simple_behaviors
            if any(b in beh for beh in behavior_set)
        )

        if complex_count > 0:
            return min(0.5 + complex_count * 0.15, 1.0)
        elif simple_count > 0:
            return 0.3
        else:
            return 0.5

    def _infer_design_type(self, tasks: list[str], behaviors: list[str]) -> str:
        """Infer experimental design type from task and behavior labels."""
        task_str = " ".join(t.lower() for t in tasks)
        behavior_str = " ".join(b.lower() for b in behaviors)
        combined = f"{task_str} {behavior_str}"

        if "2afc" in combined or "two alternative" in combined:
            return "2afc"
        elif "go no" in combined or "go_nogo" in combined:
            return "go_nogo"
        elif "reversal" in combined:
            return "reversal_learning"
        elif "delay" in combined or "discounting" in combined:
            return "delay_discounting"
        elif "bandit" in combined or "arm" in combined:
            return "multi_armed_bandit"
        elif "free" in combined or "open field" in combined:
            return "free_behavior"
        elif "pavlov" in combined or "conditioning" in combined:
            return "classical_conditioning"
        elif "operant" in combined:
            return "operant_conditioning"
        else:
            return "unknown"

    def _get_field(self, record: Any, field: str, default: Any = None) -> Any:
        """Get field from record, handling both dict and object access."""
        if isinstance(record, dict):
            return record.get(field, default)
        return getattr(record, field, default)

    def _extract_labels(self, items: list[Any]) -> list[str]:
        """Extract string labels from list of items (handles EvidenceLabel objects)."""
        labels = []
        for item in items:
            if isinstance(item, str):
                labels.append(item)
            elif hasattr(item, "id"):
                labels.append(str(item.id))
            elif hasattr(item, "label"):
                labels.append(str(item.label))
            else:
                labels.append(str(item))
        return labels

    def build_fingerprint(
        self,
        record: Any,
    ) -> SemanticDatasetFingerprint:
        """Build semantic fingerprint from normalized dataset record.

        Args:
            record: NormalizedDatasetRecord or dict with dataset info

        Returns:
            SemanticDatasetFingerprint
        """
        # Extract fields from record using helper
        dataset_id = self._get_field(record, "id") or self._get_field(record, "source_id", "unknown")

        title = self._get_field(record, "title", "") or ""
        description = self._get_field(record, "description", "") or ""
        source = self._get_field(record, "source", "") or ""

        tasks = self._extract_labels(self._get_field(record, "tasks", []) or [])
        modalities = self._extract_labels(self._get_field(record, "modalities", []) or [])
        behaviors = self._extract_labels(self._get_field(record, "behaviors", []) or [])
        behavioral_events = self._extract_labels(self._get_field(record, "behavioral_events", []) or [])
        regions = self._extract_labels(self._get_field(record, "brain_regions", []) or [])
        affordances = self._extract_labels(self._get_field(record, "analysis_affordances", []) or [])

        # Combine behaviors and behavioral events
        all_behaviors = list(set(behaviors + behavioral_events))

        # Track missing dimensions
        missing = []
        if not tasks:
            missing.append("tasks")
        if not modalities:
            missing.append("modalities")
        if not all_behaviors:
            missing.append("behaviors")
        if not affordances:
            missing.append("affordances")
        if not regions:
            missing.append("regions")

        # Build embeddings
        text = f"{title} {description}"
        text_embedding = self._embed_text(text)
        task_embedding = self._embed_concept(tasks, "task")
        modality_embedding = self._embed_concept(modalities, "modality")
        behavior_embedding = self._embed_minor(all_behaviors, "behavior")
        analysis_embedding = self._embed_minor(affordances, "analysis")
        region_embedding = self._embed_minor(regions, "region")

        # Infer design type
        design_type = self._infer_design_type(tasks, all_behaviors)
        design_embedding = self._design_hasher.embed_text(design_type)

        # Compute behavior complexity
        behavior_complexity = self._compute_behavior_complexity(all_behaviors)

        # Combine all embeddings
        combined = np.concatenate([
            text_embedding,
            task_embedding,
            modality_embedding,
            behavior_embedding,
            analysis_embedding,
            region_embedding,
            design_embedding,
        ])
        # Normalize combined
        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        combined_embedding = combined.tolist()

        # Compute embedding confidence
        present_count = 5 - len(missing)
        embedding_confidence = present_count / 5.0

        return SemanticDatasetFingerprint(
            dataset_id=dataset_id,
            text_embedding=text_embedding,
            task_embedding=task_embedding,
            modality_embedding=modality_embedding,
            behavior_embedding=behavior_embedding,
            behavior_labels=all_behaviors,
            behavior_complexity=behavior_complexity,
            analysis_embedding=analysis_embedding,
            analysis_affordance_ids=affordances,
            region_embedding=region_embedding,
            region_labels=regions,
            design_embedding=design_embedding,
            design_type=design_type,
            combined_embedding=combined_embedding,
            title=title,
            source=source,
            task_labels=tasks,
            modality_labels=modalities,
            embedding_confidence=embedding_confidence,
            missing_dimensions=missing,
        )

    def build_fingerprints(
        self,
        records: list[Any],
    ) -> list[SemanticDatasetFingerprint]:
        """Build fingerprints for multiple records."""
        return [self.build_fingerprint(r) for r in records]


def write_semantic_fingerprints(
    fingerprints: list[SemanticDatasetFingerprint],
    path: str | Path,
) -> Path:
    """Write semantic fingerprints to JSONL file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as f:
        for fp in fingerprints:
            f.write(json.dumps(fp.to_dict(), sort_keys=True))
            f.write("\n")

    return output


def read_semantic_fingerprints(path: str | Path) -> list[SemanticDatasetFingerprint]:
    """Read semantic fingerprints from JSONL file."""
    fingerprints = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                fingerprints.append(SemanticDatasetFingerprint.from_dict(data))
    return fingerprints
