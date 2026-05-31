"""Scientific embedding providers with model comparison support.

This module implements embedding providers for scientific dataset retrieval:

1. HashingEmbeddingProvider - Deterministic hashing (no dependencies, for CI)
2. SentenceTransformerProvider - General sentence embeddings
3. SPECTER2Provider - Scientific document embeddings (optional)
4. SciBERTProvider - Biomedical/scientific embeddings (optional)

Each provider implements the EmbeddingProvider protocol with:
- Consistent interface for embedding generation
- Version tracking for reproducibility
- Batch processing support
- Metadata for model comparison

Usage:
    from neural_search.embeddings.providers import get_provider

    # Get default provider (hashing for CI, sentence-transformer for production)
    provider = get_provider()

    # Get specific provider
    specter = get_provider("specter2")
    scibert = get_provider("scibert")

    # Embed text
    embedding = provider.embed_text("neural activity in visual cortex")
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


class EmbeddingModelType(StrEnum):
    """Types of embedding models."""

    HASHING = "hashing"
    SENTENCE_TRANSFORMER = "sentence-transformer"
    SPECTER2 = "specter2"
    SCIBERT = "scibert"
    PUBMEDBERT = "pubmedbert"
    COLBERT = "colbert"


class EmbeddingRecord(BaseModel):
    """Record of an embedding with full provenance.

    Tracks what was embedded, by which model, and when.
    Essential for cache validation and reproducibility.
    """

    entity_id: str                       # e.g., "dataset:dandi:000026"
    field: str                           # e.g., "text_card", "title", "description"
    vector: list[float]
    vector_hash: str                     # Hash of the vector for validation

    # Provider metadata
    provider: str                        # e.g., "sentence-transformer"
    model_name: str                      # e.g., "all-MiniLM-L6-v2"
    model_version: str                   # e.g., "1.0.1"
    dimension: int

    # Input metadata
    text_hash: str                       # Hash of input text
    text_length: int                     # Character count of input

    # Timestamps
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    corpus_snapshot_id: str | None = None

    @staticmethod
    def compute_vector_hash(vector: list[float]) -> str:
        """Compute a hash of the vector for validation."""
        # Use first/last values and length for quick hash
        key = f"{len(vector)}:{vector[0]:.6f}:{vector[-1]:.6f}:{sum(vector):.6f}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]

    @staticmethod
    def compute_text_hash(text: str) -> str:
        """Compute a hash of the input text."""
        return hashlib.sha256(text.encode()).hexdigest()[:12]


class EmbeddingProviderBase(ABC):
    """Base class for embedding providers with common functionality."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Stable provider identifier."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model or implementation name."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Model version for reproducibility."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Number of dimensions in embeddings."""

    @property
    @abstractmethod
    def normalize(self) -> bool:
        """Whether embeddings are L2-normalized."""

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple texts."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Backward-compatible alias."""
        return self.embed_batch(texts)

    def create_embedding_record(
        self,
        entity_id: str,
        field: str,
        text: str,
        corpus_snapshot_id: str | None = None,
    ) -> EmbeddingRecord:
        """Create an embedding record with full provenance."""
        vector = self.embed_text(text)
        return EmbeddingRecord(
            entity_id=entity_id,
            field=field,
            vector=vector,
            vector_hash=EmbeddingRecord.compute_vector_hash(vector),
            provider=self.provider_name,
            model_name=self.model_name,
            model_version=self.model_version,
            dimension=self.dimension,
            text_hash=EmbeddingRecord.compute_text_hash(text),
            text_length=len(text),
            corpus_snapshot_id=corpus_snapshot_id,
        )

    def get_metadata(self) -> dict[str, Any]:
        """Get provider metadata for logging and comparison."""
        return {
            "provider": self.provider_name,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "dimension": self.dimension,
            "normalize": self.normalize,
        }


@dataclass(frozen=True)
class HashingEmbeddingProvider(EmbeddingProviderBase):
    """Deterministic token hashing provider.

    No model downloads required. Suitable for CI and testing.
    Provides consistent embeddings based on token hashing.
    """

    dimensions: int = 64
    normalize_embeddings: bool = True

    @property
    def provider_name(self) -> str:
        return "hashing"

    @property
    def model_name(self) -> str:
        return f"signed-token-hashing-{self.dimensions}"

    @property
    def model_version(self) -> str:
        return "1.0.0"

    @property
    def dimension(self) -> int:
        return self.dimensions

    @property
    def normalize(self) -> bool:
        return self.normalize_embeddings

    def embed_text(self, text: str) -> list[float]:
        return self._embed_one(text)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        if self.dimensions < 1:
            raise ValueError("dimensions must be at least 1")

        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.casefold()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        if not self.normalize_embeddings:
            return vector
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformerProvider(EmbeddingProviderBase):
    """Provider using sentence-transformers library.

    Default model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
    Alternative: all-mpnet-base-v2 (768 dimensions, slower, better quality)
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        model: Any | None = None,
        normalize: bool = True,
    ):
        self._model_name = model_name
        self._normalize = normalize
        self._model_version = "unknown"

        if model is not None:
            self._model = model
            return

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)

            # Try to get version
            try:
                import sentence_transformers

                self._model_version = getattr(
                    sentence_transformers, "__version__", "unknown"
                )
            except Exception:
                pass

        except ImportError as exc:
            raise RuntimeError(
                "Install neural-search[embeddings] to use SentenceTransformerProvider. "
                "Run: pip install sentence-transformers"
            ) from exc

    @property
    def provider_name(self) -> str:
        return "sentence-transformer"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_sentence_embedding_dimension"):
            dim = self._model.get_sentence_embedding_dimension()
            if dim:
                return int(dim)
        # Fallback: compute from sample
        sample = self.embed_text("test")
        return len(sample)

    @property
    def normalize(self) -> bool:
        return self._normalize

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            list(texts),
            normalize_embeddings=self._normalize,
        )
        return [list(map(float, row)) for row in embeddings]


class SPECTER2Provider(EmbeddingProviderBase):
    """SPECTER2 provider for scientific document embeddings.

    SPECTER2 is trained on scientific papers and provides
    better representations for scientific text than general models.

    Models:
    - allenai/specter2_base: Base model (768d)
    - allenai/specter2: With proximity adapter
    - allenai/specter2_aug2023refresh: Latest refresh

    Requires: pip install transformers torch
    """

    def __init__(
        self,
        model_name: str = "allenai/specter2_base",
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        max_length: int = 512,
        normalize: bool = True,
    ):
        self._model_name = model_name
        self._max_length = max_length
        self._normalize = normalize
        self._model_version = "unknown"
        self._dimension = 768  # SPECTER2 dimension

        if model is not None and tokenizer is not None:
            self._model = model
            self._tokenizer = tokenizer
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading SPECTER2 model: {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.eval()

            # Get version
            try:
                import transformers

                self._model_version = getattr(transformers, "__version__", "unknown")
            except Exception:
                pass

            logger.info(f"SPECTER2 loaded: {model_name}")

        except ImportError as exc:
            raise RuntimeError(
                "Install transformers and torch to use SPECTER2Provider. "
                "Run: pip install transformers torch"
            ) from exc

    @property
    def provider_name(self) -> str:
        return "specter2"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalize(self) -> bool:
        return self._normalize

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        import torch

        # Tokenize
        inputs = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        # Get embeddings
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Use CLS token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]

        # Normalize if requested
        if self._normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()


class SciBERTProvider(EmbeddingProviderBase):
    """SciBERT provider for scientific text embeddings.

    SciBERT is trained on scientific papers from Semantic Scholar
    and provides better representations for scientific domains.

    Models:
    - allenai/scibert_scivocab_uncased: Base SciBERT (768d)

    Requires: pip install transformers torch
    """

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        max_length: int = 512,
        normalize: bool = True,
        pooling: str = "mean",  # "cls", "mean", "max"
    ):
        self._model_name = model_name
        self._max_length = max_length
        self._normalize = normalize
        self._pooling = pooling
        self._model_version = "unknown"
        self._dimension = 768

        if model is not None and tokenizer is not None:
            self._model = model
            self._tokenizer = tokenizer
            return

        try:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading SciBERT model: {model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._model.eval()

            try:
                import transformers

                self._model_version = getattr(transformers, "__version__", "unknown")
            except Exception:
                pass

            logger.info(f"SciBERT loaded: {model_name}")

        except ImportError as exc:
            raise RuntimeError(
                "Install transformers and torch to use SciBERTProvider. "
                "Run: pip install transformers torch"
            ) from exc

    @property
    def provider_name(self) -> str:
        return "scibert"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalize(self) -> bool:
        return self._normalize

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        import torch

        inputs = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Pooling
        if self._pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif self._pooling == "mean":
            # Mean pooling with attention mask
            attention_mask = inputs["attention_mask"]
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embeddings = sum_embeddings / sum_mask
        elif self._pooling == "max":
            embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling: {self._pooling}")

        if self._normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.tolist()


@dataclass
class EmbeddingModelComparison:
    """Results of comparing embedding models on a benchmark."""

    models: list[str]
    corpus_size: int
    query_count: int

    # Per-model metrics
    model_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    # model -> {precision_at_5, mrr, recall_at_10, latency_ms}

    # Head-to-head comparison
    wins: dict[str, dict[str, int]] = field(default_factory=dict)
    # model -> {other_model -> win_count}

    # Dimension and speed info
    model_dimensions: dict[str, int] = field(default_factory=dict)
    model_latencies: dict[str, float] = field(default_factory=dict)

    generated_at: str = ""


# Provider registry and factory


_PROVIDER_REGISTRY: dict[str, type[EmbeddingProviderBase]] = {
    "hashing": HashingEmbeddingProvider,
    "sentence-transformer": SentenceTransformerProvider,
    "specter2": SPECTER2Provider,
    "scibert": SciBERTProvider,
}


def get_provider(
    name: str = "auto",
    **kwargs: Any,
) -> EmbeddingProviderBase:
    """Get an embedding provider by name.

    Args:
        name: Provider name. Options:
            - "auto": Use sentence-transformer if available, else hashing
            - "hashing": Deterministic hashing (no dependencies)
            - "sentence-transformer": General sentence embeddings
            - "specter2": Scientific document embeddings
            - "scibert": Scientific text embeddings
        **kwargs: Provider-specific arguments

    Returns:
        Configured embedding provider
    """
    if name == "auto":
        # Try sentence-transformer first, fall back to hashing
        try:
            return SentenceTransformerProvider(**kwargs)
        except RuntimeError:
            logger.info("sentence-transformers not available, using hashing provider")
            return HashingEmbeddingProvider(**kwargs)

    if name not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider: {name}. "
            f"Available: {list(_PROVIDER_REGISTRY.keys())}"
        )

    return _PROVIDER_REGISTRY[name](**kwargs)


def list_available_providers() -> list[str]:
    """List all registered provider names."""
    return list(_PROVIDER_REGISTRY.keys())


def check_provider_availability() -> dict[str, bool]:
    """Check which providers are available (dependencies installed)."""
    availability = {}

    # Hashing is always available
    availability["hashing"] = True

    # Check sentence-transformers
    try:
        import sentence_transformers  # noqa: F401

        availability["sentence-transformer"] = True
    except ImportError:
        availability["sentence-transformer"] = False

    # Check transformers (for SPECTER2, SciBERT)
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        availability["specter2"] = True
        availability["scibert"] = True
    except ImportError:
        availability["specter2"] = False
        availability["scibert"] = False

    return availability
