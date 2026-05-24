"""Common embedding provider protocol."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Minimal interface for text embedding providers.

    Providers expose metadata so generated embedding caches can be validated
    before retrieval mixes incompatible vectors.
    """

    @property
    def provider_name(self) -> str:
        """Stable provider identifier such as ``hashing``."""

    @property
    def model_name(self) -> str:
        """Model or implementation name used to generate embeddings."""

    @property
    def dimension(self) -> int:
        """Number of dimensions each embedding contains."""

    @property
    def normalize(self) -> bool:
        """Whether provider output is L2-normalized."""

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text into a fixed-length vector."""

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one or more texts into fixed-length vectors."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Backward-compatible alias for older retrieval code."""
