"""Common embedding provider protocol."""

from __future__ import annotations

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Minimal interface for text embedding providers."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed one or more texts into fixed-length vectors."""
