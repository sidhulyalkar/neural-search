"""Deterministic hashing embeddings for local tests and demos."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class HashingEmbeddingProvider:
    """Lightweight token hashing provider with no model downloads."""

    dimensions: int = 64
    normalize_embeddings: bool = True

    @property
    def provider_name(self) -> str:
        return "hashing"

    @property
    def model_name(self) -> str:
        return f"signed-token-hashing-{self.dimensions}"

    @property
    def dimension(self) -> int:
        return self.dimensions

    @property
    def normalize(self) -> bool:
        return self.normalize_embeddings

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text using a signed hashing trick."""

        return self._embed_one(text)

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts using a signed hashing trick."""

        return [self._embed_one(text) for text in texts]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Backward-compatible alias for older retrieval code."""

        return self.embed_batch(texts)

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
