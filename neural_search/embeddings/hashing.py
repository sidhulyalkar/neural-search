"""Deterministic hashing embeddings for local tests and demos."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass(frozen=True)
class HashingEmbeddingProvider:
    """Lightweight token hashing provider with no model downloads."""

    dimensions: int = 64

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using a signed hashing trick and L2 normalization."""

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

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]
