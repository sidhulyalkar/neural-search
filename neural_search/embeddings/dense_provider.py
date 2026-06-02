"""Dense embedding provider for BGE-large-en-v1.5.

BGE-large-en-v1.5 (BAAI/bge-large-en-v1.5):
  - 1024 dimensions
  - MTEB top-5 for scientific retrieval
  - Same model for corpus + query (cosine similarity requires identical spaces)
  - ~4GB VRAM on 3070 Ti; batch-embeds ~738 datasets in ~45s on GPU
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from neural_search.embeddings.providers import EmbeddingProviderBase

logger = logging.getLogger(__name__)


class DenseEmbeddingProvider(EmbeddingProviderBase):
    """BGE-large-en-v1.5 provider using sentence-transformers."""

    MODEL_NAME = "BAAI/bge-large-en-v1.5"
    DIMENSION = 1024

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 64,
        normalize: bool = True,
        model: Any | None = None,
    ) -> None:
        self._batch_size = batch_size
        self._normalize = normalize
        self._model_version = "unknown"

        if model is not None:
            self._model = model
            self._device = device or "cpu"
            return

        self._device = device or self._auto_device()
        self._load_model()

    def _auto_device(self) -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading %s on %s", self.MODEL_NAME, self._device)
            self._model = SentenceTransformer(self.MODEL_NAME, device=self._device)
            try:
                import sentence_transformers
                self._model_version = getattr(sentence_transformers, "__version__", "unknown")
            except Exception:
                pass
        except ImportError as exc:
            raise RuntimeError(
                "Install sentence-transformers to use DenseEmbeddingProvider: "
                "pip install sentence-transformers"
            ) from exc

    @property
    def provider_name(self) -> str:
        return "bge-large"

    @property
    def model_name(self) -> str:
        return self.MODEL_NAME

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def normalize(self) -> bool:
        return self._normalize

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches. Returns list of 1024-dim float vectors."""
        all_vecs: list[list[float]] = []
        texts_list = list(texts)
        for i in range(0, len(texts_list), self._batch_size):
            batch = texts_list[i : i + self._batch_size]
            vecs = self._model.encode(
                batch,
                normalize_embeddings=self._normalize,
                show_progress_bar=False,
            )
            all_vecs.extend(list(map(float, v)) for v in vecs)
        return all_vecs

    def embed_corpus_batch(self, texts: list[str]) -> np.ndarray:
        """Embed corpus texts. Returns (N, 1024) float32 array."""
        if not texts:
            return np.empty((0, self.DIMENSION), dtype=np.float32)
        return np.array(self.embed_batch(texts), dtype=np.float32)
