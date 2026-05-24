"""Optional sentence-transformers embedding provider."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class SentenceTransformerEmbeddingProvider:
    """Provider backed by sentence-transformers when the extra is installed."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        model: Any | None = None,
        normalize: bool = True,
    ):
        self._model_name = model_name
        self._normalize = normalize
        if model is not None:
            self.model = model
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Install neural-search[embeddings] to use "
                "SentenceTransformerEmbeddingProvider."
            ) from exc
        self.model = SentenceTransformer(model_name)

    @property
    def provider_name(self) -> str:
        return "sentence-transformer"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        if hasattr(self.model, "get_sentence_embedding_dimension"):
            dimension = self.model.get_sentence_embedding_dimension()
            if dimension:
                return int(dimension)
        embedding = self.embed_text("")
        return len(embedding)

    @property
    def normalize(self) -> bool:
        return self._normalize

    def embed_text(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self.model.encode(
            list(texts),
            normalize_embeddings=self._normalize,
        )
        return [list(map(float, row)) for row in embeddings]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Backward-compatible alias for older retrieval code."""

        return self.embed_batch(texts)


SentenceTransformerProvider = SentenceTransformerEmbeddingProvider
