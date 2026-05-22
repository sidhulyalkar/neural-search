"""Embedding service for vector search."""

from typing import Optional
import hashlib


class EmbeddingService:
    """
    Generate and manage embeddings for semantic search.

    Currently a stub - will integrate with embedding models.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._cache: dict[str, list[float]] = {}

    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                # Fallback: no embeddings available
                self._model = None

    def embed_text(self, text: str) -> Optional[list[float]]:
        """
        Generate embedding for text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector or None if unavailable.
        """
        # Check cache
        cache_key = self._cache_key(text)
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._load_model()

        if self._model is None:
            return None

        embedding = self._model.encode(text).tolist()
        self._cache[cache_key] = embedding
        return embedding

    def embed_batch(self, texts: list[str]) -> list[Optional[list[float]]]:
        """Generate embeddings for multiple texts."""
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._cache_key(text)
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)

        # Embed uncached texts
        if uncached_texts:
            self._load_model()
            if self._model is not None:
                embeddings = self._model.encode(uncached_texts)
                for i, (text, embedding) in enumerate(
                    zip(uncached_texts, embeddings)
                ):
                    cache_key = self._cache_key(text)
                    self._cache[cache_key] = embedding.tolist()
                    results[uncached_indices[i]] = embedding.tolist()

        return results

    def similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """Compute cosine similarity between embeddings."""
        import math

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = math.sqrt(sum(a * a for a in embedding1))
        norm2 = math.sqrt(sum(b * b for b in embedding2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()
