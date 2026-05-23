"""Embedding providers for offline and optional semantic retrieval."""

from neural_search.embeddings.base import EmbeddingProvider
from neural_search.embeddings.hashing import HashingEmbeddingProvider
from neural_search.embeddings.index import cosine_similarity

__all__ = [
    "EmbeddingProvider",
    "HashingEmbeddingProvider",
    "cosine_similarity",
]
