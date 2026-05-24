"""Embedding providers for offline and optional semantic retrieval."""

from neural_search.embeddings.base import EmbeddingProvider
from neural_search.embeddings.field_index import (
    DATASET_EMBEDDING_FIELDS,
    PAPER_EMBEDDING_FIELDS,
    FieldEmbeddingRecord,
    build_field_embedding_records,
    field_texts_for_record,
    read_field_embedding_cache,
    validate_field_embedding_cache,
    write_field_embedding_cache,
)
from neural_search.embeddings.hashing import HashingEmbeddingProvider
from neural_search.embeddings.index import cosine_similarity
from neural_search.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddingProvider,
    SentenceTransformerProvider,
)

__all__ = [
    "EmbeddingProvider",
    "DATASET_EMBEDDING_FIELDS",
    "PAPER_EMBEDDING_FIELDS",
    "FieldEmbeddingRecord",
    "HashingEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "SentenceTransformerProvider",
    "build_field_embedding_records",
    "cosine_similarity",
    "field_texts_for_record",
    "read_field_embedding_cache",
    "validate_field_embedding_cache",
    "write_field_embedding_cache",
]
