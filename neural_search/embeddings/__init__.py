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
from neural_search.embeddings.fingerprint import (
    DatasetFingerprint,
    FingerprintSimilarity,
    compute_fingerprint_similarity,
    read_fingerprints,
    write_fingerprints,
)
from neural_search.embeddings.fingerprint_builder import (
    DatasetFingerprintBuilder,
    build_fingerprints_from_corpus,
)
from neural_search.embeddings.fingerprint_index import (
    FingerprintIndex,
    SimilarDataset,
    load_fingerprint_index,
)
from neural_search.embeddings.hashing import HashingEmbeddingProvider
from neural_search.embeddings.index import cosine_similarity
from neural_search.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddingProvider,
    SentenceTransformerProvider,
)

__all__ = [
    # Base
    "EmbeddingProvider",
    # Field embeddings
    "DATASET_EMBEDDING_FIELDS",
    "PAPER_EMBEDDING_FIELDS",
    "FieldEmbeddingRecord",
    "build_field_embedding_records",
    "field_texts_for_record",
    "read_field_embedding_cache",
    "validate_field_embedding_cache",
    "write_field_embedding_cache",
    # Fingerprints
    "DatasetFingerprint",
    "DatasetFingerprintBuilder",
    "FingerprintIndex",
    "FingerprintSimilarity",
    "SimilarDataset",
    "build_fingerprints_from_corpus",
    "compute_fingerprint_similarity",
    "load_fingerprint_index",
    "read_fingerprints",
    "write_fingerprints",
    # Providers
    "HashingEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
    "SentenceTransformerProvider",
    "cosine_similarity",
]
