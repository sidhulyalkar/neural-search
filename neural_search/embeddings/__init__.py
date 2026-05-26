"""Embedding providers for offline and optional semantic retrieval."""

from neural_search.embeddings.base import EmbeddingProvider
from neural_search.embeddings.concept_builder import (
    ConceptEmbeddingBuilder,
    build_concept_embeddings_from_ontology,
)
from neural_search.embeddings.concept_embeddings import (
    ConceptEmbedding,
    ConceptEmbeddingIndex,
    ConceptSimilarity,
    load_concept_index,
    read_concept_embeddings,
    write_concept_embeddings,
)
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
from neural_search.embeddings.semantic_fingerprint import (
    SemanticDatasetFingerprint,
    SemanticFingerprintBuilder,
    SemanticSimilarity,
    compute_semantic_similarity,
    read_semantic_fingerprints,
    write_semantic_fingerprints,
)
from neural_search.embeddings.semantic_similarity import (
    QuerySemanticExpansion,
    SemanticMatch,
    compute_query_concept_similarity,
    concept_similarity,
    expand_query_semantically,
    find_semantically_similar,
    merge_query_with_expansion,
)
from neural_search.embeddings.sentence_transformers import (
    SentenceTransformerEmbeddingProvider,
    SentenceTransformerProvider,
)

__all__ = [
    # Base
    "EmbeddingProvider",
    # Concept embeddings
    "ConceptEmbedding",
    "ConceptEmbeddingBuilder",
    "ConceptEmbeddingIndex",
    "ConceptSimilarity",
    "build_concept_embeddings_from_ontology",
    "load_concept_index",
    "read_concept_embeddings",
    "write_concept_embeddings",
    # Semantic similarity
    "QuerySemanticExpansion",
    "SemanticMatch",
    "compute_query_concept_similarity",
    "concept_similarity",
    "expand_query_semantically",
    "find_semantically_similar",
    "merge_query_with_expansion",
    # Semantic fingerprints
    "SemanticDatasetFingerprint",
    "SemanticFingerprintBuilder",
    "SemanticSimilarity",
    "compute_semantic_similarity",
    "read_semantic_fingerprints",
    "write_semantic_fingerprints",
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
