"""Explainable in-memory search skeleton."""

from neural_search.search.core import (
    hybrid_search_with_latent,
    load_retrieval_config,
    parse_query,
    score_dataset_against_query,
    search_datasets,
)
from neural_search.search.hybrid import (
    FusionMethod,
    HybridRetrievalConfig,
    HybridScore,
    compute_hybrid_score,
    get_hybrid_config_for_intent,
    rerank_with_hybrid_scores,
)
from neural_search.search.intent import (
    IntentClassification,
    IntentProfile,
    QueryIntent,
    blend_weights,
    classify_query_intent,
    get_intent_profile,
    get_weights_for_intent,
    load_intent_profiles,
)
from neural_search.search.query_encoder import (
    EncodedQuery,
    QueryExpansion,
    encode_query_with_context,
    enrich_query_with_context,
    expand_query_terms,
)
from neural_search.search.semantic_expansion import (
    SemanticExpansion,
    compute_expansion_boost,
    enrich_query_with_semantic_context,
    expand_query_with_concepts,
    merge_expansion_into_query,
)
from neural_search.search.semantic_scoring import (
    SemanticScoreResult,
    SemanticSearchIndex,
    augment_result_with_semantic_score,
    compute_semantic_score_for_result,
    load_semantic_index,
)
from neural_search.search.weight_optimizer import (
    WEIGHT_PROFILES,
    QueryAnalysis,
    QueryComplexity,
    WeightProfile,
    WeightSensitivity,
    analyze_query_for_weights,
    analyze_weight_sensitivity,
    boost_weights_for_constraints,
    compute_weight_sensitivity,
    get_adaptive_weights,
    interpolate_weights,
    normalize_weights,
)

__all__ = [
    # Core search
    "hybrid_search_with_latent",
    "load_retrieval_config",
    "parse_query",
    "score_dataset_against_query",
    "search_datasets",
    # Hybrid retrieval
    "FusionMethod",
    "HybridRetrievalConfig",
    "HybridScore",
    "compute_hybrid_score",
    "get_hybrid_config_for_intent",
    "rerank_with_hybrid_scores",
    # Intent classification
    "IntentClassification",
    "IntentProfile",
    "QueryIntent",
    "blend_weights",
    "classify_query_intent",
    "get_intent_profile",
    "get_weights_for_intent",
    "load_intent_profiles",
    # Query encoding
    "EncodedQuery",
    "QueryExpansion",
    "encode_query_with_context",
    "enrich_query_with_context",
    "expand_query_terms",
    # Weight optimization
    "QueryAnalysis",
    "QueryComplexity",
    "WeightProfile",
    "WeightSensitivity",
    "WEIGHT_PROFILES",
    "analyze_query_for_weights",
    "analyze_weight_sensitivity",
    "boost_weights_for_constraints",
    "compute_weight_sensitivity",
    "get_adaptive_weights",
    "interpolate_weights",
    "normalize_weights",
    # Semantic scoring
    "SemanticScoreResult",
    "SemanticSearchIndex",
    "augment_result_with_semantic_score",
    "compute_semantic_score_for_result",
    "load_semantic_index",
    # Semantic expansion
    "SemanticExpansion",
    "compute_expansion_boost",
    "enrich_query_with_semantic_context",
    "expand_query_with_concepts",
    "merge_expansion_into_query",
]
