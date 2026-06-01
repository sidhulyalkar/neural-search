"""Query encoding with context enrichment for semantic search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neural_search.embeddings import EmbeddingProvider
    from neural_search.search.intent import QueryIntent


@dataclass
class EncodedQuery:
    """Query embedding with metadata about encoding process."""

    embedding: list[float]
    original_query: str
    enriched_query: str
    context_tokens: list[str]
    dimension: int


def enrich_query_with_context(
    query: str,
    parsed_query: dict[str, Any],
    max_context_tokens: int = 5,
) -> tuple[str, list[str]]:
    """Prepend detected concepts to query for better semantic matching.

    Args:
        query: Original query string
        parsed_query: Parsed query with detected tasks, modalities, etc.
        max_context_tokens: Maximum context tokens to prepend

    Returns:
        Tuple of (enriched_query, context_tokens)
    """
    context_tokens = []

    # Add detected tasks
    tasks = parsed_query.get("tasks", [])
    if tasks and len(context_tokens) < max_context_tokens:
        # Use normalized task names
        for task in tasks[:2]:
            context_tokens.append(f"task:{task}")

    # Add detected modalities
    modalities = parsed_query.get("modalities", [])
    if modalities and len(context_tokens) < max_context_tokens:
        for mod in modalities[:2]:
            context_tokens.append(f"modality:{mod}")

    # Add detected brain regions
    regions = parsed_query.get("brain_regions", [])
    if regions and len(context_tokens) < max_context_tokens:
        for region in regions[:1]:
            context_tokens.append(f"region:{region}")

    # Add detected affordances/analyses
    affordances = parsed_query.get("affordances", [])
    if affordances and len(context_tokens) < max_context_tokens:
        for aff in affordances[:1]:
            context_tokens.append(f"analysis:{aff}")

    # Build enriched query
    if context_tokens:
        enriched_query = " ".join(context_tokens) + " " + query
    else:
        enriched_query = query

    return enriched_query, context_tokens


def encode_query_with_context(
    query: str,
    parsed_query: dict[str, Any],
    provider: EmbeddingProvider,
    intent: QueryIntent | None = None,
) -> EncodedQuery:
    """Encode query with task/modality context for better matching.

    Args:
        query: Original query string
        parsed_query: Parsed query with detected concepts
        provider: Embedding provider to use
        intent: Optional query intent for adaptive encoding

    Returns:
        EncodedQuery with embedding and metadata
    """
    enriched_query, context_tokens = enrich_query_with_context(query, parsed_query)

    # Encode the enriched query
    embedding = provider.embed_text(enriched_query)

    return EncodedQuery(
        embedding=embedding,
        original_query=query,
        enriched_query=enriched_query,
        context_tokens=context_tokens,
        dimension=len(embedding),
    )


def compute_query_dataset_similarity(
    query_embedding: list[float],
    dataset_embeddings: dict[str, list[float]],
    field_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute similarity between query and dataset field embeddings.

    Args:
        query_embedding: Query embedding vector
        dataset_embeddings: Dict mapping field names to embeddings
        field_weights: Optional weights per field

    Returns:
        Dict with similarity scores per field and combined score
    """
    from neural_search.embeddings import cosine_similarity

    default_weights = {
        "title": 0.20,
        "description": 0.18,
        "combined_scientific_summary": 0.22,
        "tasks": 0.15,
        "modalities": 0.10,
        "brain_regions": 0.08,
        "behavioral_events": 0.12,
        "analysis_goals": 0.10,
    }

    weights = {**default_weights, **(field_weights or {})}

    field_scores = {}
    weighted_sum = 0.0
    weight_total = 0.0

    for field_name, field_embedding in dataset_embeddings.items():
        similarity = max(0.0, cosine_similarity(query_embedding, field_embedding))
        field_scores[field_name] = round(similarity, 4)

        weight = weights.get(field_name, 0.05)
        weighted_sum += similarity * weight
        weight_total += weight

    # Compute combined score
    combined = weighted_sum / weight_total if weight_total > 0 else 0.0
    field_scores["combined"] = round(combined, 4)

    return field_scores


@dataclass
class QueryExpansion:
    """Expanded query terms for broader matching."""

    original_terms: list[str]
    expanded_terms: list[str]
    synonym_expansions: dict[str, list[str]]
    task_related_terms: list[str]
    modality_related_terms: list[str]


def expand_query_terms(
    query: str,
    parsed_query: dict[str, Any],
    max_expansions: int = 10,
) -> QueryExpansion:
    """Expand query with related terms from ontology matching.

    Args:
        query: Original query
        parsed_query: Parsed query with matched concepts
        max_expansions: Maximum expansion terms to add

    Returns:
        QueryExpansion with original and expanded terms
    """
    original_terms = query.lower().split()
    expanded_terms = list(original_terms)
    synonym_expansions: dict[str, list[str]] = {}
    task_related = []
    modality_related = []

    # Add task synonyms
    for task in parsed_query.get("tasks", []):
        task_lower = task.lower().replace("_", " ")
        if task_lower not in expanded_terms:
            expanded_terms.append(task_lower)
            task_related.append(task_lower)

    # Add modality synonyms
    for modality in parsed_query.get("modalities", []):
        mod_lower = modality.lower().replace("_", " ")
        if mod_lower not in expanded_terms:
            expanded_terms.append(mod_lower)
            modality_related.append(mod_lower)

    # Add behavioral events from transitive matches
    transitive = parsed_query.get("transitive_matches", [])
    for match in transitive[:max_expansions]:
        if isinstance(match, dict):
            target = match.get("target", "").lower()
            if target and target not in expanded_terms:
                expanded_terms.append(target)

    return QueryExpansion(
        original_terms=original_terms,
        expanded_terms=expanded_terms[:max_expansions + len(original_terms)],
        synonym_expansions=synonym_expansions,
        task_related_terms=task_related,
        modality_related_terms=modality_related,
    )
