"""Semantic fingerprint scoring for search results."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from neural_search.embeddings.concept_embeddings import (
        ConceptEmbeddingIndex,
    )
    from neural_search.embeddings.semantic_fingerprint import (
        SemanticDatasetFingerprint,
    )


@dataclass
class SemanticScoreResult:
    """Result of semantic fingerprint scoring."""

    score: float  # Overall semantic score 0-1
    task_relevance: float  # Task dimension relevance
    modality_relevance: float  # Modality dimension relevance
    behavior_relevance: float  # Behavior dimension relevance
    analysis_relevance: float  # Analysis/affordance relevance
    design_relevance: float  # Design type relevance
    explanation: str
    similar_datasets: list[str] = field(default_factory=list)


@dataclass
class SemanticSearchIndex:
    """Index for semantic search scoring."""

    fingerprints: dict[str, SemanticDatasetFingerprint]
    concept_index: ConceptEmbeddingIndex | None

    def get_fingerprint(self, dataset_id: str) -> SemanticDatasetFingerprint | None:
        """Get fingerprint for a dataset."""
        return self.fingerprints.get(dataset_id)

    def find_similar(
        self,
        query_embedding: list[float],
        field_name: str,
        top_k: int = 5,
        min_similarity: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Find datasets similar to a query embedding in a specific dimension."""
        results: list[tuple[str, float]] = []

        query_vec = np.array(query_embedding)
        query_norm = np.linalg.norm(query_vec)
        if query_norm < 1e-9:
            return results

        for dataset_id, fp in self.fingerprints.items():
            fp_vec = np.array(getattr(fp, f"{field_name}_embedding", []))
            if len(fp_vec) != len(query_vec):
                continue

            fp_norm = np.linalg.norm(fp_vec)
            if fp_norm < 1e-9:
                continue

            similarity = float(np.dot(query_vec, fp_vec) / (query_norm * fp_norm))
            if similarity >= min_similarity:
                results.append((dataset_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]


def load_semantic_index(
    fingerprints_path: str | Path | None = None,
    concept_embeddings_path: str | Path | None = None,
) -> SemanticSearchIndex | None:
    """Load semantic search index from files.

    Args:
        fingerprints_path: Path to semantic fingerprints JSONL
        concept_embeddings_path: Path to concept embeddings JSONL

    Returns:
        SemanticSearchIndex or None if files don't exist
    """
    from neural_search.embeddings.concept_embeddings import load_concept_index
    from neural_search.embeddings.semantic_fingerprint import (
        read_semantic_fingerprints,
    )

    fingerprints: dict[str, SemanticDatasetFingerprint] = {}
    concept_index = None

    if fingerprints_path:
        fp_path = Path(fingerprints_path)
        if fp_path.exists():
            fps = read_semantic_fingerprints(fp_path)
            fingerprints = {fp.dataset_id: fp for fp in fps}

    if concept_embeddings_path:
        ce_path = Path(concept_embeddings_path)
        if ce_path.exists():
            concept_index = load_concept_index(ce_path)

    if not fingerprints:
        return None

    return SemanticSearchIndex(
        fingerprints=fingerprints,
        concept_index=concept_index,
    )


@lru_cache(maxsize=2)
def _cached_semantic_index(fingerprints_path: str, concept_path: str | None) -> SemanticSearchIndex | None:
    """Cached version of semantic index loading."""
    return load_semantic_index(fingerprints_path, concept_path)


def compute_semantic_score_for_result(
    index: SemanticSearchIndex,
    dataset_id: str,
    parsed_query: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> SemanticScoreResult:
    """Compute semantic score for a search result.

    Args:
        index: Semantic search index
        dataset_id: Dataset ID to score
        parsed_query: Parsed query with extracted intents
        weights: Optional dimension weights

    Returns:
        SemanticScoreResult
    """
    default_weights = {
        "task": 0.30,
        "modality": 0.20,
        "behavior": 0.15,
        "analysis": 0.15,
        "design": 0.10,
        "text": 0.10,
    }
    weights = weights or default_weights

    fingerprint = index.get_fingerprint(dataset_id)
    if fingerprint is None:
        return SemanticScoreResult(
            score=0.0,
            task_relevance=0.0,
            modality_relevance=0.0,
            behavior_relevance=0.0,
            analysis_relevance=0.0,
            design_relevance=0.0,
            explanation="No semantic fingerprint available for this dataset",
        )

    # Compute relevance per dimension based on query concepts
    task_relevance = _compute_task_relevance(fingerprint, parsed_query, index)
    modality_relevance = _compute_modality_relevance(fingerprint, parsed_query, index)
    behavior_relevance = _compute_behavior_relevance(fingerprint, parsed_query)
    analysis_relevance = _compute_analysis_relevance(fingerprint, parsed_query)
    design_relevance = _compute_design_relevance(fingerprint, parsed_query)

    # Weighted combination
    total_score = (
        weights.get("task", 0.3) * task_relevance
        + weights.get("modality", 0.2) * modality_relevance
        + weights.get("behavior", 0.15) * behavior_relevance
        + weights.get("analysis", 0.15) * analysis_relevance
        + weights.get("design", 0.1) * design_relevance
    )

    # Find similar datasets
    similar = _find_semantically_similar_datasets(fingerprint, index, top_k=3)

    # Generate explanation
    explanation_parts = []
    if task_relevance > 0.5:
        explanation_parts.append(f"task alignment ({task_relevance:.0%})")
    if modality_relevance > 0.5:
        explanation_parts.append(f"modality match ({modality_relevance:.0%})")
    if behavior_relevance > 0.5:
        explanation_parts.append(f"behavioral similarity ({behavior_relevance:.0%})")
    if analysis_relevance > 0.5:
        explanation_parts.append(f"analysis fit ({analysis_relevance:.0%})")

    explanation = (
        f"Semantic match: {', '.join(explanation_parts)}"
        if explanation_parts
        else "Limited semantic relevance to query"
    )

    return SemanticScoreResult(
        score=round(min(total_score, 1.0), 3),
        task_relevance=round(task_relevance, 3),
        modality_relevance=round(modality_relevance, 3),
        behavior_relevance=round(behavior_relevance, 3),
        analysis_relevance=round(analysis_relevance, 3),
        design_relevance=round(design_relevance, 3),
        explanation=explanation,
        similar_datasets=similar,
    )


def _extract_label_core(label: str) -> str:
    """Extract core label from prefixed format like 'label:task:reversal_learning:source'."""
    parts = label.split(":")
    # Look for the main identifier after 'label:type:'
    if len(parts) >= 3 and parts[0] == "label":
        return parts[2].lower()
    return label.lower().replace("_", " ")


def _compute_task_relevance(
    fingerprint: SemanticDatasetFingerprint,
    parsed_query: dict[str, Any],
    index: SemanticSearchIndex,
) -> float:
    """Compute task dimension relevance."""
    query_tasks = parsed_query.get("tasks", [])
    if not query_tasks:
        return 0.5  # Neutral if no task specified

    # Extract core labels from fingerprint task labels
    dataset_tasks = {_extract_label_core(t) for t in (fingerprint.task_labels or [])}
    query_tasks_lower = {t.lower().replace("_", " ") for t in query_tasks}

    # Direct match
    direct_match = len(query_tasks_lower & dataset_tasks) / len(query_tasks_lower)
    if direct_match > 0:
        return min(direct_match + 0.2, 1.0)  # Boost for direct match

    # Semantic similarity using concept index
    if index.concept_index is not None:
        max_similarity = 0.0
        for query_task in query_tasks:
            query_concept = index.concept_index.get_by_label(query_task)
            if query_concept is None:
                continue

            for dataset_task in dataset_tasks:
                dataset_concept = index.concept_index.get_by_label(dataset_task)
                if dataset_concept is None:
                    continue

                sim = index.concept_index.compute_similarity(
                    query_concept.concept_id,
                    dataset_concept.concept_id,
                )
                max_similarity = max(max_similarity, sim)

        return max_similarity

    return 0.0


def _compute_modality_relevance(
    fingerprint: SemanticDatasetFingerprint,
    parsed_query: dict[str, Any],
    index: SemanticSearchIndex,
) -> float:
    """Compute modality dimension relevance."""
    query_modalities = parsed_query.get("modalities", [])
    if not query_modalities:
        return 0.5  # Neutral if no modality specified

    # Extract core labels from fingerprint modality labels
    dataset_modalities = {_extract_label_core(m) for m in (fingerprint.modality_labels or [])}
    query_modalities_lower = {m.lower().replace("_", " ") for m in query_modalities}

    # Direct match
    direct_match = len(query_modalities_lower & dataset_modalities) / len(
        query_modalities_lower
    )
    if direct_match > 0:
        return min(direct_match + 0.2, 1.0)

    # Semantic similarity using concept index
    if index.concept_index is not None:
        max_similarity = 0.0
        for query_mod in query_modalities:
            query_concept = index.concept_index.get_by_label(query_mod)
            if query_concept is None:
                continue

            for dataset_mod in dataset_modalities:
                dataset_concept = index.concept_index.get_by_label(dataset_mod)
                if dataset_concept is None:
                    continue

                sim = index.concept_index.compute_similarity(
                    query_concept.concept_id,
                    dataset_concept.concept_id,
                )
                max_similarity = max(max_similarity, sim)

        return max_similarity

    return 0.0


def _compute_behavior_relevance(
    fingerprint: SemanticDatasetFingerprint,
    parsed_query: dict[str, Any],
) -> float:
    """Compute behavior dimension relevance."""
    query_behaviors = parsed_query.get("behaviors", [])
    if not query_behaviors:
        return 0.5  # Neutral if no behavior specified

    # Extract core labels
    dataset_behaviors = {_extract_label_core(b) for b in (fingerprint.behavior_labels or [])}
    query_behaviors_lower = {b.lower().replace("_", " ") for b in query_behaviors}

    # Direct match
    if dataset_behaviors and query_behaviors_lower:
        direct_match = len(query_behaviors_lower & dataset_behaviors) / len(
            query_behaviors_lower
        )
        if direct_match > 0:
            return min(direct_match + 0.2, 1.0)

    # Use behavior complexity as proxy
    complexity = fingerprint.behavior_complexity
    # High complexity behaviors are often more relevant to specific queries
    return complexity * 0.5


def _compute_analysis_relevance(
    fingerprint: SemanticDatasetFingerprint,
    parsed_query: dict[str, Any],
) -> float:
    """Compute analysis/affordance dimension relevance."""
    query_analyses = set(parsed_query.get("analysis", []))
    query_affordances = set(parsed_query.get("affordances", []))
    query_terms = query_analyses | query_affordances

    if not query_terms:
        return 0.5  # Neutral if no analysis specified

    # Extract core labels
    dataset_affordances = {
        _extract_label_core(a) for a in (fingerprint.analysis_affordance_ids or [])
    }
    query_terms_lower = {t.lower().replace("_", " ") for t in query_terms}

    # Direct match
    if dataset_affordances and query_terms_lower:
        direct_match = len(query_terms_lower & dataset_affordances) / len(
            query_terms_lower
        )
        if direct_match > 0:
            return min(direct_match + 0.2, 1.0)

    return 0.0


def _compute_design_relevance(
    fingerprint: SemanticDatasetFingerprint,
    parsed_query: dict[str, Any],
) -> float:
    """Compute experimental design type relevance."""
    # Infer expected design from query
    query = parsed_query.get("query", "").lower()
    tasks = [t.lower() for t in parsed_query.get("tasks", [])]

    dataset_design = (fingerprint.design_type or "").lower()

    # Check for design-related terms in query
    design_terms = {
        "2afc": ["2afc", "two alternative", "forced choice"],
        "go_nogo": ["go no", "go/no", "nogo"],
        "reversal_learning": ["reversal", "reversal learning"],
        "delay_discounting": ["delay", "discounting", "temporal"],
        "multi_armed_bandit": ["bandit", "multi-arm", "explore exploit"],
        "free_behavior": ["free", "natural", "spontaneous"],
        "classical_conditioning": ["pavlov", "conditioning", "associative"],
        "operant_conditioning": ["operant", "instrumental"],
    }

    for design_type, terms in design_terms.items():
        if any(term in query for term in terms):
            if design_type == dataset_design:
                return 0.9  # Strong match
            elif dataset_design:
                return 0.3  # Different design
            else:
                return 0.5  # Unknown design

    # Check if tasks imply a design
    for task in tasks:
        for design_type, terms in design_terms.items():
            if any(term in task for term in terms):
                if design_type == dataset_design:
                    return 0.8

    return 0.5  # Neutral


def _find_semantically_similar_datasets(
    fingerprint: SemanticDatasetFingerprint,
    index: SemanticSearchIndex,
    top_k: int = 3,
    min_similarity: float = 0.6,
) -> list[str]:
    """Find datasets semantically similar to the given fingerprint."""
    from neural_search.embeddings.semantic_fingerprint import (
        compute_semantic_similarity,
    )

    results: list[tuple[str, float]] = []

    for dataset_id, other_fp in index.fingerprints.items():
        if dataset_id == fingerprint.dataset_id:
            continue

        sim = compute_semantic_similarity(fingerprint, other_fp)
        if sim.combined_similarity >= min_similarity:
            results.append((dataset_id, sim.combined_similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in results[:top_k]]


def augment_result_with_semantic_score(
    result: Any,
    index: SemanticSearchIndex,
    parsed_query: dict[str, Any],
    weight: float = 0.08,
) -> None:
    """Augment a search result with semantic fingerprint scoring.

    Args:
        result: SearchResult to augment
        index: Semantic search index
        parsed_query: Parsed query
        weight: Weight for semantic score in final score
    """
    semantic = compute_semantic_score_for_result(
        index,
        str(result.dataset_id),
        parsed_query,
    )

    # Add to score breakdown
    result.score_breakdown["semantic_fingerprint_score"] = semantic.score
    result.score_breakdown["task_relevance"] = semantic.task_relevance
    result.score_breakdown["modality_relevance"] = semantic.modality_relevance
    result.score_breakdown["behavior_relevance"] = semantic.behavior_relevance
    result.score_breakdown["analysis_relevance"] = semantic.analysis_relevance

    # Update final score
    base_final = result.score_breakdown.get("final_score", result.score / 100.0)
    new_final = min(base_final + weight * semantic.score, 1.0)
    result.score_breakdown["final_score"] = round(new_final, 3)
    result.score = round(new_final * 100, 2)

    # Add explanation
    if semantic.score > 0.3:
        result.why_matched.append(semantic.explanation)

    # Add similar datasets info
    if semantic.similar_datasets:
        if not hasattr(result, "similar_datasets"):
            result.similar_datasets = []
        result.similar_datasets.extend(semantic.similar_datasets)
