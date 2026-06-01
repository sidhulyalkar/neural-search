"""Intent-aware usefulness scorer for latent future usefulness ranking.

Scores a candidate dataset against a query context across 10 dimensions,
weighted by the user's UsefulnessIntent.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

from neural_search.retrieval.query_intent import UsefulnessIntent

if TYPE_CHECKING:
    from neural_search.graph.schema import KnowledgeGraph


@dataclass
class DatasetContext:
    """Structured metadata context for scoring. Works as both query and candidate."""
    dataset_id: str
    modalities: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    species: list[str] = field(default_factory=list)
    brain_regions: list[str] = field(default_factory=list)
    affordances: list[str] = field(default_factory=list)
    data_standards: list[str] = field(default_factory=list)
    session_count: int | None = None
    trial_count: int | None = None
    subject_count: int | None = None
    has_timestamps: bool = False
    quality_score: float = 0.0


@dataclass
class UsefulnessScore:
    total_score: float
    intent: UsefulnessIntent
    dimension_scores: dict[str, float]
    weights: dict[str, float]
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# Normalized weight profiles (must sum to 1.0) per intent.
# neural_signature_similarity is always 0.0 in v0.8 (future work).
INTENT_WEIGHT_PROFILES: dict[UsefulnessIntent, dict[str, float]] = {
    UsefulnessIntent.STRICT_LOOKUP: {
        "modality_alignment": 0.18,
        "task_compatibility": 0.18,
        "species_match": 0.12,
        "region_overlap": 0.12,
        "affordance_compatibility": 0.16,
        "graph_proximity": 0.08,
        "provenance_quality": 0.08,
        "statistical_power": 0.05,
        "pipeline_transferability": 0.03,
        "neural_signature_similarity": 0.00,
    },
    UsefulnessIntent.PIPELINE_REUSE: {
        "modality_alignment": 0.18,
        "affordance_compatibility": 0.24,
        "pipeline_transferability": 0.22,
        "provenance_quality": 0.10,
        "statistical_power": 0.08,
        "graph_proximity": 0.08,
        "task_compatibility": 0.05,
        "species_match": 0.03,
        "region_overlap": 0.02,
        "neural_signature_similarity": 0.00,
    },
    UsefulnessIntent.REPLICATION: {
        "task_compatibility": 0.22,
        "species_match": 0.16,
        "region_overlap": 0.16,
        "modality_alignment": 0.14,
        "affordance_compatibility": 0.12,
        "provenance_quality": 0.08,
        "statistical_power": 0.07,
        "graph_proximity": 0.03,
        "pipeline_transferability": 0.02,
        "neural_signature_similarity": 0.00,
    },
    UsefulnessIntent.META_ANALYSIS: {
        "task_compatibility": 0.18,
        "provenance_quality": 0.16,
        "statistical_power": 0.16,
        "affordance_compatibility": 0.14,
        "species_match": 0.10,
        "modality_alignment": 0.10,
        "graph_proximity": 0.08,
        "region_overlap": 0.05,
        "pipeline_transferability": 0.03,
        "neural_signature_similarity": 0.00,
    },
    UsefulnessIntent.EXPLORATION: {
        "graph_proximity": 0.20,
        "affordance_compatibility": 0.16,
        "neural_signature_similarity": 0.16,
        "task_compatibility": 0.12,
        "modality_alignment": 0.10,
        "provenance_quality": 0.10,
        "pipeline_transferability": 0.08,
        "species_match": 0.04,
        "region_overlap": 0.04,
        "statistical_power": 0.00,
    },
    UsefulnessIntent.CROSS_DATASET_COMPARISON: {
        "task_compatibility": 0.20,
        "species_match": 0.18,
        "modality_alignment": 0.15,
        "region_overlap": 0.14,
        "affordance_compatibility": 0.12,
        "statistical_power": 0.08,
        "provenance_quality": 0.07,
        "graph_proximity": 0.04,
        "pipeline_transferability": 0.02,
        "neural_signature_similarity": 0.00,
    },
    UsefulnessIntent.METHOD_TRANSFER: {
        "affordance_compatibility": 0.26,
        "task_compatibility": 0.20,
        "modality_alignment": 0.16,
        "pipeline_transferability": 0.14,
        "species_match": 0.08,
        "region_overlap": 0.06,
        "provenance_quality": 0.05,
        "statistical_power": 0.05,
        "graph_proximity": 0.00,
        "neural_signature_similarity": 0.00,
    },
}


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = {s.lower() for s in a}, {s.lower() for s in b}
    if not sa and not sb:
        return 0.5  # unknown -> neutral
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _log_power(count: int | None) -> float:
    if count is None or count == 0:
        return 0.3  # neutral when unknown
    return min(1.0, math.log1p(count) / math.log1p(10000))


def score_usefulness(
    query_context: DatasetContext,
    candidate: DatasetContext,
    intent: UsefulnessIntent | None = None,
    graph: "KnowledgeGraph | None" = None,
) -> UsefulnessScore:
    """Score a candidate dataset against a query context for latent usefulness."""
    if intent is None:
        intent = UsefulnessIntent.STRICT_LOOKUP

    weights = INTENT_WEIGHT_PROFILES.get(intent, INTENT_WEIGHT_PROFILES[UsefulnessIntent.STRICT_LOOKUP])
    evidence: list[str] = []
    warnings: list[str] = []

    dims: dict[str, float] = {}

    # modality_alignment
    ma = _jaccard(query_context.modalities, candidate.modalities)
    dims["modality_alignment"] = ma
    if query_context.modalities and candidate.modalities:
        shared = {m.lower() for m in query_context.modalities} & {m.lower() for m in candidate.modalities}
        evidence.append(f"Shared modalities: {sorted(shared) or 'none'}")

    # task_compatibility
    tc = _jaccard(query_context.tasks, candidate.tasks)
    dims["task_compatibility"] = tc
    if query_context.tasks and candidate.tasks:
        shared = {t.lower() for t in query_context.tasks} & {t.lower() for t in candidate.tasks}
        evidence.append(f"Shared tasks: {sorted(shared) or 'none'}")

    # species_match
    dims["species_match"] = _jaccard(query_context.species, candidate.species)

    # region_overlap
    dims["region_overlap"] = _jaccard(query_context.brain_regions, candidate.brain_regions)

    # affordance_compatibility
    dims["affordance_compatibility"] = _jaccard(query_context.affordances, candidate.affordances)
    if query_context.affordances:
        shared = {a.lower() for a in query_context.affordances} & {a.lower() for a in candidate.affordances}
        evidence.append(f"Shared affordances: {sorted(shared) or 'none'}")

    # s9: graph_proximity
    if graph is not None:
        from neural_search.retrieval.graph_usefulness import normalized_metapath_score
        graph_dict = graph.model_dump(mode="json")
        node_ids = graph_dict.get("nodes", {})

        def _resolve_node_id(did: str) -> str | None:
            if not did:
                return None
            if did in node_ids:
                return did
            # Graph nodes use "node:" prefix; dataset_ids use "dataset:source:id"
            prefixed = f"node:{did}"
            if prefixed in node_ids:
                return prefixed
            return None

        q_node = _resolve_node_id(query_context.dataset_id)
        c_node = _resolve_node_id(candidate.dataset_id)
        if q_node and c_node:
            graph_proximity = normalized_metapath_score(
                graph_dict, q_node, c_node, "dataset_has_task",
            )
        else:
            graph_proximity = 0.3
            warnings.append(
                "graph_proximity: dataset_id(s) not found in graph nodes; using neutral prior 0.3"
            )
    else:
        graph_proximity = 0.3
        warnings.append("graph_proximity: using neutral prior 0.3 (graph not available)")
    dims["graph_proximity"] = graph_proximity

    # provenance_quality — quality_score is 0-1 float
    dims["provenance_quality"] = min(1.0, max(0.0, candidate.quality_score))
    if candidate.quality_score > 0:
        evidence.append(f"Candidate quality score: {candidate.quality_score:.2f}")

    # statistical_power
    power = max(
        _log_power(candidate.trial_count),
        _log_power(candidate.session_count),
    )
    dims["statistical_power"] = power

    # pipeline_transferability — based on shared data standards
    dims["pipeline_transferability"] = _jaccard(query_context.data_standards, candidate.data_standards)
    if query_context.data_standards and candidate.data_standards:
        shared = {s.lower() for s in query_context.data_standards} & {s.lower() for s in candidate.data_standards}
        evidence.append(f"Shared data standards: {sorted(shared) or 'none'}")

    # neural_signature_similarity — not implemented in v0.8
    dims["neural_signature_similarity"] = 0.0
    warnings.append("neural_signature_similarity: not implemented in v0.8; score fixed at 0.0")

    # Weighted sum
    total = sum(weights.get(d, 0.0) * v for d, v in dims.items())

    return UsefulnessScore(
        total_score=min(1.0, max(0.0, total)),
        intent=intent,
        dimension_scores=dims,
        weights=dict(weights),
        evidence=evidence,
        warnings=warnings,
    )
