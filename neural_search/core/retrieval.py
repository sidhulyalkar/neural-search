"""Multi-Stage Retrieval Pipeline.

This module implements a modular, multi-stage retrieval pipeline that:

Stage A: Candidate Generation
- Lexical search (keyword/BM25-style matching)
- Metadata/entity filters (species, modality, task, region)
- Embedding search (dense retrieval)
- Graph-neighborhood expansion
- Synonym/ontology expansion

Stage B: Fusion
- Combine scores from multiple sources
- Query-intent-specific weighting
- Configurable fusion strategy

Stage C: Reranking
- Deterministic reranking based on combined signals
- Optional cross-encoder reranking (scaffolded)

Stage D: Calibration and Uncertainty
- Separate relevance score from confidence
- Flag uncertain results
- Track provenance strength

Stage E: Explanation
- Explain why each result matched
- Show which constraints were satisfied
- Indicate graph relationships that contributed

The pipeline is designed to be:
- Modular: Stages can be enabled/disabled
- Explainable: Every score is traceable
- Extensible: New stages can be added
- Testable: Deterministic with fixture data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel
from pydantic import Field as PydanticField

from neural_search.core.query import QueryPlan, RetrievalStage
from neural_search.core.records import ScientificRecord


class CandidateSource(StrEnum):
    """Source of a candidate result."""

    LEXICAL = "lexical"
    METADATA = "metadata"
    EMBEDDING = "embedding"
    GRAPH = "graph"
    ONTOLOGY = "ontology"
    PAPER_LINK = "paper_link"
    AFFORDANCE = "affordance"


@dataclass
class ScoreComponent:
    """A single component of a result's score."""

    source: CandidateSource
    score: float              # 0.0 to 1.0
    weight: float             # Weight used in fusion
    evidence: str | None = None
    matched_terms: list[str] = field(default_factory=list)


@dataclass
class ScoredCandidate:
    """A candidate result with all its scoring components."""

    record_id: str
    record: ScientificRecord | dict[str, Any] | None = None

    # Scores
    final_score: float = 0.0
    components: list[ScoreComponent] = field(default_factory=list)

    # Explanation
    why_matched: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    matched_constraints: list[str] = field(default_factory=list)
    unmatched_constraints: list[str] = field(default_factory=list)

    # Confidence/uncertainty
    relevance_score: float = 0.0       # How relevant is this result?
    confidence_score: float = 0.0      # How confident are we in the score?
    uncertainty_flags: list[str] = field(default_factory=list)
    provenance_strength: float = 0.0

    # Graph context
    graph_edges_used: list[str] = field(default_factory=list)
    linked_papers: list[str] = field(default_factory=list)

    def get_score_breakdown(self) -> dict[str, float]:
        """Get a breakdown of score by source."""
        return {
            comp.source.value: round(comp.score * comp.weight, 4)
            for comp in self.components
        }


@dataclass
class RetrievalStageResult:
    """Result from a single retrieval stage."""

    stage: RetrievalStage
    candidates: list[ScoredCandidate]
    latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)


class RetrievalResult(BaseModel):
    """Final result from the multi-stage retrieval pipeline."""

    # Query info
    query_text: str
    query_plan: QueryPlan

    # Results
    results: list[dict[str, Any]] = PydanticField(default_factory=list)
    total_candidates: int = 0

    # Stage summaries
    stage_results: dict[str, dict[str, Any]] = PydanticField(default_factory=dict)

    # Metrics
    total_latency_ms: float = 0.0
    stages_executed: list[str] = PydanticField(default_factory=list)

    # Explanation
    retrieval_notes: list[str] = PydanticField(default_factory=list)


class CandidateGenerator(Protocol):
    """Protocol for candidate generation stages."""

    def generate(
        self,
        query_plan: QueryPlan,
        corpus: list[dict[str, Any]],
    ) -> list[ScoredCandidate]:
        """Generate candidates from the corpus."""
        ...


class Scorer(Protocol):
    """Protocol for scoring stages."""

    def score(
        self,
        candidates: list[ScoredCandidate],
        query_plan: QueryPlan,
    ) -> list[ScoredCandidate]:
        """Score or re-score candidates."""
        ...


class Reranker(Protocol):
    """Protocol for reranking stages."""

    def rerank(
        self,
        candidates: list[ScoredCandidate],
        query_plan: QueryPlan,
        top_k: int,
    ) -> list[ScoredCandidate]:
        """Rerank candidates."""
        ...


def _normalize_text(text: str) -> str:
    """Simple text normalization."""
    return text.lower().strip()


def _text_overlap_score(query_terms: list[str], text: str) -> tuple[float, list[str]]:
    """Compute term overlap between query and text."""
    if not query_terms or not text:
        return 0.0, []

    normalized_text = _normalize_text(text)
    matched = [t for t in query_terms if _normalize_text(t) in normalized_text]
    score = len(matched) / len(query_terms) if query_terms else 0.0
    return score, matched


def _list_overlap_score(
    query_values: list[str],
    record_values: list[str],
) -> tuple[float, list[str]]:
    """Compute overlap between two lists of values."""
    if not query_values:
        return 0.0, []

    query_set = {_normalize_text(v) for v in query_values}
    record_set = {_normalize_text(v) for v in record_values}

    matched = query_set & record_set
    score = len(matched) / len(query_set) if query_set else 0.0
    return score, list(matched)


class LexicalGenerator:
    """Generate candidates via lexical/keyword matching."""

    def generate(
        self,
        query_plan: QueryPlan,
        corpus: list[dict[str, Any]],
    ) -> list[ScoredCandidate]:
        candidates: list[ScoredCandidate] = []
        query_terms = query_plan.detected_keywords

        for record in corpus:
            record_id = record.get("dataset_id") or record.get("record_id") or record.get("id", "unknown")

            # Combine text fields
            text = " ".join([
                str(record.get("title", "")),
                str(record.get("description", "")),
                str(record.get("abstract", "")),
            ])

            score, matched = _text_overlap_score(query_terms, text)

            if score > 0:
                candidates.append(ScoredCandidate(
                    record_id=str(record_id),
                    record=record,
                    components=[ScoreComponent(
                        source=CandidateSource.LEXICAL,
                        score=score,
                        weight=1.0,
                        evidence=f"Matched {len(matched)} terms",
                        matched_terms=matched,
                    )],
                    why_matched=[f"Lexical match: {', '.join(matched[:5])}"] if matched else [],
                    matched_terms=matched,
                ))

        return candidates


class OntologyGenerator:
    """Generate candidates via ontology matching."""

    def generate(
        self,
        query_plan: QueryPlan,
        corpus: list[dict[str, Any]],
    ) -> list[ScoredCandidate]:
        candidates: list[ScoredCandidate] = []

        for record in corpus:
            record_id = record.get("dataset_id") or record.get("record_id") or record.get("id", "unknown")
            components: list[ScoreComponent] = []
            why_matched: list[str] = []
            all_matched: list[str] = []
            matched_constraints: list[str] = []
            unmatched_constraints: list[str] = []

            # Task matching
            if query_plan.required_tasks:
                record_tasks = self._get_labels(record, "tasks")
                score, matched = _list_overlap_score(query_plan.required_tasks, record_tasks)
                components.append(ScoreComponent(
                    source=CandidateSource.ONTOLOGY,
                    score=score,
                    weight=query_plan.weight_overrides.get("ontology", 0.25),
                    evidence=f"Task overlap: {len(matched)}/{len(query_plan.required_tasks)}",
                    matched_terms=matched,
                ))
                if matched:
                    why_matched.append(f"Tasks matched: {', '.join(matched)}")
                    matched_constraints.extend([f"task:{t}" for t in matched])
                    all_matched.extend(matched)
                else:
                    unmatched_constraints.extend([f"task:{t}" for t in query_plan.required_tasks])

            # Modality matching
            if query_plan.required_modalities:
                record_modalities = self._get_labels(record, "modalities")
                score, matched = _list_overlap_score(query_plan.required_modalities, record_modalities)
                components.append(ScoreComponent(
                    source=CandidateSource.ONTOLOGY,
                    score=score,
                    weight=query_plan.weight_overrides.get("modality", 0.15),
                    evidence="Modality overlap",
                    matched_terms=matched,
                ))
                if matched:
                    why_matched.append(f"Modalities matched: {', '.join(matched)}")
                    matched_constraints.extend([f"modality:{m}" for m in matched])
                    all_matched.extend(matched)
                else:
                    unmatched_constraints.extend([f"modality:{m}" for m in query_plan.required_modalities])

            # Species matching
            if query_plan.required_species:
                record_species = self._get_labels(record, "species")
                score, matched = _list_overlap_score(query_plan.required_species, record_species)
                components.append(ScoreComponent(
                    source=CandidateSource.ONTOLOGY,
                    score=score,
                    weight=query_plan.weight_overrides.get("metadata", 0.10),
                    evidence="Species overlap",
                    matched_terms=matched,
                ))
                if matched:
                    why_matched.append(f"Species matched: {', '.join(matched)}")
                    matched_constraints.extend([f"species:{s}" for s in matched])
                    all_matched.extend(matched)
                else:
                    unmatched_constraints.extend([f"species:{s}" for s in query_plan.required_species])

            # Brain region matching
            if query_plan.required_regions:
                record_regions = self._get_labels(record, "brain_regions")
                score, matched = _list_overlap_score(query_plan.required_regions, record_regions)
                components.append(ScoreComponent(
                    source=CandidateSource.ONTOLOGY,
                    score=score,
                    weight=query_plan.weight_overrides.get("metadata", 0.10),
                    evidence="Region overlap",
                    matched_terms=matched,
                ))
                if matched:
                    why_matched.append(f"Brain regions matched: {', '.join(matched)}")
                    matched_constraints.extend([f"region:{r}" for r in matched])
                    all_matched.extend(matched)
                else:
                    unmatched_constraints.extend([f"region:{r}" for r in query_plan.required_regions])

            # Only include if there's some match
            if components and any(c.score > 0 for c in components):
                candidates.append(ScoredCandidate(
                    record_id=str(record_id),
                    record=record,
                    components=components,
                    why_matched=why_matched,
                    matched_terms=all_matched,
                    matched_constraints=matched_constraints,
                    unmatched_constraints=unmatched_constraints,
                ))

        return candidates

    def _get_labels(self, record: dict[str, Any], field: str) -> list[str]:
        """Extract label strings from a field."""
        values = record.get(field, [])
        if not values:
            return []

        labels = []
        for v in values:
            if isinstance(v, dict):
                labels.append(str(v.get("label", v.get("id", ""))))
            else:
                labels.append(str(v))
        return [label for label in labels if label]


class AffordanceGenerator:
    """Generate candidates via analysis affordance matching."""

    def generate(
        self,
        query_plan: QueryPlan,
        corpus: list[dict[str, Any]],
    ) -> list[ScoredCandidate]:
        candidates: list[ScoredCandidate] = []

        if not query_plan.required_analyses:
            return candidates

        for record in corpus:
            record_id = record.get("dataset_id") or record.get("record_id") or record.get("id", "unknown")

            # Get affordances from record
            affordances = record.get("analysis_affordances", [])
            if not affordances:
                affordances = record.get("suggested_analyses", [])

            affordance_ids = []
            for aff in affordances:
                if isinstance(aff, dict):
                    affordance_ids.append(str(aff.get("analysis_id", aff.get("id", ""))))
                else:
                    affordance_ids.append(str(aff))

            score, matched = _list_overlap_score(query_plan.required_analyses, affordance_ids)

            if score > 0:
                candidates.append(ScoredCandidate(
                    record_id=str(record_id),
                    record=record,
                    components=[ScoreComponent(
                        source=CandidateSource.AFFORDANCE,
                        score=score,
                        weight=query_plan.weight_overrides.get("affordance", 0.15),
                        evidence=f"Supports analyses: {', '.join(matched)}",
                        matched_terms=matched,
                    )],
                    why_matched=[f"Supports analysis: {', '.join(matched)}"],
                    matched_terms=matched,
                ))

        return candidates


class EmbeddingGenerator:
    """Generate candidates via embedding-based semantic search.

    Uses dense vector similarity to find semantically related records,
    even when there's no lexical overlap.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_cache: bool = True,
    ):
        self.model_name = model_name
        self.use_cache = use_cache
        self._embedding_cache: dict[str, list[float]] = {}
        self._provider: Any = None

    def _get_provider(self) -> Any:
        """Lazy-load the embedding provider."""
        if self._provider is None:
            try:
                from neural_search.embeddings.sentence_transformers import (
                    SentenceTransformerEmbeddingProvider,
                )
                self._provider = SentenceTransformerEmbeddingProvider(self.model_name)
            except ImportError:
                # Fall back to hashing embeddings for testing
                from neural_search.embeddings.hashing import HashingEmbeddingProvider
                self._provider = HashingEmbeddingProvider()
        return self._provider

    def _embed_text(self, text: str) -> list[float]:
        """Get embedding for a text, using cache if available."""
        if self.use_cache and text in self._embedding_cache:
            return self._embedding_cache[text]

        provider = self._get_provider()
        embedding = provider.embed_text(text)

        if self.use_cache:
            self._embedding_cache[text] = embedding
        return embedding

    def _cosine_similarity(self, vec_a: list[float], vec_b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def _get_record_text(self, record: dict[str, Any]) -> str:
        """Extract searchable text from a record."""
        parts = [
            record.get("title", ""),
            record.get("description", ""),
            record.get("abstract", ""),
            record.get("scientific_summary", ""),
        ]
        # Add metadata
        tasks = record.get("tasks", [])
        if tasks:
            task_strs = [t.get("label", str(t)) if isinstance(t, dict) else str(t) for t in tasks]
            parts.append(" ".join(task_strs))
        modalities = record.get("modalities", [])
        if modalities:
            mod_strs = [m.get("label", str(m)) if isinstance(m, dict) else str(m) for m in modalities]
            parts.append(" ".join(mod_strs))

        return " ".join(p for p in parts if p)

    def generate(
        self,
        query_plan: QueryPlan,
        corpus: list[dict[str, Any]],
        min_similarity: float = 0.3,
    ) -> list[ScoredCandidate]:
        """Generate candidates based on embedding similarity.

        Args:
            query_plan: The query plan with text and constraints
            corpus: List of records to search
            min_similarity: Minimum similarity threshold (0-1)

        Returns:
            List of candidates above the similarity threshold
        """
        candidates: list[ScoredCandidate] = []

        # Embed the query
        query_embedding = self._embed_text(query_plan.query_text)

        for record in corpus:
            record_id = record.get("dataset_id") or record.get("record_id") or record.get("id", "unknown")

            # Get or compute record embedding
            record_text = self._get_record_text(record)
            if not record_text:
                continue

            record_embedding = self._embed_text(record_text)
            similarity = self._cosine_similarity(query_embedding, record_embedding)

            if similarity >= min_similarity:
                candidates.append(ScoredCandidate(
                    record_id=str(record_id),
                    record=record,
                    components=[ScoreComponent(
                        source=CandidateSource.EMBEDDING,
                        score=similarity,
                        weight=query_plan.weight_overrides.get("semantic", 0.10),
                        evidence=f"Semantic similarity: {similarity:.3f}",
                    )],
                    why_matched=[f"Semantic match (similarity: {similarity:.2f})"],
                ))

        # Sort by similarity
        candidates.sort(key=lambda c: c.components[0].score if c.components else 0, reverse=True)
        return candidates


class ScoreFuser:
    """Fuse scores from multiple candidate sources."""

    def fuse(
        self,
        candidate_lists: list[list[ScoredCandidate]],
        query_plan: QueryPlan,
    ) -> list[ScoredCandidate]:
        """Fuse candidates from multiple sources into a single ranked list."""
        # Merge candidates by record_id
        merged: dict[str, ScoredCandidate] = {}

        for candidates in candidate_lists:
            for cand in candidates:
                if cand.record_id not in merged:
                    merged[cand.record_id] = ScoredCandidate(
                        record_id=cand.record_id,
                        record=cand.record,
                    )

                # Merge components
                merged[cand.record_id].components.extend(cand.components)
                merged[cand.record_id].why_matched.extend(cand.why_matched)
                merged[cand.record_id].matched_terms.extend(cand.matched_terms)
                merged[cand.record_id].matched_constraints.extend(cand.matched_constraints)
                merged[cand.record_id].unmatched_constraints.extend(cand.unmatched_constraints)

        # Compute final scores
        for cand in merged.values():
            # Deduplicate
            cand.why_matched = list(dict.fromkeys(cand.why_matched))
            cand.matched_terms = list(dict.fromkeys(cand.matched_terms))
            cand.matched_constraints = list(dict.fromkeys(cand.matched_constraints))
            cand.unmatched_constraints = list(dict.fromkeys(cand.unmatched_constraints))

            # Compute weighted sum
            total_score = sum(c.score * c.weight for c in cand.components)
            total_weight = sum(c.weight for c in cand.components)

            cand.final_score = total_score / total_weight if total_weight > 0 else 0.0
            cand.relevance_score = cand.final_score

            # Compute confidence based on number of signals
            unique_sources = {c.source for c in cand.components}
            cand.confidence_score = min(len(unique_sources) / 4, 1.0)

            # Add uncertainty flags
            if cand.unmatched_constraints:
                cand.uncertainty_flags.append(
                    f"Unmatched constraints: {len(cand.unmatched_constraints)}"
                )

        return sorted(merged.values(), key=lambda x: x.final_score, reverse=True)


class DeterministicReranker:
    """Deterministic reranking based on secondary signals."""

    def rerank(
        self,
        candidates: list[ScoredCandidate],
        query_plan: QueryPlan,
        top_k: int = 10,
    ) -> list[ScoredCandidate]:
        """Rerank candidates using secondary signals.

        Secondary signals include:
        - Provenance strength (linked papers, DOI)
        - Metadata completeness
        - Constraint satisfaction rate
        """
        for cand in candidates:
            # Compute provenance boost
            record = cand.record or {}
            provenance_boost = 0.0

            if record.get("linked_papers") or record.get("linked_paper_ids"):
                provenance_boost += 0.05
            if record.get("doi"):
                provenance_boost += 0.03

            # Compute constraint satisfaction rate
            total_constraints = len(cand.matched_constraints) + len(cand.unmatched_constraints)
            if total_constraints > 0:
                satisfaction = len(cand.matched_constraints) / total_constraints
                cand.final_score += 0.02 * satisfaction

            # Apply provenance boost
            cand.final_score += provenance_boost
            cand.provenance_strength = provenance_boost * 10  # Scale to 0-1 range

            # Clamp score
            cand.final_score = min(1.0, max(0.0, cand.final_score))

        # Re-sort
        candidates.sort(key=lambda x: x.final_score, reverse=True)
        return candidates[:top_k]


class MultiStageRetriever:
    """Multi-stage retrieval pipeline.

    Executes retrieval in configurable stages:
    1. Candidate generation (lexical, ontology, embedding, graph)
    2. Score fusion
    3. Reranking
    4. Explanation generation

    Example:
        ```python
        retriever = MultiStageRetriever()
        plan = parse_and_plan_query("Find datasets for latent-state modeling")
        result = retriever.search(plan, corpus)
        ```
    """

    def __init__(
        self,
        lexical_generator: LexicalGenerator | None = None,
        ontology_generator: OntologyGenerator | None = None,
        affordance_generator: AffordanceGenerator | None = None,
        embedding_generator: EmbeddingGenerator | None = None,
        fuser: ScoreFuser | None = None,
        reranker: DeterministicReranker | None = None,
        use_embeddings: bool = True,
    ):
        self.lexical_generator = lexical_generator or LexicalGenerator()
        self.ontology_generator = ontology_generator or OntologyGenerator()
        self.affordance_generator = affordance_generator or AffordanceGenerator()
        self.embedding_generator = embedding_generator or (EmbeddingGenerator() if use_embeddings else None)
        self.fuser = fuser or ScoreFuser()
        self.reranker = reranker or DeterministicReranker()

    def search(
        self,
        query_plan: QueryPlan,
        corpus: list[dict[str, Any]],
        top_k: int = 10,
    ) -> RetrievalResult:
        """Execute multi-stage retrieval.

        Args:
            query_plan: Structured query plan from parse_and_plan_query
            corpus: List of record dictionaries to search
            top_k: Number of results to return

        Returns:
            RetrievalResult with ranked results and explanations
        """
        import time

        start_time = time.time()
        stage_results: dict[str, dict[str, Any]] = {}
        stages_executed: list[str] = []
        candidate_lists: list[list[ScoredCandidate]] = []

        # Stage A: Candidate Generation

        # A1: Lexical
        if query_plan.is_stage_enabled(RetrievalStage.LEXICAL):
            stage_start = time.time()
            lexical_candidates = self.lexical_generator.generate(query_plan, corpus)
            candidate_lists.append(lexical_candidates)
            stages_executed.append("lexical")
            stage_results["lexical"] = {
                "count": len(lexical_candidates),
                "latency_ms": (time.time() - stage_start) * 1000,
            }

        # A2: Ontology
        if query_plan.is_stage_enabled(RetrievalStage.ONTOLOGY_MATCH):
            stage_start = time.time()
            ontology_candidates = self.ontology_generator.generate(query_plan, corpus)
            candidate_lists.append(ontology_candidates)
            stages_executed.append("ontology")
            stage_results["ontology"] = {
                "count": len(ontology_candidates),
                "latency_ms": (time.time() - stage_start) * 1000,
            }

        # A3: Affordance
        if query_plan.is_stage_enabled(RetrievalStage.AFFORDANCE_MATCH):
            stage_start = time.time()
            affordance_candidates = self.affordance_generator.generate(query_plan, corpus)
            candidate_lists.append(affordance_candidates)
            stages_executed.append("affordance")
            stage_results["affordance"] = {
                "count": len(affordance_candidates),
                "latency_ms": (time.time() - stage_start) * 1000,
            }

        # A4: Embedding (semantic search)
        if query_plan.is_stage_enabled(RetrievalStage.EMBEDDING_SEARCH) and self.embedding_generator:
            stage_start = time.time()
            embedding_candidates = self.embedding_generator.generate(query_plan, corpus)
            candidate_lists.append(embedding_candidates)
            stages_executed.append("embedding")
            stage_results["embedding"] = {
                "count": len(embedding_candidates),
                "latency_ms": (time.time() - stage_start) * 1000,
            }

        # Stage B: Fusion
        stage_start = time.time()
        fused_candidates = self.fuser.fuse(candidate_lists, query_plan)
        stages_executed.append("fusion")
        stage_results["fusion"] = {
            "count": len(fused_candidates),
            "latency_ms": (time.time() - stage_start) * 1000,
        }

        # Stage C: Reranking
        if query_plan.is_stage_enabled(RetrievalStage.RERANK):
            stage_start = time.time()
            reranked = self.reranker.rerank(fused_candidates, query_plan, top_k)
            stages_executed.append("rerank")
            stage_results["rerank"] = {
                "count": len(reranked),
                "latency_ms": (time.time() - stage_start) * 1000,
            }
        else:
            reranked = fused_candidates[:top_k]

        # Stage E: Convert to result format
        results = []
        for cand in reranked:
            record = cand.record or {}
            results.append({
                "dataset_id": cand.record_id,
                "score": round(cand.final_score * 100, 2),
                "relevance_score": round(cand.relevance_score, 4),
                "confidence_score": round(cand.confidence_score, 4),
                "why_matched": cand.why_matched,
                "matched_terms": cand.matched_terms,
                "matched_constraints": cand.matched_constraints,
                "unmatched_constraints": cand.unmatched_constraints,
                "warnings": cand.warnings,
                "uncertainty_flags": cand.uncertainty_flags,
                "provenance_strength": round(cand.provenance_strength, 4),
                "score_breakdown": cand.get_score_breakdown(),
                "title": record.get("title", ""),
                "source": record.get("source", ""),
            })

        total_latency = (time.time() - start_time) * 1000

        # Build retrieval notes
        retrieval_notes = [
            f"Intent: {query_plan.primary_intent.value} (confidence: {query_plan.intent_confidence:.2f})",
            f"Stages executed: {', '.join(stages_executed)}",
            f"Total candidates: {len(fused_candidates)}",
        ]
        retrieval_notes.extend(query_plan.planning_notes)

        return RetrievalResult(
            query_text=query_plan.query_text,
            query_plan=query_plan,
            results=results,
            total_candidates=len(fused_candidates),
            stage_results=stage_results,
            total_latency_ms=total_latency,
            stages_executed=stages_executed,
            retrieval_notes=retrieval_notes,
        )


def search_with_plan(
    query: str,
    corpus: list[dict[str, Any]],
    top_k: int = 10,
) -> RetrievalResult:
    """Convenience function to parse query and search in one call.

    Args:
        query: Natural language search query
        corpus: List of record dictionaries to search
        top_k: Number of results to return

    Returns:
        RetrievalResult with ranked results
    """
    from neural_search.core.query import parse_and_plan_query

    plan = parse_and_plan_query(query)
    retriever = MultiStageRetriever()
    return retriever.search(plan, corpus, top_k)
