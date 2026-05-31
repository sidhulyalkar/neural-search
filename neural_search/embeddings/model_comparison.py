"""Embedding model comparison framework.

This module provides tools for comparing embedding models on retrieval tasks:

1. Run the same queries across multiple embedding models
2. Compute standard retrieval metrics for each
3. Generate comparison reports
4. Identify model strengths/weaknesses by query type

Usage:
    from neural_search.embeddings.model_comparison import (
        compare_embedding_models,
        generate_comparison_report,
    )

    results = compare_embedding_models(
        queries=["decision-making task", "calcium imaging V1"],
        corpus=dataset_cards,
        models=["hashing", "sentence-transformer", "specter2"],
        relevance_labels=labels,
    )

    report = generate_comparison_report(results)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from neural_search.embeddings.providers import (
    EmbeddingProviderBase,
    check_provider_availability,
    get_provider,
)

logger = logging.getLogger(__name__)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class ModelQueryResult(BaseModel):
    """Result for a single query with a single model."""

    query: str
    model_name: str
    result_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    embedding_latency_ms: float = 0.0
    search_latency_ms: float = 0.0


class ModelMetrics(BaseModel):
    """Metrics for a single model across all queries."""

    model_name: str
    provider: str
    dimension: int

    # Retrieval metrics
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0

    # Latency
    embedding_latency_mean_ms: float = 0.0
    embedding_latency_p95_ms: float = 0.0
    search_latency_mean_ms: float = 0.0

    # Coverage
    queries_with_results: int = 0
    total_queries: int = 0
    coverage: float = 0.0


class ModelComparisonReport(BaseModel):
    """Complete model comparison report."""

    models: list[str] = Field(default_factory=list)
    model_metrics: dict[str, ModelMetrics] = Field(default_factory=dict)

    # Rankings
    best_precision_model: str = ""
    best_mrr_model: str = ""
    best_recall_model: str = ""
    fastest_model: str = ""

    # Delta analysis
    model_deltas: dict[str, dict[str, float]] = Field(default_factory=dict)
    # model -> {vs_baseline_precision, vs_baseline_mrr, ...}

    # Per-query breakdown
    query_best_model: dict[str, str] = Field(default_factory=dict)
    # query -> best_model_for_that_query

    # Metadata
    corpus_size: int = 0
    query_count: int = 0
    baseline_model: str = ""
    generated_at: str = ""


@dataclass
class EmbeddingIndex:
    """Simple in-memory embedding index for comparison."""

    provider: EmbeddingProviderBase
    embeddings: dict[str, list[float]] = field(default_factory=dict)  # id -> vector
    ids: list[str] = field(default_factory=list)

    def add(self, entity_id: str, text: str) -> None:
        """Add an entity to the index."""
        embedding = self.provider.embed_text(text)
        self.embeddings[entity_id] = embedding
        if entity_id not in self.ids:
            self.ids.append(entity_id)

    def add_batch(self, entities: list[tuple[str, str]]) -> None:
        """Add multiple entities to the index."""
        if not entities:
            return

        ids, texts = zip(*entities)
        embeddings = self.provider.embed_batch(list(texts))

        for entity_id, embedding in zip(ids, embeddings):
            self.embeddings[entity_id] = embedding
            if entity_id not in self.ids:
                self.ids.append(entity_id)

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """Search the index for similar entities."""
        if not self.embeddings:
            return []

        query_embedding = self.provider.embed_text(query)
        scores = []

        for entity_id in self.ids:
            entity_embedding = self.embeddings[entity_id]
            score = cosine_similarity(query_embedding, entity_embedding)
            scores.append((entity_id, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def _compute_precision_at_k(result_ids: list[str], relevant: set[str], k: int) -> float:
    """Compute precision@k."""
    if not result_ids or not relevant:
        return 0.0
    top_k = result_ids[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / k


def _compute_recall_at_k(result_ids: list[str], relevant: set[str], k: int) -> float:
    """Compute recall@k."""
    if not relevant:
        return 0.0
    top_k = result_ids[:k]
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(relevant)


def _compute_mrr(result_ids: list[str], relevant: set[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    if not result_ids or not relevant:
        return 0.0
    for i, rid in enumerate(result_ids):
        if rid in relevant:
            return 1.0 / (i + 1)
    return 0.0


def _compute_ndcg_at_k(
    result_ids: list[str], relevance: dict[str, int], k: int
) -> float:
    """Compute NDCG@k with graded relevance."""
    if not result_ids or not relevance:
        return 0.0

    # DCG
    dcg = 0.0
    for i, rid in enumerate(result_ids[:k]):
        grade = relevance.get(rid, 0)
        dcg += (2**grade - 1) / math.log2(i + 2)

    # Ideal DCG
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))

    return dcg / idcg if idcg > 0 else 0.0


def compare_embedding_models(
    queries: list[str],
    corpus: list[dict[str, Any]],
    relevance_labels: dict[str, set[str]],
    models: list[str] | None = None,
    text_field: str = "text_card",
    id_field: str = "dataset_id",
    top_k: int = 10,
) -> ModelComparisonReport:
    """Compare embedding models on retrieval tasks.

    Args:
        queries: List of query strings
        corpus: List of dataset records (dicts with text_field and id_field)
        relevance_labels: Map from query -> set of relevant IDs
        models: List of model names to compare (default: all available)
        text_field: Field name for text to embed
        id_field: Field name for entity ID
        top_k: Number of results to retrieve

    Returns:
        ModelComparisonReport with metrics for each model
    """
    # Determine which models to use
    availability = check_provider_availability()

    if models is None:
        # Use all available models
        models = [name for name, available in availability.items() if available]

    # Validate models are available
    for model in models:
        if not availability.get(model, False):
            logger.warning(f"Model {model} not available, skipping")
            models = [m for m in models if m != model]

    if not models:
        raise ValueError("No embedding models available for comparison")

    logger.info(f"Comparing models: {models}")

    # Build indices and run evaluation for each model
    model_metrics: dict[str, ModelMetrics] = {}
    all_query_results: dict[str, list[ModelQueryResult]] = {}
    query_best: dict[str, str] = {}

    for model_name in models:
        logger.info(f"Evaluating model: {model_name}")

        # Get provider
        provider = get_provider(model_name)

        # Build index
        index = EmbeddingIndex(provider=provider)

        # Prepare corpus entries
        entries = []
        for record in corpus:
            entity_id = record.get(id_field)
            text = record.get(text_field, "")
            if entity_id and text:
                entries.append((entity_id, text))

        # Index corpus
        start = time.time()
        index.add_batch(entries)
        indexing_time = (time.time() - start) * 1000

        logger.info(
            f"  Indexed {len(entries)} documents in {indexing_time:.1f}ms"
        )

        # Run queries
        query_results: list[ModelQueryResult] = []
        precisions_5: list[float] = []
        precisions_10: list[float] = []
        recalls_5: list[float] = []
        recalls_10: list[float] = []
        mrrs: list[float] = []
        embedding_latencies: list[float] = []
        search_latencies: list[float] = []
        queries_with_results = 0

        for query in queries:
            start = time.time()
            results = index.search(query, top_k=top_k)
            search_time = (time.time() - start) * 1000

            result_ids = [r[0] for r in results]
            scores = [r[1] for r in results]

            qr = ModelQueryResult(
                query=query,
                model_name=model_name,
                result_ids=result_ids,
                scores=scores,
                search_latency_ms=search_time,
            )
            query_results.append(qr)

            # Compute metrics
            relevant = relevance_labels.get(query, set())
            precisions_5.append(_compute_precision_at_k(result_ids, relevant, 5))
            precisions_10.append(_compute_precision_at_k(result_ids, relevant, 10))
            recalls_5.append(_compute_recall_at_k(result_ids, relevant, 5))
            recalls_10.append(_compute_recall_at_k(result_ids, relevant, 10))
            mrrs.append(_compute_mrr(result_ids, relevant))
            search_latencies.append(search_time)

            if results:
                queries_with_results += 1

        all_query_results[model_name] = query_results

        # Aggregate metrics
        n = len(queries)
        search_latencies_sorted = sorted(search_latencies)

        metrics = ModelMetrics(
            model_name=model_name,
            provider=provider.provider_name,
            dimension=provider.dimension,
            precision_at_5=sum(precisions_5) / n if n else 0.0,
            precision_at_10=sum(precisions_10) / n if n else 0.0,
            recall_at_5=sum(recalls_5) / n if n else 0.0,
            recall_at_10=sum(recalls_10) / n if n else 0.0,
            mrr=sum(mrrs) / n if n else 0.0,
            search_latency_mean_ms=sum(search_latencies) / n if n else 0.0,
            queries_with_results=queries_with_results,
            total_queries=n,
            coverage=queries_with_results / n if n else 0.0,
        )

        model_metrics[model_name] = metrics
        logger.info(f"  P@5={metrics.precision_at_5:.3f}, MRR={metrics.mrr:.3f}")

    # Determine best model per query
    for i, query in enumerate(queries):
        best_model = ""
        best_mrr = -1.0
        relevant = relevance_labels.get(query, set())

        for model_name in models:
            result_ids = all_query_results[model_name][i].result_ids
            mrr = _compute_mrr(result_ids, relevant)
            if mrr > best_mrr:
                best_mrr = mrr
                best_model = model_name

        query_best[query] = best_model

    # Find best overall models
    best_precision = max(model_metrics.values(), key=lambda m: m.precision_at_5)
    best_mrr = max(model_metrics.values(), key=lambda m: m.mrr)
    best_recall = max(model_metrics.values(), key=lambda m: m.recall_at_10)
    fastest = min(model_metrics.values(), key=lambda m: m.search_latency_mean_ms)

    # Compute deltas vs baseline (first model)
    baseline = models[0]
    deltas: dict[str, dict[str, float]] = {}

    for model_name in models:
        if model_name == baseline:
            deltas[model_name] = {}
            continue

        deltas[model_name] = {
            "precision_at_5_delta": (
                model_metrics[model_name].precision_at_5
                - model_metrics[baseline].precision_at_5
            ),
            "mrr_delta": (
                model_metrics[model_name].mrr - model_metrics[baseline].mrr
            ),
            "recall_at_10_delta": (
                model_metrics[model_name].recall_at_10
                - model_metrics[baseline].recall_at_10
            ),
        }

    return ModelComparisonReport(
        models=models,
        model_metrics=model_metrics,
        best_precision_model=best_precision.model_name,
        best_mrr_model=best_mrr.model_name,
        best_recall_model=best_recall.model_name,
        fastest_model=fastest.model_name,
        model_deltas=deltas,
        query_best_model=query_best,
        corpus_size=len(corpus),
        query_count=len(queries),
        baseline_model=baseline,
        generated_at=datetime.now(UTC).isoformat(),
    )


def generate_comparison_report_markdown(report: ModelComparisonReport) -> str:
    """Generate a markdown report from model comparison results."""
    lines = [
        "# Embedding Model Comparison Report",
        "",
        f"**Generated:** {report.generated_at}",
        f"**Corpus Size:** {report.corpus_size} documents",
        f"**Queries:** {report.query_count}",
        f"**Baseline Model:** {report.baseline_model}",
        "",
        "## Model Comparison",
        "",
        "| Model | Dimension | P@5 | P@10 | MRR | R@10 | Coverage | Latency |",
        "|-------|-----------|-----|------|-----|------|----------|---------|",
    ]

    for model_name in report.models:
        m = report.model_metrics[model_name]
        lines.append(
            f"| {model_name} | {m.dimension} | "
            f"{m.precision_at_5:.3f} | {m.precision_at_10:.3f} | "
            f"{m.mrr:.3f} | {m.recall_at_10:.3f} | "
            f"{m.coverage:.1%} | {m.search_latency_mean_ms:.1f}ms |"
        )

    lines.extend([
        "",
        "## Best Models",
        "",
        f"- **Best Precision@5:** {report.best_precision_model}",
        f"- **Best MRR:** {report.best_mrr_model}",
        f"- **Best Recall@10:** {report.best_recall_model}",
        f"- **Fastest:** {report.fastest_model}",
    ])

    # Add deltas vs baseline
    if report.model_deltas:
        lines.extend([
            "",
            f"## Delta vs Baseline ({report.baseline_model})",
            "",
            "| Model | P@5 Delta | MRR Delta | R@10 Delta |",
            "|-------|-----------|-----------|------------|",
        ])

        for model_name, deltas in report.model_deltas.items():
            if deltas:
                lines.append(
                    f"| {model_name} | "
                    f"{deltas.get('precision_at_5_delta', 0):+.3f} | "
                    f"{deltas.get('mrr_delta', 0):+.3f} | "
                    f"{deltas.get('recall_at_10_delta', 0):+.3f} |"
                )

    return "\n".join(lines)
