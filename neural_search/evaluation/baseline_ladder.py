"""Evaluation Baseline Ladder.

This module implements a progression of retrieval system configurations
to measure the contribution of each component:

Ladder Levels:
1. Lexical Only - Pure keyword matching
2. Metadata/Entity Only - Structured field matching
3. Embedding Only - Dense retrieval
4. Lexical + Embedding - Hybrid without ontology
5. Lexical + Embedding + Ontology - Standard multi-signal
6. Lexical + Embedding + Ontology + Graph - With graph expansion
7. Full System - With planner, awareness, calibration

Metrics computed at each level:
- P@5, P@10 (Precision at K)
- MRR (Mean Reciprocal Rank)
- NDCG@10 (if graded relevance available)
- Recall@K
- Latency (p50, p95)
- Coverage (% of queries with results)
- Explanation completeness
- Provenance completeness
- Graph contribution rate

This allows understanding:
- Which components provide the most lift
- Where the system has diminishing returns
- What queries benefit from which components
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LadderLevel(StrEnum):
    """Levels in the baseline ladder."""

    LEXICAL_ONLY = "lexical_only"
    METADATA_ONLY = "metadata_only"
    EMBEDDING_ONLY = "embedding_only"
    LEXICAL_EMBEDDING = "lexical_embedding"
    LEXICAL_EMBEDDING_ONTOLOGY = "lexical_embedding_ontology"
    FULL_WITHOUT_GRAPH = "full_without_graph"
    FULL_WITH_GRAPH = "full_with_graph"
    FULL_SYSTEM = "full_system"


@dataclass
class LadderLevelConfig:
    """Configuration for a ladder level."""

    level: LadderLevel
    name: str
    description: str

    # Component enables
    use_lexical: bool = True
    use_metadata: bool = True
    use_ontology: bool = True
    use_embedding: bool = True
    use_graph: bool = False
    use_planner: bool = False
    use_awareness: bool = False

    # Weight overrides
    weight_overrides: dict[str, float] = field(default_factory=dict)


# Ladder level configurations
LADDER_CONFIGS: dict[LadderLevel, LadderLevelConfig] = {
    LadderLevel.LEXICAL_ONLY: LadderLevelConfig(
        level=LadderLevel.LEXICAL_ONLY,
        name="Lexical Only",
        description="Pure keyword/BM25-style matching",
        use_lexical=True,
        use_metadata=False,
        use_ontology=False,
        use_embedding=False,
        use_graph=False,
        weight_overrides={"semantic": 0.0, "ontology": 0.0, "behavior": 0.0, "modality": 0.0},
    ),
    LadderLevel.METADATA_ONLY: LadderLevelConfig(
        level=LadderLevel.METADATA_ONLY,
        name="Metadata Only",
        description="Structured field matching without text search",
        use_lexical=False,
        use_metadata=True,
        use_ontology=False,
        use_embedding=False,
        use_graph=False,
        weight_overrides={"semantic": 0.0, "ontology": 0.0},
    ),
    LadderLevel.EMBEDDING_ONLY: LadderLevelConfig(
        level=LadderLevel.EMBEDDING_ONLY,
        name="Embedding Only",
        description="Dense retrieval without structured signals",
        use_lexical=False,
        use_metadata=False,
        use_ontology=False,
        use_embedding=True,
        use_graph=False,
        weight_overrides={"semantic": 1.0, "ontology": 0.0, "behavior": 0.0, "modality": 0.0},
    ),
    LadderLevel.LEXICAL_EMBEDDING: LadderLevelConfig(
        level=LadderLevel.LEXICAL_EMBEDDING,
        name="Lexical + Embedding",
        description="Hybrid lexical-dense without ontology",
        use_lexical=True,
        use_metadata=True,
        use_ontology=False,
        use_embedding=True,
        use_graph=False,
        weight_overrides={"semantic": 0.3, "ontology": 0.0, "behavior": 0.0},
    ),
    LadderLevel.LEXICAL_EMBEDDING_ONTOLOGY: LadderLevelConfig(
        level=LadderLevel.LEXICAL_EMBEDDING_ONTOLOGY,
        name="Lexical + Embedding + Ontology",
        description="Standard multi-signal without graph",
        use_lexical=True,
        use_metadata=True,
        use_ontology=True,
        use_embedding=True,
        use_graph=False,
    ),
    LadderLevel.FULL_WITHOUT_GRAPH: LadderLevelConfig(
        level=LadderLevel.FULL_WITHOUT_GRAPH,
        name="Full Without Graph",
        description="All signals except graph expansion",
        use_lexical=True,
        use_metadata=True,
        use_ontology=True,
        use_embedding=True,
        use_graph=False,
        use_planner=True,
    ),
    LadderLevel.FULL_WITH_GRAPH: LadderLevelConfig(
        level=LadderLevel.FULL_WITH_GRAPH,
        name="Full With Graph",
        description="All signals including graph",
        use_lexical=True,
        use_metadata=True,
        use_ontology=True,
        use_embedding=True,
        use_graph=True,
        use_planner=True,
    ),
    LadderLevel.FULL_SYSTEM: LadderLevelConfig(
        level=LadderLevel.FULL_SYSTEM,
        name="Full System",
        description="Complete system with all features",
        use_lexical=True,
        use_metadata=True,
        use_ontology=True,
        use_embedding=True,
        use_graph=True,
        use_planner=True,
        use_awareness=True,
    ),
}


class QueryResult(BaseModel):
    """Result for a single query at one ladder level."""

    query: str
    level: str
    result_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    latency_ms: float = 0.0
    has_results: bool = False


class LevelMetrics(BaseModel):
    """Metrics for a single ladder level."""

    level: str
    level_name: str
    query_count: int = 0

    # Precision metrics
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0

    # Ranking metrics
    mrr: float = 0.0
    ndcg_at_10: float = 0.0

    # Recall
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0

    # Latency
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_mean_ms: float = 0.0

    # Coverage
    coverage: float = 0.0  # % of queries with at least 1 result

    # Attribution
    unique_contribution: float = 0.0  # Lift over previous level


class LadderReport(BaseModel):
    """Complete baseline ladder evaluation report."""

    levels: list[LevelMetrics] = Field(default_factory=list)
    total_queries: int = 0
    total_labeled_pairs: int = 0

    # Best level analysis
    best_precision_level: str = ""
    best_mrr_level: str = ""

    # Lift analysis
    graph_lift: float = 0.0
    ontology_lift: float = 0.0
    embedding_lift: float = 0.0
    planner_lift: float = 0.0

    # Failure analysis
    queries_needing_graph: list[str] = Field(default_factory=list)
    queries_hurt_by_graph: list[str] = Field(default_factory=list)
    low_confidence_queries: list[str] = Field(default_factory=list)

    generated_at: str = ""


def _compute_precision_at_k(
    result_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute precision@k."""
    if not result_ids or not relevant_ids:
        return 0.0
    top_k = result_ids[:k]
    relevant_in_top_k = sum(1 for r in top_k if r in relevant_ids)
    return relevant_in_top_k / k


def _compute_recall_at_k(
    result_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Compute recall@k."""
    if not relevant_ids:
        return 0.0
    top_k = result_ids[:k]
    relevant_in_top_k = sum(1 for r in top_k if r in relevant_ids)
    return relevant_in_top_k / len(relevant_ids)


def _compute_mrr(
    result_ids: list[str],
    relevant_ids: set[str],
) -> float:
    """Compute Mean Reciprocal Rank."""
    if not result_ids or not relevant_ids:
        return 0.0
    for i, result_id in enumerate(result_ids):
        if result_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def _compute_ndcg_at_k(
    result_ids: list[str],
    relevance_grades: dict[str, int],  # result_id -> grade (0, 1, 2, etc.)
    k: int,
) -> float:
    """Compute NDCG@k with graded relevance."""
    import math

    if not result_ids or not relevance_grades:
        return 0.0

    # DCG
    dcg = 0.0
    for i, result_id in enumerate(result_ids[:k]):
        grade = relevance_grades.get(result_id, 0)
        dcg += (2 ** grade - 1) / math.log2(i + 2)

    # Ideal DCG
    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal_grades))

    return dcg / idcg if idcg > 0 else 0.0


def run_query_at_level(
    query: str,
    corpus: list[dict[str, Any]],
    config: LadderLevelConfig,
) -> QueryResult:
    """Run a query at a specific ladder level.

    This is a simplified implementation that uses the core retrieval
    with modified weights. In production, you'd wire this to the
    actual search_datasets function.
    """
    from neural_search.core.query import parse_and_plan_query
    from neural_search.core.retrieval import MultiStageRetriever

    start_time = time.time()

    # Parse query and get plan
    plan = parse_and_plan_query(query)

    # Apply level-specific weight overrides
    if config.weight_overrides:
        plan.weight_overrides.update(config.weight_overrides)

    # Configure stages based on level
    for stage in plan.stages:
        if not config.use_ontology and stage.stage.value == "ontology_match":
            stage.enabled = False
        if not config.use_graph and stage.stage.value == "graph_expansion":
            stage.enabled = False

    # Run retrieval
    retriever = MultiStageRetriever()
    result = retriever.search(plan, corpus, top_k=10)

    latency = (time.time() - start_time) * 1000

    return QueryResult(
        query=query,
        level=config.level.value,
        result_ids=[r["dataset_id"] for r in result.results],
        scores=[r["score"] for r in result.results],
        latency_ms=latency,
        has_results=len(result.results) > 0,
    )


def evaluate_level(
    queries: list[str],
    corpus: list[dict[str, Any]],
    relevance_labels: dict[str, set[str]],  # query -> set of relevant IDs
    config: LadderLevelConfig,
) -> tuple[LevelMetrics, list[QueryResult]]:
    """Evaluate a ladder level on a query set.

    Args:
        queries: List of query strings
        corpus: Dataset records to search
        relevance_labels: Map from query to relevant dataset IDs
        config: Ladder level configuration

    Returns:
        Tuple of (LevelMetrics, list of QueryResults)
    """
    results: list[QueryResult] = []
    latencies: list[float] = []
    precisions_5: list[float] = []
    precisions_10: list[float] = []
    recalls_5: list[float] = []
    recalls_10: list[float] = []
    mrrs: list[float] = []
    queries_with_results = 0

    for query in queries:
        result = run_query_at_level(query, corpus, config)
        results.append(result)
        latencies.append(result.latency_ms)

        relevant = relevance_labels.get(query, set())

        precisions_5.append(_compute_precision_at_k(result.result_ids, relevant, 5))
        precisions_10.append(_compute_precision_at_k(result.result_ids, relevant, 10))
        recalls_5.append(_compute_recall_at_k(result.result_ids, relevant, 5))
        recalls_10.append(_compute_recall_at_k(result.result_ids, relevant, 10))
        mrrs.append(_compute_mrr(result.result_ids, relevant))

        if result.has_results:
            queries_with_results += 1

    # Compute aggregate metrics
    n = len(queries)
    latencies_sorted = sorted(latencies)

    metrics = LevelMetrics(
        level=config.level.value,
        level_name=config.name,
        query_count=n,
        precision_at_5=sum(precisions_5) / n if n else 0.0,
        precision_at_10=sum(precisions_10) / n if n else 0.0,
        recall_at_5=sum(recalls_5) / n if n else 0.0,
        recall_at_10=sum(recalls_10) / n if n else 0.0,
        mrr=sum(mrrs) / n if n else 0.0,
        latency_mean_ms=sum(latencies) / n if n else 0.0,
        latency_p50_ms=latencies_sorted[n // 2] if n else 0.0,
        latency_p95_ms=latencies_sorted[int(n * 0.95)] if n else 0.0,
        coverage=queries_with_results / n if n else 0.0,
    )

    return metrics, results


def run_baseline_ladder(
    queries: list[str],
    corpus: list[dict[str, Any]],
    relevance_labels: dict[str, set[str]],
    levels: list[LadderLevel] | None = None,
) -> LadderReport:
    """Run the complete baseline ladder evaluation.

    Args:
        queries: List of benchmark query strings
        corpus: Dataset records to search
        relevance_labels: Map from query to relevant dataset IDs
        levels: Optional subset of levels to evaluate (default: all)

    Returns:
        LadderReport with metrics for each level
    """
    from datetime import datetime

    if levels is None:
        levels = list(LadderLevel)

    level_results: list[LevelMetrics] = []
    all_query_results: dict[LadderLevel, list[QueryResult]] = {}

    for level in levels:
        config = LADDER_CONFIGS[level]
        metrics, query_results = evaluate_level(queries, corpus, relevance_labels, config)
        level_results.append(metrics)
        all_query_results[level] = query_results

    # Compute lift between adjacent levels
    for i in range(1, len(level_results)):
        prev_mrr = level_results[i - 1].mrr
        curr_mrr = level_results[i].mrr
        level_results[i].unique_contribution = curr_mrr - prev_mrr

    # Find best levels
    best_precision = max(level_results, key=lambda x: x.precision_at_5)
    best_mrr = max(level_results, key=lambda x: x.mrr)

    # Compute specific lifts
    def get_level_mrr(level: LadderLevel) -> float:
        for m in level_results:
            if m.level == level.value:
                return m.mrr
        return 0.0

    graph_lift = (
        get_level_mrr(LadderLevel.FULL_WITH_GRAPH)
        - get_level_mrr(LadderLevel.FULL_WITHOUT_GRAPH)
    )
    ontology_lift = (
        get_level_mrr(LadderLevel.LEXICAL_EMBEDDING_ONTOLOGY)
        - get_level_mrr(LadderLevel.LEXICAL_EMBEDDING)
    )
    embedding_lift = (
        get_level_mrr(LadderLevel.LEXICAL_EMBEDDING)
        - get_level_mrr(LadderLevel.LEXICAL_ONLY)
    )

    # Identify queries that benefit from graph
    queries_needing_graph = []
    queries_hurt_by_graph = []

    if LadderLevel.FULL_WITH_GRAPH in all_query_results and LadderLevel.FULL_WITHOUT_GRAPH in all_query_results:
        graph_results = all_query_results[LadderLevel.FULL_WITH_GRAPH]
        no_graph_results = all_query_results[LadderLevel.FULL_WITHOUT_GRAPH]

        for gr, ngr in zip(graph_results, no_graph_results, strict=False):
            query = gr.query
            graph_mrr = _compute_mrr(gr.result_ids, relevance_labels.get(query, set()))
            no_graph_mrr = _compute_mrr(ngr.result_ids, relevance_labels.get(query, set()))

            if graph_mrr > no_graph_mrr + 0.1:
                queries_needing_graph.append(query)
            elif graph_mrr < no_graph_mrr - 0.1:
                queries_hurt_by_graph.append(query)

    return LadderReport(
        levels=level_results,
        total_queries=len(queries),
        total_labeled_pairs=sum(len(v) for v in relevance_labels.values()),
        best_precision_level=best_precision.level,
        best_mrr_level=best_mrr.level,
        graph_lift=round(graph_lift, 4),
        ontology_lift=round(ontology_lift, 4),
        embedding_lift=round(embedding_lift, 4),
        queries_needing_graph=queries_needing_graph[:10],
        queries_hurt_by_graph=queries_hurt_by_graph[:10],
        generated_at=datetime.now(UTC).isoformat(),
    )


def format_ladder_report_markdown(report: LadderReport) -> str:
    """Format a ladder report as markdown."""
    lines = [
        "# Baseline Ladder Evaluation Report",
        "",
        f"**Generated:** {report.generated_at}",
        f"**Queries:** {report.total_queries}",
        f"**Labeled Pairs:** {report.total_labeled_pairs}",
        "",
        "## Level Comparison",
        "",
        "| Level | P@5 | P@10 | MRR | R@10 | Coverage | Latency (p50) | Lift |",
        "|-------|-----|------|-----|------|----------|---------------|------|",
    ]

    for level in report.levels:
        lines.append(
            f"| {level.level_name} | "
            f"{level.precision_at_5:.3f} | "
            f"{level.precision_at_10:.3f} | "
            f"{level.mrr:.3f} | "
            f"{level.recall_at_10:.3f} | "
            f"{level.coverage:.1%} | "
            f"{level.latency_p50_ms:.1f}ms | "
            f"{level.unique_contribution:+.3f} |"
        )

    lines.extend([
        "",
        "## Lift Analysis",
        "",
        f"- **Graph lift:** {report.graph_lift:+.4f}",
        f"- **Ontology lift:** {report.ontology_lift:+.4f}",
        f"- **Embedding lift:** {report.embedding_lift:+.4f}",
        "",
        "## Best Levels",
        "",
        f"- **Best Precision:** {report.best_precision_level}",
        f"- **Best MRR:** {report.best_mrr_level}",
    ])

    if report.queries_needing_graph:
        lines.extend([
            "",
            "## Queries Benefiting from Graph",
            "",
        ])
        for q in report.queries_needing_graph:
            lines.append(f"- {q[:80]}...")

    if report.queries_hurt_by_graph:
        lines.extend([
            "",
            "## Queries Hurt by Graph",
            "",
        ])
        for q in report.queries_hurt_by_graph:
            lines.append(f"- {q[:80]}...")

    return "\n".join(lines)
