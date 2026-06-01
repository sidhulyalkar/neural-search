"""Ablation runner comparing 8 retrieval variants on the usefulness benchmark.

Variants are scoring functions over DatasetContext objects — no external
BM25/dense infrastructure required for default execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from neural_search.evaluation.usefulness_benchmark import (
    BenchmarkReport,
    PairLabel,
    UsefulnessQuery,
    run_usefulness_benchmark,
)
from neural_search.retrieval.query_intent import UsefulnessIntent, classify_query_intent
from neural_search.retrieval.usefulness_scorer import (
    DatasetContext,
    score_usefulness,
    _jaccard,
)

VARIANT_NAMES = (
    "bm25_only",
    "dense_only",
    "graph_only",
    "affordance_only",
    "bm25_dense_rrf",
    "hybrid_static",
    "hybrid_intent_aware",
    "latent_usefulness_v08",
)


# Type alias for a scoring function that takes (query_context, candidate) -> float
ScoringFn = Callable[[DatasetContext, DatasetContext], float]


@dataclass
class AblationVariant:
    name: str
    score_fn: ScoringFn


@dataclass
class CandidatePool:
    candidates: dict[str, DatasetContext]


@dataclass
class AblationConfig:
    queries: list[UsefulnessQuery]
    labels: list[PairLabel]
    pool: CandidatePool
    k: int = 10
    out_path: Path | None = None


@dataclass
class AblationReport:
    variant_metrics: dict[str, dict[str, float]]
    k: int

    def to_markdown(self) -> str:
        metrics_keys = ["ndcg_at_k", "mrr", "precision_at_k", "hard_negative_violation_rate"]
        header_cols = ["NDCG@K", "MRR", "P@K", "HN-VIOL"]
        header = "| Variant | " + " | ".join(header_cols) + " |"
        sep = "|---------|" + "--------|" * len(metrics_keys)
        rows = [header, sep]
        for variant in VARIANT_NAMES:
            m = self.variant_metrics.get(variant, {})
            vals = " | ".join(f"{m.get(k, 0.0):.4f}" for k in metrics_keys)
            rows.append(f"| {variant} | {vals} |")
        return "# Ablation Report\n\n## Metric Table\n\n" + "\n".join(rows) + "\n"


# --- Scoring functions (proxy implementations, no external retrieval needed) ---

def _score_bm25_only(qctx: DatasetContext, cand: DatasetContext) -> float:
    return 0.6 * _jaccard(qctx.tasks, cand.tasks) + 0.4 * _jaccard(qctx.modalities, cand.modalities)


def _score_dense_only(qctx: DatasetContext, cand: DatasetContext) -> float:
    return 0.5 * _jaccard(qctx.affordances, cand.affordances) + 0.5 * _jaccard(qctx.brain_regions, cand.brain_regions)


def _score_graph_only(qctx: DatasetContext, cand: DatasetContext) -> float:
    return _jaccard(qctx.data_standards, cand.data_standards)


def _score_affordance_only(qctx: DatasetContext, cand: DatasetContext) -> float:
    return _jaccard(qctx.affordances, cand.affordances)


def _score_bm25_dense_rrf(qctx: DatasetContext, cand: DatasetContext) -> float:
    return 0.5 * _score_bm25_only(qctx, cand) + 0.5 * _score_dense_only(qctx, cand)


def _score_hybrid_static(qctx: DatasetContext, cand: DatasetContext) -> float:
    return (
        _score_bm25_only(qctx, cand) * 0.25
        + _score_dense_only(qctx, cand) * 0.25
        + _score_affordance_only(qctx, cand) * 0.25
        + min(1.0, max(0.0, cand.quality_score)) * 0.25
    )


def _score_hybrid_intent_aware(
    qctx: DatasetContext, cand: DatasetContext, intent: UsefulnessIntent
) -> float:
    if intent == UsefulnessIntent.PIPELINE_REUSE:
        return 0.5 * _score_affordance_only(qctx, cand) + 0.5 * _jaccard(qctx.data_standards, cand.data_standards)
    elif intent == UsefulnessIntent.REPLICATION:
        return (
            0.4 * _jaccard(qctx.tasks, cand.tasks)
            + 0.3 * _jaccard(qctx.species, cand.species)
            + 0.3 * _jaccard(qctx.brain_regions, cand.brain_regions)
        )
    elif intent == UsefulnessIntent.METHOD_TRANSFER:
        return 0.7 * _score_affordance_only(qctx, cand) + 0.3 * _score_bm25_only(qctx, cand)
    else:
        return _score_hybrid_static(qctx, cand)


def _build_query_context(query: UsefulnessQuery, pool: CandidatePool) -> DatasetContext:
    """Synthesize a DatasetContext for the query from its top candidate metadata."""
    candidates = [pool.candidates[cid] for cid in query.candidate_ids if cid in pool.candidates]
    if not candidates:
        return DatasetContext(dataset_id="__query__")
    modalities: set[str] = set()
    tasks: set[str] = set()
    affordances: set[str] = set()
    data_standards: set[str] = set()
    for c in candidates[:3]:
        modalities.update(c.modalities)
        tasks.update(c.tasks)
        affordances.update(c.affordances)
        data_standards.update(c.data_standards)
    return DatasetContext(
        dataset_id="__query__",
        modalities=list(modalities),
        tasks=list(tasks),
        affordances=list(affordances),
        data_standards=list(data_standards),
    )


def _rank_candidates(
    query: UsefulnessQuery,
    pool: CandidatePool,
    score_fn: ScoringFn,
) -> list[str]:
    qctx = _build_query_context(query, pool)
    scored = [
        (cid, score_fn(qctx, pool.candidates[cid]))
        for cid in query.candidate_ids
        if cid in pool.candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored]


def run_ablation(config: AblationConfig) -> AblationReport:
    """Run all 8 variants and return an AblationReport."""
    variant_metrics: dict[str, dict[str, float]] = {}

    for variant in VARIANT_NAMES:
        run: dict[str, list[str]] = {}

        for q in config.queries:
            intent = classify_query_intent(q.query).intent

            if variant == "bm25_only":
                ranked = _rank_candidates(q, config.pool, _score_bm25_only)
            elif variant == "dense_only":
                ranked = _rank_candidates(q, config.pool, _score_dense_only)
            elif variant == "graph_only":
                ranked = _rank_candidates(q, config.pool, _score_graph_only)
            elif variant == "affordance_only":
                ranked = _rank_candidates(q, config.pool, _score_affordance_only)
            elif variant == "bm25_dense_rrf":
                ranked = _rank_candidates(q, config.pool, _score_bm25_dense_rrf)
            elif variant == "hybrid_static":
                ranked = _rank_candidates(q, config.pool, _score_hybrid_static)
            elif variant == "hybrid_intent_aware":
                fn: ScoringFn = lambda qc, c, i=intent: _score_hybrid_intent_aware(qc, c, i)
                ranked = _rank_candidates(q, config.pool, fn)
            elif variant == "latent_usefulness_v08":
                qctx = _build_query_context(q, config.pool)
                scored = [
                    (cid, score_usefulness(qctx, config.pool.candidates[cid], intent).total_score)
                    for cid in q.candidate_ids
                    if cid in config.pool.candidates
                ]
                scored.sort(key=lambda x: x[1], reverse=True)
                ranked = [cid for cid, _ in scored]
            else:
                ranked = list(q.candidate_ids)

            run[q.query_id] = ranked

        bench = run_usefulness_benchmark(config.queries, config.labels, run, k=config.k)
        variant_metrics[variant] = {
            "ndcg_at_k": bench.ndcg_at_k,
            "mrr": bench.mrr,
            "precision_at_k": bench.precision_at_k,
            "hard_negative_violation_rate": bench.hard_negative_violation_rate,
        }

    report = AblationReport(variant_metrics=variant_metrics, k=config.k)

    if config.out_path:
        config.out_path.parent.mkdir(parents=True, exist_ok=True)
        config.out_path.write_text(report.to_markdown(), encoding="utf-8")

    return report
