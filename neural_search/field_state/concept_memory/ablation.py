"""Qrels-backed Concept Memory retrieval ablation harness."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from neural_search.field_state.concept_memory.artifact_utils import (
    artifact_timestamp,
    deterministic_enabled,
)
from neural_search.field_state.concept_memory.graph_builder import (
    build_concept_graph,
    read_concept_artifacts,
)
from neural_search.field_state.concept_memory.reranker import rerank_datasets
from neural_search.field_state.concept_memory.schema import (
    ConceptNode,
    ConceptRerankedResult,
    EvidenceLink,
)

DEFAULT_JSON_OUT = Path("reports/eval/concept_memory_ablation.json")

UNSUPPORTED_CLAIM_REMINDERS = [
    "This report does not establish general retrieval improvement.",
    "Metric changes are limited by qrels coverage, corpus snapshot, and annotation quality.",
    "Metadata-derived links are not reviewed scientific evidence.",
]


@dataclass(frozen=True)
class BenchmarkQuery:
    query_id: str
    query_text: str
    intent: str | None = None


@dataclass(frozen=True)
class Qrel:
    query_id: str
    record_id: str
    relevance: int
    hard_negative: bool = False


@dataclass(frozen=True)
class RunRow:
    query_id: str
    record_id: str
    rank: int
    score: float


def _sha256_path(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_queries(path: Path) -> tuple[list[BenchmarkQuery], list[str]]:
    """Load benchmark queries, accepting v1 and legacy field names."""
    warnings: list[str] = []
    queries: list[BenchmarkQuery] = []
    if not path.exists():
        return queries, [f"Queries file not found: {path}"]
    for idx, row in enumerate(_read_jsonl(path), start=1):
        query_id = str(row.get("query_id") or row.get("id") or f"query_{idx}")
        query_text = str(row.get("query_text") or row.get("query") or "").strip()
        if not query_text:
            warnings.append(f"Skipping query {query_id}: missing query text")
            continue
        intent = row.get("intent")
        queries.append(
            BenchmarkQuery(
                query_id=query_id,
                query_text=query_text,
                intent=str(intent) if intent else None,
            )
        )
    return queries, warnings


def load_qrels(path: Path) -> tuple[list[Qrel], list[str]]:
    """Load qrels from common Neural Search qrels shapes."""
    warnings: list[str] = []
    if not path.exists():
        return [], [f"Qrels file not found: {path}"]
    qrels: list[Qrel] = []
    try:
        rows = _read_jsonl(path)
    except json.JSONDecodeError as exc:
        return [], [f"Qrels file is malformed JSONL: {exc}"]
    for idx, row in enumerate(rows, start=1):
        query_id = row.get("query_id")
        record_id = row.get("record_id") or row.get("dataset_id")
        raw_relevance = (
            row.get("relevance")
            if "relevance" in row
            else row.get("label", row.get("final_relevance_score"))
        )
        if query_id is None or record_id is None or raw_relevance is None:
            warnings.append(f"Skipping malformed qrel row {idx}")
            continue
        try:
            relevance = int(raw_relevance)
        except (TypeError, ValueError):
            warnings.append(f"Skipping qrel row {idx}: non-integer relevance")
            continue
        qrels.append(
            Qrel(
                query_id=str(query_id),
                record_id=str(record_id),
                relevance=relevance,
                hard_negative=bool(
                    row.get("hard_negative_violation")
                    or row.get("final_hard_negative_violation")
                    or relevance <= 0
                ),
            )
        )
    return qrels, warnings


def _qrels_by_query(qrels: list[Qrel]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for qrel in qrels:
        grouped[qrel.query_id][qrel.record_id] = qrel.relevance
    return dict(grouped)


def _dcg(labels: list[int]) -> float:
    return float(sum((2**label - 1) / math.log2(idx + 2) for idx, label in enumerate(labels)))


def _ndcg_at_k(qrel: dict[str, int], ranked: list[str], k: int) -> float:
    gains = [qrel.get(record_id, 0) for record_id in ranked[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(gains) / ideal_dcg if ideal_dcg else 0.0


def _mrr_at_k(qrel: dict[str, int], ranked: list[str], k: int, threshold: int = 2) -> float:
    for idx, record_id in enumerate(ranked[:k], start=1):
        if qrel.get(record_id, 0) >= threshold:
            return 1.0 / idx
    return 0.0


def _precision_at_k(qrel: dict[str, int], ranked: list[str], k: int, threshold: int = 2) -> float:
    top_k = ranked[:k]
    if not top_k:
        return 0.0
    return sum(1 for record_id in top_k if qrel.get(record_id, 0) >= threshold) / len(top_k)


def _recall_at_k(qrel: dict[str, int], ranked: list[str], k: int, threshold: int = 2) -> float:
    relevant = {record_id for record_id, label in qrel.items() if label >= threshold}
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def _hard_negative_violation_rate(qrel: dict[str, int], ranked: list[str], k: int = 10) -> float:
    hard_negatives = {record_id for record_id, label in qrel.items() if label <= 0}
    if not hard_negatives:
        return 0.0
    violations = sum(1 for record_id in ranked[:k] if record_id in hard_negatives)
    return violations / len(hard_negatives)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _bootstrap_delta_ci(
    baseline: list[float],
    variant: list[float],
    *,
    seed: int = 42,
    samples: int = 1000,
) -> list[float]:
    if len(baseline) < 2 or len(baseline) != len(variant):
        delta = _mean(variant) - _mean(baseline)
        rounded = round(delta, 6)
        return [rounded, rounded]
    rng = random.Random(seed)
    n = len(baseline)
    deltas: list[float] = []
    for _ in range(samples):
        idxs = [rng.randrange(n) for _ in range(n)]
        deltas.append(_mean([variant[i] - baseline[i] for i in idxs]))
    deltas.sort()
    return [round(deltas[int(samples * 0.025)], 6), round(deltas[int(samples * 0.975)], 6)]


def _score_to_probability(score: float) -> float:
    return max(0.0, min(1.0, score))


def _compute_ece(rows: list[RunRow], qrels_by_query: dict[str, dict[str, int]]) -> dict[str, Any]:
    pairs: list[tuple[float, int]] = []
    for row in rows:
        label = qrels_by_query.get(row.query_id, {}).get(row.record_id)
        if label is not None:
            pairs.append((_score_to_probability(row.score), label))
    if not pairs:
        return {"status": "no_overlap", "ece": None, "reliability_bins": []}
    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for score, label in pairs:
        bins[min(int(score * 10), 9)].append((score, label))
    ece = 0.0
    reliability_bins: list[dict[str, Any]] = []
    total = len(pairs)
    for idx in range(10):
        items = bins.get(idx, [])
        if not items:
            reliability_bins.append({"bin": idx, "count": 0})
            continue
        mean_conf = _mean([score for score, _ in items])
        accuracy = sum(1 for _, label in items if label >= 2) / len(items)
        ece += (len(items) / total) * abs(mean_conf - accuracy)
        reliability_bins.append({
            "bin": idx,
            "count": len(items),
            "mean_confidence": round(mean_conf, 6),
            "accuracy": round(accuracy, 6),
        })
    return {"ece": round(ece, 6), "reliability_bins": reliability_bins}


def _rows_from_results(query_id: str, results: list[ConceptRerankedResult]) -> list[RunRow]:
    return [
        RunRow(query_id=query_id, record_id=result.dataset_id, rank=rank, score=result.final_score)
        for rank, result in enumerate(results, start=1)
    ]


def _rank_variant(
    *,
    variant: str,
    query: BenchmarkQuery,
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
) -> list[RunRow]:
    graph = build_concept_graph(concepts, evidence_links)
    if variant in {"baseline_no_concept", "concept_memory_lexical_only"}:
        results = rerank_datasets(
            query=query.query_text,
            concepts=concepts,
            evidence_links=evidence_links,
            graph=graph,
            limit=len(concepts),
            enable_concept_boost=False,
            enable_evidence_boost=False,
            enable_hard_negative_penalty=False,
        )
    elif variant == "concept_memory_graph_degree_normalized":
        results = rerank_datasets(
            query=query.query_text,
            concepts=concepts,
            evidence_links=evidence_links,
            graph=graph,
            limit=len(concepts),
            enable_concept_boost=True,
            enable_evidence_boost=False,
            enable_hard_negative_penalty=False,
        )
    else:
        results = rerank_datasets(
            query=query.query_text,
            concepts=concepts,
            evidence_links=evidence_links,
            graph=graph,
            limit=len(concepts),
            enable_concept_boost=True,
            enable_evidence_boost=True,
            enable_hard_negative_penalty=True,
        )
    return _rows_from_results(query.query_id, results)


def _compute_variant_metrics(
    rows_by_query: dict[str, list[RunRow]],
    qrels_by_query: dict[str, dict[str, int]],
    query_intents: dict[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, list[dict[str, float]]]]:
    per_query_metrics: list[dict[str, float]] = []
    per_query_by_id: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_intent_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    skipped = 0
    for query_id, qrel in qrels_by_query.items():
        rows = rows_by_query.get(query_id, [])
        ranked = [row.record_id for row in sorted(rows, key=lambda item: item.rank)]
        if not ranked:
            skipped += 1
            continue
        metrics = {
            "ndcg_at_10": _ndcg_at_k(qrel, ranked, 10),
            "mrr_at_10": _mrr_at_k(qrel, ranked, 10),
            "recall_at_10": _recall_at_k(qrel, ranked, 10),
            "precision_at_5": _precision_at_k(qrel, ranked, 5),
            "hard_negative_violation_rate": _hard_negative_violation_rate(qrel, ranked, 10),
        }
        per_query_metrics.append(metrics)
        per_query_by_id[query_id].append(metrics)
        intent = query_intents.get(query_id)
        if intent:
            per_intent_rows[intent].append(metrics)
    aggregate = {
        key: round(_mean([row[key] for row in per_query_metrics]), 6)
        for key in (
            "ndcg_at_10",
            "mrr_at_10",
            "recall_at_10",
            "precision_at_5",
            "hard_negative_violation_rate",
        )
    }
    aggregate["evaluated_queries"] = len(per_query_metrics)
    aggregate["skipped_queries"] = skipped
    per_intent = {
        intent: {
            key: round(_mean([row[key] for row in rows]), 6)
            for key in (
                "ndcg_at_10",
                "mrr_at_10",
                "recall_at_10",
                "precision_at_5",
                "hard_negative_violation_rate",
            )
        } | {"evaluated_queries": len(rows)}
        for intent, rows in sorted(per_intent_rows.items())
    }
    return aggregate, per_intent, dict(per_query_by_id)


def run_ablation(
    *,
    root: Path | None,
    queries_path: Path,
    qrels_path: Path,
    corpus_path: Path | None,
    out_json: Path = DEFAULT_JSON_OUT,
    deterministic: bool | None = None,
) -> dict[str, Any]:
    """Run qrels-backed Concept Memory ablation and write JSON/Markdown reports."""
    base = root if root is not None else Path.cwd()
    det = deterministic_enabled(deterministic)
    out_json = out_json if out_json.is_absolute() else base / out_json
    out_md = out_json.with_suffix(".md")

    queries, query_warnings = load_queries(base / queries_path if not queries_path.is_absolute() else queries_path)
    qrels, qrel_warnings = load_qrels(base / qrels_path if not qrels_path.is_absolute() else qrels_path)
    concepts, evidence_links = read_concept_artifacts(base)
    warnings = query_warnings + qrel_warnings

    abs_queries = base / queries_path if not queries_path.is_absolute() else queries_path
    abs_qrels = base / qrels_path if not qrels_path.is_absolute() else qrels_path
    abs_corpus = None if corpus_path is None else (base / corpus_path if not corpus_path.is_absolute() else corpus_path)

    metadata = {
        "generated_at": artifact_timestamp(det),
        "deterministic": det,
        "input_paths": {
            "queries": str(queries_path),
            "qrels": str(qrels_path),
            "corpus": str(corpus_path) if corpus_path is not None else None,
        },
        "input_hashes": {
            "queries_sha256": _sha256_path(abs_queries),
            "qrels_sha256": _sha256_path(abs_qrels),
            "corpus_sha256": _sha256_path(abs_corpus),
        },
        "num_queries": len(queries),
        "num_qrels": len(qrels),
    }

    if not qrels:
        warnings.append("No qrels loaded; metrics were not computed.")
        report = {
            "status": "pending_qrels",
            "run_metadata": metadata,
            "retrieval_variants": [],
            "metrics_by_variant": {},
            "metric_deltas_against_baseline": {},
            "per_intent_metrics": {},
            "warnings": warnings,
            "unsupported_claim_reminders": UNSUPPORTED_CLAIM_REMINDERS,
        }
        _write_reports(report, out_json, out_md)
        return report

    qrels_by_query = _qrels_by_query(qrels)
    query_by_id = {query.query_id: query for query in queries}
    query_intents = {
        query.query_id: query.intent
        for query in queries
        if query.intent
    }
    evaluated_queries = [query for query in queries if query.query_id in qrels_by_query]
    if not evaluated_queries:
        warnings.append("No overlap between queries and qrels; metrics were not computed.")
    if len(evaluated_queries) < 5:
        warnings.append("Qrels set is small; confidence intervals are unstable and should not be over-interpreted.")

    variants = [
        "baseline_no_concept",
        "concept_memory_enabled",
        "concept_memory_lexical_only",
        "concept_memory_graph_degree_normalized",
    ]
    rows_by_variant: dict[str, dict[str, list[RunRow]]] = {}
    for variant in variants:
        rows_by_variant[variant] = {}
        for query in evaluated_queries:
            rows_by_variant[variant][query.query_id] = _rank_variant(
                variant=variant,
                query=query,
                concepts=concepts,
                evidence_links=evidence_links,
            )

    metrics_by_variant: dict[str, Any] = {}
    per_intent_metrics: dict[str, Any] = {}
    per_query_by_variant: dict[str, dict[str, list[dict[str, float]]]] = {}
    for variant, rows_by_query in rows_by_variant.items():
        flat_rows = [row for rows in rows_by_query.values() for row in rows]
        aggregate, per_intent, per_query = _compute_variant_metrics(
            rows_by_query,
            qrels_by_query,
            query_intents,
        )
        aggregate.update(_compute_ece(flat_rows, qrels_by_query))
        metrics_by_variant[variant] = aggregate
        per_intent_metrics[variant] = per_intent
        per_query_by_variant[variant] = per_query

    baseline_query_metrics = [
        rows[0] for _, rows in sorted(per_query_by_variant["baseline_no_concept"].items())
        if rows
    ]
    metric_deltas: dict[str, Any] = {}
    for variant in variants:
        if variant == "baseline_no_concept":
            continue
        variant_query_metrics = [
            rows[0] for _, rows in sorted(per_query_by_variant[variant].items())
            if rows
        ]
        deltas: dict[str, Any] = {}
        for key in ("ndcg_at_10", "mrr_at_10", "recall_at_10", "precision_at_5", "hard_negative_violation_rate"):
            baseline_values = [row[key] for row in baseline_query_metrics]
            variant_values = [row[key] for row in variant_query_metrics]
            delta = _mean(variant_values) - _mean(baseline_values)
            deltas[key] = round(delta, 6)
            deltas[f"{key}_delta_ci95"] = _bootstrap_delta_ci(baseline_values, variant_values)
        metric_deltas[variant] = deltas

    skipped_queries = len([query for query in queries if query.query_id not in qrels_by_query])
    metadata["skipped_queries"] = skipped_queries
    report = {
        "status": "computed" if evaluated_queries else "no_query_qrel_overlap",
        "run_metadata": metadata,
        "retrieval_variants": variants,
        "metrics_by_variant": metrics_by_variant,
        "metric_deltas_against_baseline": metric_deltas,
        "per_intent_metrics": per_intent_metrics,
        "warnings": warnings,
        "unsupported_claim_reminders": UNSUPPORTED_CLAIM_REMINDERS,
        "query_ids_evaluated": [query.query_id for query in evaluated_queries],
        "query_ids_without_qrels": [
            query_id for query_id in query_by_id if query_id not in qrels_by_query
        ],
    }
    _write_reports(report, out_json, out_md)
    return report


def _write_reports(report: dict[str, Any], out_json: Path, out_md: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out_md.write_text(render_markdown_report(report), encoding="utf-8")


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a paper-safe Markdown ablation report."""
    lines = [
        "# Concept Memory Retrieval Ablation",
        "",
        "## Claim Safety Notice",
        "",
        "Concept Memory is structural/provenance infrastructure. This report only describes qrels-backed associations on the specified benchmark snapshot.",
        "This does not establish general retrieval improvement without broader validation.",
        "",
        f"Status: `{report.get('status')}`",
        "",
    ]
    warnings = report.get("warnings") or []
    if warnings:
        lines += ["## Warnings", ""]
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")

    metrics = report.get("metrics_by_variant") or {}
    if metrics:
        lines += [
            "## Metrics",
            "",
            "| Variant | NDCG@10 | MRR@10 | Recall@10 | Precision@5 | Hard-negative violation rate | Evaluated queries |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for variant, vals in metrics.items():
            lines.append(
                "| "
                + " | ".join([
                    variant,
                    str(vals.get("ndcg_at_10")),
                    str(vals.get("mrr_at_10")),
                    str(vals.get("recall_at_10")),
                    str(vals.get("precision_at_5")),
                    str(vals.get("hard_negative_violation_rate")),
                    str(vals.get("evaluated_queries")),
                ])
                + " |"
            )
        lines += ["", "## Baseline Deltas", ""]
        deltas = report.get("metric_deltas_against_baseline") or {}
        if deltas:
            lines += [
                "| Variant | Delta NDCG@10 | Delta MRR@10 | Delta Recall@10 | Delta Precision@5 |",
                "| --- | --- | --- | --- | --- |",
            ]
            for variant, vals in deltas.items():
                lines.append(
                    "| "
                    + " | ".join([
                        variant,
                        str(vals.get("ndcg_at_10")),
                        str(vals.get("mrr_at_10")),
                        str(vals.get("recall_at_10")),
                        str(vals.get("precision_at_5")),
                    ])
                    + " |"
                )
            lines.append("")
            lines.append("Concept Memory was associated with the metric changes above on this qrels set.")
            lines.append("This result is limited by qrels coverage, corpus snapshot, and annotation quality.")
        else:
            lines.append("_No deltas computed._")
    else:
        lines += [
            "## Metrics",
            "",
            "_No qrels-backed metrics computed._",
        ]

    lines += ["", "## Unsupported Claim Reminders", ""]
    lines.extend(f"- {item}" for item in UNSUPPORTED_CLAIM_REMINDERS)
    return "\n".join(lines) + "\n"
