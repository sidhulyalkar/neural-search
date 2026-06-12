"""Ablation evaluation for Concept Memory retrieval integration.

Compares retrieval variants using IR metrics (NDCG@10, MRR, Recall@10, Recall@50).
Degrades gracefully when qrels are absent — never fabricates metrics.

Variants evaluated:
  lexical_only         — base lexical score, no concept graph
  concept_boost        — lexical + concept graph boost
  concept_boost_ev     — lexical + concept boost + evidence boost
  full                 — lexical + concept boost + evidence boost + hard-negative penalty
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx  # type: ignore[import-untyped]

from neural_search.field_state.concept_memory.graph_builder import (
    build_concept_graph,
    read_concept_artifacts,
)
from neural_search.field_state.concept_memory.reranker import rerank_datasets
from neural_search.field_state.concept_memory.retrieval import (
    _lexical_score,
)
from neural_search.field_state.concept_memory.schema import (
    ConceptNode,
    EvidenceLink,
)
from neural_search.field_state.store import resolve_path

_ADJUDICATED_QRELS_PATH = Path("artifacts/field_state/adjudicated_qrels.jsonl")
_BENCHMARK_QUERIES_PATH = Path("artifacts/benchmark_queries.jsonl")
_EVAL_REPORT_PATH = Path("reports/eval/concept_memory_eval.md")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class QrelEntry:
    query_id: str
    dataset_id: str
    relevance: int  # 0–3


@dataclass
class VariantMetrics:
    variant: str
    ndcg_at_10: float | None = None
    mrr: float | None = None
    recall_at_10: float | None = None
    recall_at_50: float | None = None
    hard_negative_violation_rate: float | None = None
    source_skew: dict[str, float] = field(default_factory=dict)
    evaluated_queries: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "variant": self.variant,
            "ndcg_at_10": self.ndcg_at_10,
            "mrr": self.mrr,
            "recall_at_10": self.recall_at_10,
            "recall_at_50": self.recall_at_50,
            "hard_negative_violation_rate": self.hard_negative_violation_rate,
            "source_skew": self.source_skew,
            "evaluated_queries": self.evaluated_queries,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# IR metric helpers
# ---------------------------------------------------------------------------


def _dcg(ranked_relevances: list[int], k: int) -> float:
    total = 0.0
    for i, rel in enumerate(ranked_relevances[:k], start=1):
        if rel > 0:
            total += rel / math.log2(i + 1)
    return total


def _idcg(relevances: list[int], k: int) -> float:
    sorted_rels = sorted(relevances, reverse=True)
    return _dcg(sorted_rels, k)


def _ndcg_at_k(
    ranked_ids: list[str],
    qrel_map: dict[str, int],
    k: int = 10,
) -> float:
    ranked_rels = [qrel_map.get(did, 0) for did in ranked_ids]
    ideal_rels = list(qrel_map.values())
    idcg = _idcg(ideal_rels, k)
    if idcg == 0:
        return 0.0
    return _dcg(ranked_rels, k) / idcg


def _mrr(
    ranked_ids: list[str],
    qrel_map: dict[str, int],
    relevance_threshold: int = 1,
) -> float:
    for rank, did in enumerate(ranked_ids, start=1):
        if qrel_map.get(did, 0) >= relevance_threshold:
            return 1.0 / rank
    return 0.0


def _recall_at_k(
    ranked_ids: list[str],
    qrel_map: dict[str, int],
    k: int,
    relevance_threshold: int = 1,
) -> float:
    relevant = {did for did, rel in qrel_map.items() if rel >= relevance_threshold}
    if not relevant:
        return 0.0
    retrieved = set(ranked_ids[:k])
    return len(retrieved & relevant) / len(relevant)


def _hard_negative_violation_rate(
    ranked_ids: list[str],
    hard_negatives: set[str],
    k: int = 10,
) -> float:
    if not hard_negatives:
        return 0.0
    top_k = ranked_ids[:k]
    violations = sum(1 for did in top_k if did in hard_negatives)
    return violations / k


# ---------------------------------------------------------------------------
# JSONL loaders
# ---------------------------------------------------------------------------


def _load_benchmark_queries(queries_path: Path) -> list[dict[str, object]]:
    if not queries_path.exists():
        return []
    queries = []
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            queries.append(json.loads(line))
    return queries


def _load_adjudicated_qrels(qrels_path: Path) -> list[QrelEntry]:
    if not qrels_path.exists():
        return []
    entries = []
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        entries.append(
            QrelEntry(
                query_id=str(obj.get("query_id", "")),
                dataset_id=str(obj.get("dataset_id", "")),
                relevance=int(obj.get("final_relevance_score", 0)),
            )
        )
    return entries


def _load_qrels_hard_negatives(qrels_path: Path) -> set[str]:
    """Return dataset_ids flagged as hard negatives in adjudicated qrels."""
    if not qrels_path.exists():
        return set()
    hard_negs: set[str] = set()
    for line in qrels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("final_hard_negative_violation"):
            hard_negs.add(str(obj.get("dataset_id", "")))
    return hard_negs


# ---------------------------------------------------------------------------
# Variant ranking builders
# ---------------------------------------------------------------------------


def _rank_lexical(
    query: str,
    concepts: list[ConceptNode],
) -> list[str]:
    dataset_scores = [
        (c.source_ids[0] if c.source_ids else c.concept_id, _lexical_score(query, c))
        for c in concepts
        if c.concept_type == "dataset"
    ]
    dataset_scores.sort(key=lambda x: x[1], reverse=True)
    return [did for did, _ in dataset_scores]


def _rank_variant(
    query: str,
    concepts: list[ConceptNode],
    evidence_links: list[EvidenceLink],
    graph: nx.DiGraph,
    enable_concept: bool,
    enable_evidence: bool,
    enable_penalty: bool,
) -> list[str]:
    results = rerank_datasets(
        query=query,
        concepts=concepts,
        evidence_links=evidence_links,
        graph=graph,
        limit=len(concepts),  # full ranking
        enable_concept_boost=enable_concept,
        enable_evidence_boost=enable_evidence,
        enable_hard_negative_penalty=enable_penalty,
    )
    return [r.dataset_id for r in results]


# ---------------------------------------------------------------------------
# Source skew analysis
# ---------------------------------------------------------------------------


def _compute_source_skew(ranked_ids: list[str], k: int = 10) -> dict[str, float]:
    top = ranked_ids[:k]
    counts: dict[str, int] = {}
    for did in top:
        prefix = did.split("_")[0] if "_" in did else did[:8]
        counts[prefix] = counts.get(prefix, 0) + 1
    return {src: round(cnt / len(top), 3) for src, cnt in sorted(counts.items())}


# ---------------------------------------------------------------------------
# Variant evaluation runner
# ---------------------------------------------------------------------------


def _evaluate_variant(
    variant_name: str,
    ranked_lists: dict[str, list[str]],
    qrels_by_query: dict[str, dict[str, int]],
    hard_negatives: set[str],
) -> VariantMetrics:
    if not qrels_by_query:
        return VariantMetrics(
            variant=variant_name,
            note="No adjudicated qrels available. Cannot compute metrics.",
        )

    ndcg_scores: list[float] = []
    mrr_scores: list[float] = []
    r10_scores: list[float] = []
    r50_scores: list[float] = []
    hn_rates: list[float] = []

    for query_id, qrel_map in qrels_by_query.items():
        ranked = ranked_lists.get(query_id, [])
        if not ranked:
            continue
        ndcg_scores.append(_ndcg_at_k(ranked, qrel_map, k=10))
        mrr_scores.append(_mrr(ranked, qrel_map))
        r10_scores.append(_recall_at_k(ranked, qrel_map, k=10))
        r50_scores.append(_recall_at_k(ranked, qrel_map, k=50))
        hn_rates.append(_hard_negative_violation_rate(ranked, hard_negatives))

    def _mean(vals: list[float]) -> float | None:
        return round(sum(vals) / len(vals), 4) if vals else None

    # Use the first variant's top-10 ranking for source skew
    first_query = next(iter(ranked_lists), None)
    source_skew = {}
    if first_query:
        source_skew = _compute_source_skew(ranked_lists[first_query])

    return VariantMetrics(
        variant=variant_name,
        ndcg_at_10=_mean(ndcg_scores),
        mrr=_mean(mrr_scores),
        recall_at_10=_mean(r10_scores),
        recall_at_50=_mean(r50_scores),
        hard_negative_violation_rate=_mean(hn_rates),
        source_skew=source_skew,
        evaluated_queries=len(ndcg_scores),
    )


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def _render_eval_report_markdown(
    field: str,
    queries: list[dict[str, object]],
    qrels_available: bool,
    qrels_count: int,
    metrics_by_variant: list[VariantMetrics],
    generated_at: str,
) -> str:
    lines = [
        "# Concept Memory Retrieval Evaluation",
        "",
        f"**Field:** {field}  ",
        f"**Generated:** {generated_at}  ",
        f"**Benchmark queries:** {len(queries)}  ",
        f"**Adjudicated qrels:** {qrels_count}  ",
        "",
    ]

    if not qrels_available:
        lines += [
            "## Status: No Qrels Available",
            "",
            "Adjudicated relevance judgements are required for metric computation.",
            "No metrics have been computed. This report is a placeholder.",
            "",
            "### What Is Needed",
            "",
            "1. Export qrels candidates:",
            "   ```",
            "   python -m neural_search.field_state.cli qrels-export \\",
            "       --pool artifacts/field_state/qrels_candidates.jsonl \\",
            "       --queries artifacts/benchmark_queries.jsonl \\",
            "       --corpus data/corpus.jsonl \\",
            "       --vault /path/to/vault",
            "   ```",
            "2. Label the exported Obsidian notes (relevance 0–3 scale).",
            "3. Import reviews:",
            "   ```",
            "   python -m neural_search.field_state.cli qrels-import --vault /path/to/vault",
            "   ```",
            "4. Adjudicate disagreements:",
            "   ```",
            "   python -m neural_search.field_state.cli qrels-adjudicate",
            "   ```",
            "5. Re-run this evaluation:",
            "   ```",
            "   python -m neural_search.field_state.cli concept-eval \\",
            "       --qrels artifacts/field_state/adjudicated_qrels.jsonl \\",
            "       --queries artifacts/benchmark_queries.jsonl \\",
            "       --field neuroscience_dataset_reuse",
            "   ```",
            "",
            "### Benchmark Queries Available",
            "",
        ]
        for q in queries:
            lines.append(f"- `{q.get('query_id')}` [{q.get('intent')}]: {q.get('query')}")
        lines.append("")
        return "\n".join(lines)

    # Metrics table
    lines += [
        "## Ablation Results",
        "",
        "| Variant | NDCG@10 | MRR | Recall@10 | Recall@50 | HN Violation Rate | Queries |",
        "|---------|---------|-----|-----------|-----------|-------------------|---------|",
    ]
    for m in metrics_by_variant:
        def _fmt(v: float | None) -> str:
            return f"{v:.4f}" if v is not None else "—"
        lines.append(
            f"| {m.variant} "
            f"| {_fmt(m.ndcg_at_10)} "
            f"| {_fmt(m.mrr)} "
            f"| {_fmt(m.recall_at_10)} "
            f"| {_fmt(m.recall_at_50)} "
            f"| {_fmt(m.hard_negative_violation_rate)} "
            f"| {m.evaluated_queries} |"
        )
    lines.append("")

    # Notes per variant
    for m in metrics_by_variant:
        if m.note:
            lines.append(f"> **{m.variant}:** {m.note}")
    lines.append("")

    # Source skew for full variant
    full_variant = next((m for m in metrics_by_variant if m.variant == "full"), None)
    if full_variant and full_variant.source_skew:
        lines += [
            "## Source Skew (Top-10, Full Variant)",
            "",
            "| Source Prefix | Fraction |",
            "|---------------|----------|",
        ]
        for src, frac in full_variant.source_skew.items():
            lines.append(f"| {src} | {frac:.3f} |")
        lines.append("")

    lines += [
        "## Methodology Notes",
        "",
        "- Metrics are computed over adjudicated qrels only. Unadjudicated judgements are excluded.",
        "- NDCG@10 uses graded relevance (0–3 scale). MRR uses binary relevance (threshold ≥ 1).",
        "- Hard-negative violation rate = fraction of top-10 results flagged as hard negatives.",
        "- Source skew measures concentration of top-10 results from a single data source.",
        "- These metrics are not yet peer-reviewed. Treat as internal engineering validation only.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public evaluation API
# ---------------------------------------------------------------------------


def run_concept_eval(
    queries_path: Path,
    qrels_path: Path | None,
    field: str,
    out_path: Path | None = None,
    root: Path | None = None,
) -> Path:
    """Run ablation evaluation and write a Markdown report.

    Returns the path to the written report.
    Never fabricates metrics: if qrels are absent, a placeholder report is written.
    """
    resolved_queries = resolve_path(queries_path, root)
    resolved_qrels = resolve_path(qrels_path, root) if qrels_path is not None else None
    resolved_out = resolve_path(out_path or _EVAL_REPORT_PATH, root)

    queries = _load_benchmark_queries(resolved_queries)
    qrels_entries = _load_adjudicated_qrels(resolved_qrels) if resolved_qrels else []
    hard_negatives = _load_qrels_hard_negatives(resolved_qrels) if resolved_qrels else set()

    qrels_available = bool(qrels_entries)

    # Build qrels lookup: {query_id: {dataset_id: relevance}}
    qrels_by_query: dict[str, dict[str, int]] = {}
    for entry in qrels_entries:
        if entry.query_id not in qrels_by_query:
            qrels_by_query[entry.query_id] = {}
        qrels_by_query[entry.query_id][entry.dataset_id] = entry.relevance

    # Load concept memory artifacts once
    concepts, evidence_links = read_concept_artifacts(root)
    graph = build_concept_graph(concepts, evidence_links)

    variant_defs = [
        ("lexical_only", False, False, False),
        ("concept_boost", True, False, False),
        ("concept_boost_ev", True, True, False),
        ("full", True, True, True),
    ]

    # Build ranked lists per variant per query
    ranked_lists_by_variant: dict[str, dict[str, list[str]]] = {}
    for vname, ec, ee, ep in variant_defs:
        ranked_by_query: dict[str, list[str]] = {}
        for q in queries:
            query_id = str(q.get("query_id", ""))
            query_text = str(q.get("query", ""))
            if not query_text:
                continue
            if vname == "lexical_only":
                ranked_by_query[query_id] = _rank_lexical(query_text, concepts)
            else:
                ranked_by_query[query_id] = _rank_variant(
                    query_text, concepts, evidence_links, graph, ec, ee, ep
                )
        ranked_lists_by_variant[vname] = ranked_by_query

    # Compute metrics for each variant
    metrics_list: list[VariantMetrics] = []
    for vname, _, _, _ in variant_defs:
        m = _evaluate_variant(
            vname,
            ranked_lists_by_variant[vname],
            qrels_by_query,
            hard_negatives,
        )
        metrics_list.append(m)

    report_md = _render_eval_report_markdown(
        field=field,
        queries=queries,
        qrels_available=qrels_available,
        qrels_count=len(qrels_entries),
        metrics_by_variant=metrics_list,
        generated_at=datetime.now(UTC).isoformat(),
    )

    resolved_out.parent.mkdir(parents=True, exist_ok=True)
    resolved_out.write_text(report_md, encoding="utf-8")
    return resolved_out
