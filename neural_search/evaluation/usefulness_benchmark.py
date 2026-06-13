"""Graded usefulness benchmark for evaluating latent-usefulness retrieval.

Metrics: NDCG@k, MRR (first useful), Precision@k, hard-negative violation rate,
per-intent breakdown.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class UsefulnessLabel(StrEnum):
    NOT_USEFUL = "not_useful"
    WEAKLY_USEFUL = "weakly_useful"
    USEFUL = "useful"
    HIGHLY_USEFUL = "highly_useful"


GAIN: dict[str, float] = {
    "not_useful": 0.0,
    "weakly_useful": 1.0,
    "useful": 2.0,
    "highly_useful": 3.0,
}

_RELEVANT = {"useful", "highly_useful"}


@dataclass
class UsefulnessQuery:
    query_id: str
    query: str
    intent: str
    candidate_ids: list[str] = field(default_factory=list)


@dataclass
class PairLabel:
    query_id: str
    candidate_id: str
    usefulness_label: str  # UsefulnessLabel value
    label_type: str
    notes: str = ""
    is_hard_negative: bool = False


@dataclass
class BenchmarkReport:
    ndcg_at_k: float
    mrr: float
    precision_at_k: float
    hard_negative_violation_rate: float
    per_intent_metrics: dict[str, dict[str, float]]
    k: int
    n_queries: int
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Usefulness Benchmark Report\n",
            "| Metric | Value |",
            "|--------|-------|",
            f"| NDCG@{self.k} | {self.ndcg_at_k:.4f} |",
            f"| MRR | {self.mrr:.4f} |",
            f"| P@{self.k} | {self.precision_at_k:.4f} |",
            f"| Hard-Neg Violation Rate | {self.hard_negative_violation_rate:.4f} |",
            f"| Queries evaluated | {self.n_queries} |",
            "",
            "## Per-Intent Breakdown\n",
        ]
        for intent, metrics in self.per_intent_metrics.items():
            lines.append(f"### {intent}")
            for m, v in metrics.items():
                lines.append(f"- {m}: {v:.4f}")
        if self.notes:
            lines += ["", "## Notes"] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)


def compute_ndcg_at_k(
    ranked: list[str],
    labels: dict[str, str],
    k: int,
) -> float:
    """Graded NDCG@k using GAIN values."""
    def _dcg(ids: list[str]) -> float:
        return sum(
            GAIN.get(labels.get(cid, "not_useful"), 0.0) / math.log2(i + 2)
            for i, cid in enumerate(ids[:k])
        )

    if not ranked:
        return 0.0

    dcg = _dcg(ranked)
    ideal_order = sorted(labels.keys(), key=lambda c: GAIN.get(labels[c], 0.0), reverse=True)
    idcg = _dcg(ideal_order)
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(ranked: list[str], labels: dict[str, str]) -> float:
    """Mean reciprocal rank of first USEFUL-or-better result."""
    for i, cid in enumerate(ranked):
        if labels.get(cid, "not_useful") in _RELEVANT:
            return 1.0 / (i + 1)
    return 0.0


def compute_precision_at_k(
    ranked: list[str],
    labels: dict[str, str],
    k: int,
) -> float:
    """Fraction of top-k that are USEFUL or HIGHLY_USEFUL."""
    if not ranked:
        return 0.0
    top = ranked[:k]
    relevant = sum(1 for c in top if labels.get(c, "not_useful") in _RELEVANT)
    return relevant / len(top)


def hard_negative_violation_rate(
    ranked: list[str],
    labels: dict[str, str],
    hard_negatives: set[str],
) -> float:
    """Fraction of hard-negatives ranked above the first relevant result."""
    if not hard_negatives:
        return 0.0

    first_relevant = next(
        (i for i, c in enumerate(ranked) if labels.get(c, "not_useful") in _RELEVANT),
        len(ranked),
    )
    violations = sum(
        1 for i, c in enumerate(ranked) if c in hard_negatives and i < first_relevant
    )
    return violations / len(hard_negatives)


def run_usefulness_benchmark(
    queries: list[UsefulnessQuery],
    labels: list[PairLabel],
    run: dict[str, list[str]],
    k: int = 10,
) -> BenchmarkReport:
    """Evaluate a retrieval run against usefulness labels."""
    if not labels:
        raise ValueError("No labels provided to benchmark")

    per_query_labels: dict[str, dict[str, str]] = {}
    per_query_hard_negs: dict[str, set[str]] = {}
    for lbl in labels:
        per_query_labels.setdefault(lbl.query_id, {})[lbl.candidate_id] = lbl.usefulness_label
        if lbl.is_hard_negative or lbl.usefulness_label == "not_useful":
            per_query_hard_negs.setdefault(lbl.query_id, set()).add(lbl.candidate_id)

    ndcgs, mrrs, precs, hn_rates = [], [], [], []
    intent_buckets: dict[str, list[dict[str, float]]] = {}

    for q in queries:
        qid = q.query_id
        ranked = run.get(qid, [])
        qlabels = per_query_labels.get(qid, {})
        if not qlabels:
            continue

        ndcg = compute_ndcg_at_k(ranked, qlabels, k)
        mrr = compute_mrr(ranked, qlabels)
        prec = compute_precision_at_k(ranked, qlabels, k)
        hnv = hard_negative_violation_rate(ranked, qlabels, per_query_hard_negs.get(qid, set()))

        ndcgs.append(ndcg)
        mrrs.append(mrr)
        precs.append(prec)
        hn_rates.append(hnv)

        bucket = intent_buckets.setdefault(q.intent, [])
        bucket.append({"ndcg": ndcg, "mrr": mrr, "precision": prec})

    def _avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    per_intent: dict[str, dict[str, float]] = {}
    for intent, metrics_list in intent_buckets.items():
        per_intent[intent] = {
            m: _avg([x[m] for x in metrics_list])
            for m in ("ndcg", "mrr", "precision")
        }

    return BenchmarkReport(
        ndcg_at_k=_avg(ndcgs),
        mrr=_avg(mrrs),
        precision_at_k=_avg(precs),
        hard_negative_violation_rate=_avg(hn_rates),
        per_intent_metrics=per_intent,
        k=k,
        n_queries=len(ndcgs),
    )


def load_seed_pairs(path: str | Path) -> tuple[list[UsefulnessQuery], list[PairLabel]]:
    """Load seed pairs JSONL into queries and labels."""
    path = Path(path)
    raw: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    query_map: dict[str, UsefulnessQuery] = {}
    labels: list[PairLabel] = []

    for row in raw:
        qid = row["query_id"]
        if qid not in query_map:
            query_map[qid] = UsefulnessQuery(
                query_id=qid,
                query=row.get("query", ""),
                intent=row.get("intent", "strict_lookup"),
                candidate_ids=[],
            )
        query_map[qid].candidate_ids.append(row["candidate_id"])
        labels.append(
            PairLabel(
                query_id=qid,
                candidate_id=row["candidate_id"],
                usefulness_label=row.get("usefulness_label", "not_useful"),
                label_type=row.get("label_type", ""),
                notes=row.get("notes", ""),
                is_hard_negative="hard negative" in row.get("notes", "").lower(),
            )
        )

    return list(query_map.values()), labels
