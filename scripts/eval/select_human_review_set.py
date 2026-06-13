#!/usr/bin/env python3
"""Select the highest-value examples for human review from silver qrels.

Review priority is increased for:
1. Low confidence
2. Strong labeler disagreement
3. Hard-negative conflict
4. Top-ranked results from any retrieval system
5. Large ranking disagreement between baseline and concept-full
6. Candidates likely to affect NDCG@10
7. Sparse metadata but possible relevance
8. Per-intent undercoverage
9. Per-source undercoverage

Usage:
    python scripts/eval/select_human_review_set.py \\
        --silver artifacts/qrels_silver.jsonl \\
        --queries artifacts/benchmark_queries.jsonl \\
        --pool reports/eval/benchmark_pool.jsonl \\
        --out artifacts/qrels_review_queue.jsonl \\
        --limit 300 \\
        --seed 13
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.eval.benchmark_schema import BenchmarkQueryV1
from scripts.eval.silver_qrels_schema import (
    SILVER_EVAL_WATERMARK,
    ReviewQueueEntry,
    SilverQrelsEntry,
)

_DEFAULT_LIMIT = 300


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_silver(path: Path) -> list[SilverQrelsEntry]:
    entries: list[SilverQrelsEntry] = []
    if not path.exists():
        return entries
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(SilverQrelsEntry.model_validate(json.loads(line)))
    return entries


def _load_queries(path: Path) -> dict[str, BenchmarkQueryV1]:
    queries: dict[str, BenchmarkQueryV1] = {}
    if not path.exists():
        return queries
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            q = BenchmarkQueryV1.model_validate(json.loads(line))
            queries[q.query_id] = q
    return queries


def _load_pool(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    pool: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return pool
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = str(rec.get("query_id", ""))
            rid = str(rec.get("record_id", "") or rec.get("dataset_id", ""))
            if qid and rid:
                pool[(qid, rid)] = rec
    return pool


# ---------------------------------------------------------------------------
# Priority scoring
# ---------------------------------------------------------------------------


def _compute_priority(
    entry: SilverQrelsEntry,
    pool_item: dict[str, Any] | None,
    intent_coverage: dict[str, int],
    source_coverage: dict[str, int],
    query: BenchmarkQueryV1 | None,
) -> float:
    """Compute [0, 1] review priority for a silver qrels entry."""
    score = entry.review_priority  # base from silver builder

    # Low confidence
    if entry.confidence < 0.35:
        score += 0.20
    elif entry.confidence < 0.50:
        score += 0.10

    # Hard-negative conflict
    if entry.hard_negative_violation:
        score += 0.15

    # Disagreement
    if entry.disagreements:
        score += 0.10 * min(len(entry.disagreements), 2)

    # Pool-based signals
    if pool_item:
        min_rank: int = int(str(pool_item.get("min_rank", 999)))
        priority_flag: int = int(str(pool_item.get("priority", 0)))

        # Top-ranked results are high value
        if min_rank <= 5:
            score += 0.20
        elif min_rank <= 10:
            score += 0.10

        # Appears in many retrieval variants → contested
        if priority_flag >= 3:
            score += 0.10

    # Potentially affects NDCG@10 (relevance >= 1 but uncertain)
    if 1 <= entry.relevance <= 2 and entry.confidence < 0.60:
        score += 0.08

    # Sparse metadata but non-zero relevance signal
    if entry.missing_metadata and entry.relevance >= 1:
        score += 0.05 * min(len(entry.missing_metadata), 3)

    # Per-intent undercoverage — boost if intent is underrepresented
    if query:
        intent = query.canonical_intent()
        if intent_coverage.get(intent, 0) < 5:
            score += 0.08

    # Per-source undercoverage — boost if dataset source is rare in queue
    if source_coverage:
        src = entry.dataset_id.split(":")[0] if ":" in entry.dataset_id else "unknown"
        if source_coverage.get(src, 0) < 3:
            score += 0.05

    return float(min(score, 1.0))


# ---------------------------------------------------------------------------
# Review queue entry construction
# ---------------------------------------------------------------------------


def _why_selected(
    entry: SilverQrelsEntry,
    pool_item: dict[str, Any] | None,
) -> str:
    reasons: list[str] = []
    if entry.confidence < 0.45:
        reasons.append(f"low confidence ({entry.confidence:.2f})")
    if entry.disagreements:
        reasons.append("labeler disagreement")
    if entry.hard_negative_violation:
        reasons.append("hard-negative conflict")
    if pool_item:
        mr = int(pool_item.get("min_rank", 999))
        if mr <= 10:
            reasons.append(f"top-ranked result (rank {mr})")
    if entry.missing_metadata:
        reasons.append(f"missing metadata: {entry.missing_metadata[:2]}")
    return "; ".join(reasons) if reasons else "general coverage"


def _fields_needed(entry: SilverQrelsEntry, query: BenchmarkQueryV1 | None) -> list[str]:
    needed: list[str] = list(entry.missing_metadata)
    if query:
        for req in query.must_have:
            field = req.split(":")[0]
            if field not in needed:
                needed.append(field)
    return list(dict.fromkeys(needed))[:6]


# ---------------------------------------------------------------------------
# Main selector
# ---------------------------------------------------------------------------


def select_review_set(
    silver: list[SilverQrelsEntry],
    queries: dict[str, BenchmarkQueryV1],
    pool: dict[tuple[str, str], dict[str, Any]],
    limit: int = _DEFAULT_LIMIT,
    seed: int = 13,
) -> list[ReviewQueueEntry]:
    """Select up to ``limit`` entries for human review, prioritised by value."""
    rng = random.Random(seed)

    # Compute coverage counters for diversity boosting
    intent_coverage: dict[str, int] = defaultdict(int)
    source_coverage: dict[str, int] = defaultdict(int)

    # Score all entries
    scored: list[tuple[float, SilverQrelsEntry]] = []
    for entry in silver:
        query = queries.get(entry.query_id)
        pool_item = pool.get((entry.query_id, entry.dataset_id))
        priority = _compute_priority(entry, pool_item, intent_coverage, source_coverage, query)
        scored.append((priority, entry))

    # Sort by priority descending, break ties randomly
    scored.sort(key=lambda x: (-x[0], rng.random()))

    # Convert to ReviewQueueEntry, tracking coverage for diversity
    result: list[ReviewQueueEntry] = []
    for priority_score, entry in scored[:limit]:
        query = queries.get(entry.query_id)
        pool_item = pool.get((entry.query_id, entry.dataset_id))

        # Update coverage
        if query:
            intent_coverage[query.canonical_intent()] += 1
        src = entry.dataset_id.split(":")[0] if ":" in entry.dataset_id else "unknown"
        source_coverage[src] += 1

        rq = ReviewQueueEntry(
            query_id=entry.query_id,
            dataset_id=entry.dataset_id,
            query_text=query.query_text if query else "",
            query_intent=query.canonical_intent() if query else "",
            dataset_title=str((pool_item or {}).get("dataset_title", "")),
            dataset_source=entry.dataset_id.split(":")[0] if ":" in entry.dataset_id else "",
            silver_relevance=entry.relevance,
            silver_confidence=entry.confidence,
            disagreement_summary="; ".join(entry.disagreements[:3]),
            why_selected=_why_selected(entry, pool_item),
            annotation_priority=round(priority_score, 3),
            fields_needed=_fields_needed(entry, query),
            hard_negative_violation=entry.hard_negative_violation,
            missing_metadata=entry.missing_metadata,
            evidence=entry.evidence[:6],
            all_votes=entry.all_votes,
        )
        result.append(rq)

    return result


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------


def _write_queue_summary(queue: list[ReviewQueueEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    intent_counts: dict[str, int] = defaultdict(int)
    source_counts: dict[str, int] = defaultdict(int)
    for e in queue:
        intent_counts[e.query_intent] += 1
        source_counts[e.dataset_source or "unknown"] += 1

    lines = [
        "# Review Queue Summary\n",
        f"> **{SILVER_EVAL_WATERMARK}**\n",
        f"Total entries selected for review: **{len(queue)}**\n",
        "\n## By intent\n",
    ]
    for intent, n in sorted(intent_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {intent}: {n}")
    lines.append("\n## By source\n")
    for src, n in sorted(source_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {src}: {n}")

    lines.append("\n## Top-priority entries (first 20)\n")
    for e in queue[:20]:
        lines.append(
            f"- `{e.query_id}` × `{e.dataset_id}` "
            f"silver={e.silver_relevance} conf={e.silver_confidence:.2f} "
            f"priority={e.annotation_priority:.2f} — {e.why_selected}"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select examples for human review.")
    p.add_argument("--silver", required=True, type=Path)
    p.add_argument("--queries", required=True, type=Path)
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--out", default=Path("artifacts/qrels_review_queue.jsonl"), type=Path)
    p.add_argument("--summary", default=Path("reports/eval/review_queue_summary.md"), type=Path)
    p.add_argument("--limit", default=_DEFAULT_LIMIT, type=int)
    p.add_argument("--seed", default=13, type=int)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    silver = _load_silver(args.silver)
    queries = _load_queries(args.queries)
    pool = _load_pool(args.pool)

    print(f"Loaded {len(silver)} silver qrels, {len(queries)} queries, {len(pool)} pool items")
    queue = select_review_set(silver, queries, pool, limit=args.limit, seed=args.seed)
    print(f"Selected {len(queue)} entries for review")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for e in queue:
            fh.write(e.model_dump_json() + "\n")
    print(f"Review queue written to {args.out}")

    _write_queue_summary(queue, args.summary)
    print(f"Summary written to {args.summary}")


if __name__ == "__main__":
    main()
