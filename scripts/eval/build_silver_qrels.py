#!/usr/bin/env python3
"""Build silver (machine-generated) qrels from labeling functions, affordance
probes, concept-memory signals, and an optional LLM judge.

Usage:
    python scripts/eval/build_silver_qrels.py \\
        --queries artifacts/benchmark_queries.jsonl \\
        --pool reports/eval/benchmark_pool.jsonl \\
        --corpus data/corpus/normalized/combined_corpus.jsonl \\
        --out artifacts/qrels_silver.jsonl \\
        --summary reports/eval/silver_qrels_summary.md \\
        --disagreements reports/eval/silver_qrels_disagreements.md \\
        --seed 13

Optional LLM judge (requires ANTHROPIC_API_KEY):
    python scripts/eval/build_silver_qrels.py ... --use-llm-judge

IMPORTANT: This script never writes to artifacts/qrels.jsonl.
Silver labels are NOT gold labels and MUST NOT be reported as expert validation.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from scripts.eval.affordance_probes import apply_affordance_probes
from scripts.eval.benchmark_schema import BenchmarkQueryV1
from scripts.eval.concept_labeler import label_from_concept_result
from scripts.eval.labeling_functions import apply_all_labeling_functions
from scripts.eval.llm_judge import LLMJudgeProtocol, build_judge
from scripts.eval.silver_qrels_schema import (
    SILVER_EVAL_WATERMARK,
    LabelingFunctionVote,
    SilverQrelsEntry,
    SilverQrelsSummary,
)

# ---------------------------------------------------------------------------
# Safety guard — never write to gold qrels path
# ---------------------------------------------------------------------------

GOLD_QRELS_PATH = Path("artifacts/qrels.jsonl")
_DEFAULT_SILVER_PATH = Path("artifacts/qrels_silver.jsonl")


def _guard_output_path(path: Path) -> None:
    if path.resolve() == GOLD_QRELS_PATH.resolve():
        print(
            "ERROR: refusing to write silver qrels to artifacts/qrels.jsonl "
            "(the gold qrels path). Use --out to specify a different output path.",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_queries(path: Path) -> dict[str, BenchmarkQueryV1]:
    queries: dict[str, BenchmarkQueryV1] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            q = BenchmarkQueryV1.model_validate(json.loads(line))
            queries[q.query_id] = q
    return queries


def _load_pool(path: Path) -> list[dict[str, Any]]:
    pool: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pool.append(json.loads(line))
    return pool


def _load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    corpus: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Normalise record ID — pool uses record_id, corpus uses source_id
            rid = rec.get("source_id") or rec.get("record_id") or rec.get("id", "")
            if rid:
                corpus[str(rid)] = rec
                # Also index as "source:source_id" style
                src = rec.get("source", "")
                if src:
                    corpus[f"{src}:{rid}"] = rec
    return corpus


# ---------------------------------------------------------------------------
# Vote aggregation
# ---------------------------------------------------------------------------


def _aggregate_votes(votes: list[LabelingFunctionVote]) -> tuple[int, float, list[str]]:
    """Aggregate labeling function votes into (relevance, confidence, disagreements).

    Rules:
    - Hard-negative violation from ANY labeler → relevance = 0, high confidence.
    - Otherwise: weighted median of non-abstain votes, with confidence
      proportional to vote agreement.
    """
    # Check hard-negative override
    hn_votes = [v for v in votes if v.vote == 0 and any("hard_negative" in e for e in v.evidence)]
    if hn_votes:
        conf = max(v.confidence for v in hn_votes)
        return 0, conf, [f"hard_negative_override from {v.source}" for v in hn_votes]

    real_votes = [v for v in votes if v.vote is not None]
    if not real_votes:
        return 1, 0.30, ["all labelers abstained; defaulting to 1"]

    # Weighted mean (by confidence) — votes are guaranteed non-None here
    weighted_sum = sum(int(v.vote) * v.confidence for v in real_votes)
    weight_total = sum(v.confidence for v in real_votes)
    raw_mean = weighted_sum / weight_total if weight_total > 0 else 1.0

    # Round to nearest integer 0–3
    relevance = max(0, min(3, round(raw_mean)))

    # Confidence: agreement score — 1.0 when all votes equal, decreases with spread
    vote_values = [int(v.vote) for v in real_votes]
    if len(set(vote_values)) == 1:
        agreement = 1.0
    else:
        spread = stdev(float(x) for x in vote_values) if len(vote_values) > 1 else 0.0
        agreement = max(0.0, 1.0 - spread / 3.0)

    avg_labeler_conf = mean(v.confidence for v in real_votes)
    confidence = 0.5 * agreement + 0.5 * avg_labeler_conf

    # Detect disagreements
    disagreements: list[str] = []
    if len(set(vote_values)) > 1:
        by_src = {v.source: v.vote for v in real_votes}
        disagreements.append(f"vote spread: {by_src}")

    return relevance, round(confidence, 3), disagreements


def _needs_review(
    confidence: float,
    disagreements: list[str],
    hn_violation: bool,
    missing_metadata: list[str],
) -> tuple[bool, float]:
    """Return (needs_review, review_priority) in [0, 1]."""
    priority = 0.0
    if confidence < 0.45:
        priority += 0.35
    if len(disagreements) >= 1:
        priority += 0.25
    if hn_violation:
        priority += 0.20
    if missing_metadata:
        priority += 0.10 * min(len(missing_metadata), 2)

    priority = min(priority, 1.0)
    needs = confidence < 0.50 or len(disagreements) >= 1 or hn_violation
    return needs, round(priority, 3)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_silver_qrels(
    queries: dict[str, BenchmarkQueryV1],
    pool: list[dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    seed: int = 13,
    judge: LLMJudgeProtocol | None = None,
    concept_results: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> list[SilverQrelsEntry]:
    """Build silver qrels for all (query, dataset) pairs in the pool.

    Parameters
    ----------
    queries : loaded benchmark queries keyed by query_id
    pool    : list of pool records, each with query_id and record_id
    corpus  : corpus records keyed by source_id (and "source:source_id")
    seed    : random seed for reproducibility
    judge   : optional LLM judge (None → disabled)
    concept_results : optional map of (query_id, dataset_id) → concept result dict
    """
    rng = random.Random(seed)  # noqa: F841  (available for future stochastic steps)
    entries: list[SilverQrelsEntry] = []

    for pool_item in pool:
        query_id = str(pool_item.get("query_id", ""))
        record_id = str(pool_item.get("record_id", "") or pool_item.get("dataset_id", ""))

        query = queries.get(query_id)
        if query is None:
            continue

        record = corpus.get(record_id)
        if record is None:
            # Try alternate key forms
            for key in (f"dandi:{record_id}", f"openneuro:{record_id}", record_id.split(":")[-1]):
                record = corpus.get(key)
                if record is not None:
                    break
        if record is None:
            record = {}

        # --- Rule-based labeling functions
        lf_votes = apply_all_labeling_functions(query, record)

        # --- Affordance probes (convert to votes, pick the most relevant probe)
        probe_pairs = apply_affordance_probes(record)
        probe_votes = [vote for _, vote in probe_pairs]

        # --- Concept-memory labeler
        concept_vote: LabelingFunctionVote | None = None
        if concept_results is not None:
            cr = concept_results.get((query_id, record_id))
            if cr:
                concept_vote = label_from_concept_result(query, cr)

        # --- Optional LLM judge
        judge_vote: LabelingFunctionVote | None = None
        if judge is not None:
            cr_for_judge = (concept_results or {}).get((query_id, record_id))
            judge_vote = judge.judge(query, record, cr_for_judge)

        # Combine all votes
        all_votes: list[LabelingFunctionVote] = lf_votes[:]
        all_votes.extend(probe_votes)
        if concept_vote is not None:
            all_votes.append(concept_vote)
        if judge_vote is not None:
            all_votes.append(judge_vote)

        # Aggregate
        relevance, confidence, disagreements = _aggregate_votes(all_votes)

        # Collect cross-cutting evidence
        hn_violation = any(v.vote == 0 and any("hard_negative" in e for e in v.evidence)
                           for v in all_votes)
        missing_meta = list({
            item
            for v in all_votes
            for e in v.evidence
            if e.startswith("missing")
            for item in [e.split(":", 1)[-1].strip()]
        })

        needs_review, priority = _needs_review(confidence, disagreements, hn_violation, missing_meta)

        # Combined evidence strings (from all non-abstaining votes)
        combined_evidence = list(dict.fromkeys(
            e for v in all_votes if v.vote is not None for e in v.evidence
        ))

        votes_dict = {v.source: v.vote for v in all_votes if v.vote is not None}
        label_sources = list(dict.fromkeys(v.source for v in all_votes if v.vote is not None))

        entry = SilverQrelsEntry(
            query_id=query_id,
            dataset_id=record_id,
            relevance=relevance,
            confidence=confidence,
            label_sources=label_sources,
            votes=votes_dict,
            hard_negative_violation=hn_violation,
            missing_metadata=missing_meta,
            evidence=combined_evidence[:20],
            disagreements=disagreements,
            needs_human_review=needs_review,
            review_priority=priority,
            all_votes=all_votes,
            seed=seed,
        )
        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Summary and reports
# ---------------------------------------------------------------------------


def build_summary(entries: list[SilverQrelsEntry], seed: int | None) -> SilverQrelsSummary:
    if not entries:
        return SilverQrelsSummary(total_labels=0, seed=seed)

    by_rel: dict[str, int] = defaultdict(int)
    for e in entries:
        by_rel[str(e.relevance)] += 1

    confidences = [e.confidence for e in entries]
    abstention_by_labeler: dict[str, int] = defaultdict(int)
    for e in entries:
        for v in e.all_votes:
            if v.vote is None:
                abstention_by_labeler[v.source] += 1

    disagreement_count = sum(1 for e in entries if e.disagreements)

    hc_pos = [
        {"query_id": e.query_id, "dataset_id": e.dataset_id,
         "relevance": e.relevance, "confidence": e.confidence,
         "evidence": e.evidence[:3]}
        for e in sorted(entries, key=lambda x: -x.confidence)
        if e.relevance >= 2
    ][:5]

    hc_neg = [
        {"query_id": e.query_id, "dataset_id": e.dataset_id,
         "relevance": e.relevance, "confidence": e.confidence,
         "evidence": e.evidence[:3]}
        for e in sorted(entries, key=lambda x: -x.confidence)
        if e.relevance == 0
    ][:5]

    review_examples = [
        {"query_id": e.query_id, "dataset_id": e.dataset_id,
         "relevance": e.relevance, "confidence": e.confidence,
         "disagreements": e.disagreements[:2]}
        for e in sorted(entries, key=lambda x: -x.review_priority)
        if e.needs_human_review
    ][:5]

    return SilverQrelsSummary(
        total_labels=len(entries),
        by_relevance=dict(by_rel),
        confidence_mean=round(mean(confidences), 3),
        confidence_below_0_5=sum(1 for c in confidences if c < 0.5),
        needs_human_review_count=sum(1 for e in entries if e.needs_human_review),
        hard_negative_violation_count=sum(1 for e in entries if e.hard_negative_violation),
        abstention_by_labeler=dict(abstention_by_labeler),
        disagreement_rate=round(disagreement_count / len(entries), 3),
        high_confidence_positives=hc_pos,
        high_confidence_negatives=hc_neg,
        examples_needing_review=review_examples,
        seed=seed,
    )


def _write_summary_md(summary: SilverQrelsSummary, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Silver Qrels Summary\n",
        f"> **{SILVER_EVAL_WATERMARK}**\n",
        f"Generated with seed={summary.seed}\n",
        "\n## Counts\n",
        f"- Total labels: **{summary.total_labels}**",
        f"- Needs human review: **{summary.needs_human_review_count}**",
        f"- Hard-negative violations: **{summary.hard_negative_violation_count}**",
        f"- Confidence mean: {summary.confidence_mean:.3f}",
        f"- Confidence < 0.5: {summary.confidence_below_0_5}",
        f"- Disagreement rate: {summary.disagreement_rate:.1%}",
        "\n## Labels by relevance\n",
    ]
    for k in ("0", "1", "2", "3"):
        n = summary.by_relevance.get(k, 0)
        lines.append(f"- relevance={k}: {n}")

    lines.append("\n## Abstentions by labeler\n")
    for src, n in sorted(summary.abstention_by_labeler.items()):
        lines.append(f"- {src}: {n}")

    if summary.high_confidence_positives:
        lines.append("\n## High-confidence positives (sample)\n")
        for ex in summary.high_confidence_positives:
            lines.append(
                f"- `{ex['query_id']}` × `{ex['dataset_id']}` "
                f"→ rel={ex['relevance']} conf={ex['confidence']:.2f}"
            )

    if summary.high_confidence_negatives:
        lines.append("\n## High-confidence negatives (sample)\n")
        for ex in summary.high_confidence_negatives:
            lines.append(
                f"- `{ex['query_id']}` × `{ex['dataset_id']}` "
                f"→ rel={ex['relevance']} conf={ex['confidence']:.2f}"
            )

    if summary.examples_needing_review:
        lines.append("\n## Examples needing human review (sample)\n")
        for ex in summary.examples_needing_review:
            lines.append(
                f"- `{ex['query_id']}` × `{ex['dataset_id']}` "
                f"conf={ex['confidence']:.2f} — {ex['disagreements']}"
            )

    lines.append(
        "\n## Known limitations\n"
        "- Labels are rule-based and have not been adjudicated.\n"
        "- Concept-memory votes reflect corpus metadata, not peer review.\n"
        "- Species/modality synonym coverage is partial.\n"
        "- Hard-negative matching uses keyword heuristics, not semantic reasoning.\n"
        "- These labels are for development use only.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_disagreements_md(entries: list[SilverQrelsEntry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    disagreed = [e for e in entries if e.disagreements]
    lines = [
        "# Silver Qrels Disagreements\n",
        f"> **{SILVER_EVAL_WATERMARK}**\n",
        f"Total entries with disagreements: **{len(disagreed)}** / {len(entries)}\n",
    ]
    for e in sorted(disagreed, key=lambda x: -x.review_priority)[:50]:
        lines.append(
            f"\n### `{e.query_id}` × `{e.dataset_id}`\n"
            f"- Silver relevance: {e.relevance}  confidence: {e.confidence:.2f}\n"
            f"- Disagreements: {e.disagreements}\n"
            f"- Votes: {e.votes}\n"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build silver qrels.")
    p.add_argument("--queries", required=True, type=Path)
    p.add_argument("--pool", required=True, type=Path)
    p.add_argument("--corpus", required=True, type=Path)
    p.add_argument("--out", default=_DEFAULT_SILVER_PATH, type=Path)
    p.add_argument("--summary", default=Path("reports/eval/silver_qrels_summary.md"), type=Path)
    p.add_argument("--disagreements", default=Path("reports/eval/silver_qrels_disagreements.md"), type=Path)
    p.add_argument("--seed", default=13, type=int)
    p.add_argument("--use-llm-judge", action="store_true", default=False)
    p.add_argument("--allow-overwrite", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _guard_output_path(args.out)

    if not args.allow_overwrite and args.out.exists():
        print(
            f"INFO: {args.out} already exists. Use --allow-overwrite to regenerate.",
            file=sys.stderr,
        )
        sys.exit(0)

    print(f"Loading queries from {args.queries}...")
    queries = _load_queries(args.queries)
    print(f"  {len(queries)} queries loaded")

    print(f"Loading pool from {args.pool}...")
    pool = _load_pool(args.pool)
    print(f"  {len(pool)} pool items loaded")

    print(f"Loading corpus from {args.corpus}...")
    corpus = _load_corpus(args.corpus)
    print(f"  {len(corpus)} corpus records indexed")

    judge: LLMJudgeProtocol | None = None
    if args.use_llm_judge:
        judge = build_judge(use_llm_judge=True)
        if judge is None:
            print("WARNING: LLM judge requested but unavailable (check ANTHROPIC_API_KEY).",
                  file=sys.stderr)

    print("Building silver qrels...")
    entries = build_silver_qrels(
        queries=queries,
        pool=pool,
        corpus=corpus,
        seed=args.seed,
        judge=judge,
    )
    print(f"  {len(entries)} silver qrels generated")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(e.model_dump_json() + "\n")
    print(f"Silver qrels written to {args.out}")

    summary = build_summary(entries, seed=args.seed)
    _write_summary_md(summary, args.summary)
    print(f"Summary written to {args.summary}")

    _write_disagreements_md(entries, args.disagreements)
    print(f"Disagreements written to {args.disagreements}")


if __name__ == "__main__":
    main()
