#!/usr/bin/env python3
"""Expand the benchmark candidate pool to cover all 15 queries.

Reads benchmark queries from artifacts/benchmark_queries.jsonl, runs BM25
retrieval against the normalized combined corpus, and writes ranked
candidates to artifacts/field_state/qrels_candidates_full.jsonl.

Deduplicates against existing candidates already in
artifacts/field_state/qrels_candidates.jsonl.

Usage:
    python scripts/eval/expand_candidate_pool.py
    python scripts/eval/expand_candidate_pool.py --top-k 20 --corpus data/corpus/normalized/combined_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

QUERIES_PATH = ROOT / "artifacts" / "benchmark_queries.jsonl"
CORPUS_PATH = ROOT / "data" / "corpus" / "normalized" / "combined_corpus.jsonl"
EXISTING_CANDIDATES_PATH = ROOT / "artifacts" / "field_state" / "qrels_candidates.jsonl"
OUTPUT_PATH = ROOT / "artifacts" / "field_state" / "qrels_candidates_full.jsonl"

SCHEMA_VERSION = "0.3"


def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    if not text:
        return []
    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def build_bm25_index(records: list[dict]) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Build BM25 index over title + description fields.

    Returns:
        idf: token → idf score
        doc_tf: record_id → {token: term_freq}
    """
    n = len(records)
    df: Counter[str] = Counter()
    doc_tf: dict[str, dict[str, float]] = {}
    doc_lengths: dict[str, int] = {}

    for rec in records:
        rec_id = _record_id(rec)
        text = _record_text(rec)
        tokens = tokenize(text)
        tf = Counter(tokens)
        doc_tf[rec_id] = dict(tf)
        doc_lengths[rec_id] = len(tokens)
        for tok in set(tokens):
            df[tok] += 1

    idf: dict[str, float] = {}
    for tok, freq in df.items():
        idf[tok] = math.log((n - freq + 0.5) / (freq + 0.5) + 1)

    avg_dl = sum(doc_lengths.values()) / max(len(doc_lengths), 1)
    return idf, doc_tf, doc_lengths, avg_dl


def bm25_score(
    query_tokens: list[str],
    rec_id: str,
    idf: dict[str, float],
    doc_tf: dict[str, dict[str, float]],
    doc_lengths: dict[str, int],
    avg_dl: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    tf = doc_tf.get(rec_id, {})
    dl = doc_lengths.get(rec_id, 0)
    score = 0.0
    for tok in query_tokens:
        tok_tf = tf.get(tok, 0.0)
        tok_idf = idf.get(tok, 0.0)
        numerator = tok_tf * (k1 + 1)
        denominator = tok_tf + k1 * (1 - b + b * dl / max(avg_dl, 1))
        score += tok_idf * (numerator / max(denominator, 1e-9))
    return score


def _record_id(rec: dict) -> str:
    source = rec.get("source", "unknown")
    source_id = rec.get("source_id", "unknown")
    return f"{source}:{source_id}"


def _record_text(rec: dict) -> str:
    parts = [
        rec.get("title") or "",
        rec.get("description") or "",
        " ".join(rec.get("tasks") or []),
        " ".join(rec.get("modalities") or []),
        " ".join(rec.get("brain_regions") or []),
        " ".join(rec.get("species") or []),
    ]
    return " ".join(p for p in parts if p)


def load_queries(path: Path) -> list[dict]:
    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def load_corpus(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_existing_candidate_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                ids.add(obj.get("id", ""))
    return ids


def _candidate_id(query_id: str, dataset_id: str) -> str:
    return f"qrels_candidate:{query_id}:{dataset_id}"


def retrieve_top_k(
    query: dict,
    records: list[dict],
    idf: dict[str, float],
    doc_tf: dict[str, dict[str, float]],
    doc_lengths: dict[str, int],
    avg_dl: float,
    top_k: int,
) -> list[tuple[str, float, dict]]:
    """Return (record_id, score, record) sorted by BM25 score descending."""
    query_text = query.get("query_text") or query.get("query") or ""
    query_tokens = tokenize(query_text)

    scored: list[tuple[float, str, dict]] = []
    for rec in records:
        rec_id = _record_id(rec)
        score = bm25_score(query_tokens, rec_id, idf, doc_tf, doc_lengths, avg_dl)
        if score > 0:
            scored.append((score, rec_id, rec))

    scored.sort(reverse=True)
    return [(rec_id, score, rec) for score, rec_id, rec in scored[:top_k]]


def build_candidate(
    query: dict,
    rank: int,
    rec_id: str,
    bm25_score_val: float,
    record: dict,
) -> dict:
    query_id = query["query_id"]
    return {
        "id": _candidate_id(query_id, rec_id),
        "query_id": query_id,
        "query_text": query.get("query_text") or query.get("query", ""),
        "query_intent": query.get("intent", ""),
        "dataset_id": rec_id,
        "dataset_title": record.get("title", ""),
        "dataset_source": record.get("source", ""),
        "dataset_description": record.get("description") or "",
        "rank": rank,
        "retrieval_score": round(bm25_score_val, 4),
        "retrieval_method": "bm25",
        "hard_negative_reason": None,
        "expected_relevance_hint": "needs_annotation",
        "field": "neuroscience_dataset_reuse",
        "source_artifacts": [
            "artifacts/benchmark_queries.jsonl",
            str(CORPUS_PATH.relative_to(ROOT)),
        ],
        "metadata": {
            "pool": {
                "query_id": query_id,
                "record_id": rec_id,
                "pooled_from": ["bm25"],
                "min_rank": rank,
                "priority": 2,
                "status": "needs_annotation",
            },
            "query_known_failure_modes": query.get("known_failure_modes", [])
            or query.get("hard_negatives", []),
            "record_species": record.get("species") or [],
            "record_modalities": record.get("modalities") or [],
            "record_tasks": record.get("tasks") or [],
            "record_brain_regions": record.get("brain_regions") or [],
        },
        "schema_version": SCHEMA_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand benchmark candidate pool to all queries.")
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--existing", type=Path, default=EXISTING_CANDIDATES_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=20, metavar="K")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.queries.exists():
        print(f"ERROR: queries file not found: {args.queries}", file=sys.stderr)
        sys.exit(1)
    if not args.corpus.exists():
        print(f"ERROR: corpus not found: {args.corpus}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading queries from {args.queries}...")
    queries = load_queries(args.queries)
    print(f"  {len(queries)} queries")

    print(f"Loading corpus from {args.corpus}...")
    records = load_corpus(args.corpus)
    print(f"  {len(records)} records")

    print("Building BM25 index...")
    idf, doc_tf, doc_lengths, avg_dl = build_bm25_index(records)
    print(f"  {len(idf)} unique tokens")

    print(f"Loading existing candidates from {args.existing}...")
    existing_ids = load_existing_candidate_ids(args.existing)
    print(f"  {len(existing_ids)} existing candidates")

    all_new: list[dict] = []
    stats: dict[str, int] = defaultdict(int)

    for query in queries:
        query_id = query["query_id"]
        top_results = retrieve_top_k(query, records, idf, doc_tf, doc_lengths, avg_dl, args.top_k)

        new_for_query = 0
        for rank, (rec_id, score, record) in enumerate(top_results, start=1):
            cid = _candidate_id(query_id, rec_id)
            if cid in existing_ids:
                stats["skipped_existing"] += 1
                continue
            candidate = build_candidate(query, rank, rec_id, score, record)
            all_new.append(candidate)
            existing_ids.add(cid)
            new_for_query += 1

        stats[f"{query_id}_new"] = new_for_query
        print(f"  {query_id}: {new_for_query} new candidates (top-{args.top_k} from {len(top_results)} scored)")

    print(f"\nTotal new candidates: {len(all_new)}")
    print(f"Skipped (already existed): {stats['skipped_existing']}")

    if args.dry_run:
        print("DRY RUN — not writing output")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for candidate in all_new:
            f.write(json.dumps(candidate) + "\n")
    print(f"Wrote {len(all_new)} candidates to {args.output}")


if __name__ == "__main__":
    main()
