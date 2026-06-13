#!/usr/bin/env python3
"""Build multi-system pooled qrels candidates for all benchmark queries.

Retrieval systems:
  bm25         - BM25 on title+description+tasks+modalities+brain_regions+species
  dense_prf    - Pseudo-relevance feedback via averaged BM25-top-5 BGE-large embeddings
  hybrid_rrf   - Reciprocal Rank Fusion of bm25 + dense_prf
  usefulness   - BM25 top-50 re-ranked by structured usefulness score

Output: artifacts/field_state/qrels_candidates_pooled.jsonl

Usage:
    python scripts/eval/build_pooled_qrels_candidates.py
    python scripts/eval/build_pooled_qrels_candidates.py --top-k 20 --dry-run
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

QUERIES_PATH = ROOT / "artifacts" / "benchmark_queries.jsonl"
CORPUS_PATH = ROOT / "data" / "corpus" / "normalized" / "combined_corpus.jsonl"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "real_all.dense.field_embeddings.jsonl"
OUTPUT_PATH = ROOT / "artifacts" / "field_state" / "qrels_candidates_pooled.jsonl"

SCHEMA_VERSION = "0.3"
TARGET_FIELD = "combined_scientific_summary"
PRF_FEEDBACK_K = 5  # number of BM25 results to use as pseudo-query
RRF_K = 60          # RRF combination constant

_INTENT_MAP = {
    "META_ANALYSIS": "META_ANALYSIS",
    "MODEL_VALIDATION": "CROSS_DATASET_COMPARISON",
    "PIPELINE_REUSE": "PIPELINE_REUSE",
    "CROSS_DATASET_COMPARISON": "CROSS_DATASET_COMPARISON",
    "REANALYSIS_FEASIBILITY": "REPLICATION",
    "METHOD_TRANSFER": "METHOD_TRANSFER",
    "REPLICATION": "REPLICATION",
    "EXPLORATION": "EXPLORATION",
}

_SPECIES_KEYWORDS = ["human", "mouse", "rat", "primate", "monkey", "macaque", "zebrafish", "drosophila"]
_MODALITY_KEYWORDS = ["fmri", "calcium", "ephys", "electrophysiology", "eeg", "meg", "neuropixels",
                       "two-photon", "widefield", "spike", "lfp", "patch", "miniscope"]
_TASK_KEYWORDS = ["working memory", "decision", "navigation", "reward", "learning", "fear", "attention",
                   "place", "sleep", "motor", "visual", "auditory", "somatosensory"]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


def _record_id(rec: dict) -> str:
    return f"{rec.get('source', 'unknown')}:{rec.get('source_id', 'unknown')}"


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


def build_bm25_index(
    records: list[dict],
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, int], float]:
    n = len(records)
    df: Counter[str] = Counter()
    doc_tf: dict[str, dict[str, float]] = {}
    doc_lengths: dict[str, int] = {}

    for rec in records:
        rec_id = _record_id(rec)
        tokens = tokenize(_record_text(rec))
        tf: Counter[str] = Counter(tokens)
        doc_tf[rec_id] = dict(tf)
        doc_lengths[rec_id] = len(tokens)
        for tok in set(tokens):
            df[tok] += 1

    idf: dict[str, float] = {
        tok: math.log((n - freq + 0.5) / (freq + 0.5) + 1)
        for tok, freq in df.items()
    }
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


def bm25_retrieve(
    query_text: str,
    records: list[dict],
    idf: dict[str, float],
    doc_tf: dict[str, dict[str, float]],
    doc_lengths: dict[str, int],
    avg_dl: float,
    top_k: int,
) -> list[tuple[str, float, dict]]:
    query_tokens = tokenize(query_text)
    scored: list[tuple[float, str, dict]] = []
    for rec in records:
        rec_id = _record_id(rec)
        score = bm25_score(query_tokens, rec_id, idf, doc_tf, doc_lengths, avg_dl)
        if score > 0:
            scored.append((score, rec_id, rec))
    scored.sort(reverse=True)
    return [(rid, s, r) for s, rid, r in scored[:top_k]]


# ---------------------------------------------------------------------------
# Dense retrieval (PRF with pre-computed BGE-large embeddings)
# ---------------------------------------------------------------------------

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=False))


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm < 1e-12:
        return vec
    return [x / norm for x in vec]


def _vec_add(acc: list[float], v: list[float]) -> list[float]:
    return [a + b for a, b in zip(acc, v, strict=True)]


def load_embeddings(emb_path: Path, field: str = TARGET_FIELD) -> dict[str, list[float]]:
    """Load normalized embeddings for a specific field. Key: 'source:source_id'."""
    emb: dict[str, list[float]] = {}
    with open(emb_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("field_name") != field:
                continue
            # record_id format: "dataset:dandi:000003" -> strip leading "dataset:"
            rid = rec["record_id"]
            if rid.startswith("dataset:"):
                rid = rid[len("dataset:"):]
            emb[rid] = rec["embedding"]
    return emb


def dense_retrieve_prf(
    bm25_top_results: list[tuple[str, float, dict]],
    emb_index: dict[str, list[float]],
    top_k: int,
    feedback_k: int = PRF_FEEDBACK_K,
) -> list[tuple[str, float]]:
    """Return top-k datasets by cosine similarity to the pseudo-query vector.

    Pseudo-query = normalized average of BM25 top-feedback_k embeddings.
    """
    if not emb_index or not bm25_top_results:
        return []

    dim = len(next(iter(emb_index.values())))
    pseudo_q: list[float] = [0.0] * dim
    found = 0
    for rid, _score, _rec in bm25_top_results[:feedback_k]:
        if rid in emb_index:
            pseudo_q = _vec_add(pseudo_q, emb_index[rid])
            found += 1

    if found == 0:
        return []

    pseudo_q = _normalize(pseudo_q)

    scored: list[tuple[float, str]] = []
    for rid, vec in emb_index.items():
        sim = _dot(pseudo_q, vec)
        scored.append((sim, rid))

    scored.sort(reverse=True)
    return [(rid, sim) for sim, rid in scored[:top_k]]


# ---------------------------------------------------------------------------
# Hybrid: Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def hybrid_rrf(
    bm25_ranked: list[tuple[str, float, dict]],
    dense_ranked: list[tuple[str, float]],
    corpus_by_id: dict[str, dict],
    top_k: int,
    k: int = RRF_K,
) -> list[tuple[str, float, dict]]:
    """Combine BM25 and dense rankings via RRF. Returns (dataset_id, rrf_score, record)."""
    scores: dict[str, float] = defaultdict(float)

    for rank, (rid, _s, _r) in enumerate(bm25_ranked, start=1):
        scores[rid] += 1.0 / (k + rank)

    for rank, (rid, _s) in enumerate(dense_ranked, start=1):
        scores[rid] += 1.0 / (k + rank)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for rid, rrf_score in ranked[:top_k]:
        rec = corpus_by_id.get(rid)
        if rec:
            results.append((rid, rrf_score, rec))
    return results


# ---------------------------------------------------------------------------
# Usefulness-scored retrieval
# ---------------------------------------------------------------------------

def _parse_query_context_fields(query_text: str) -> tuple[list[str], list[str], list[str]]:
    """Extract species, modalities, tasks from query text by keyword matching."""
    text = query_text.lower()
    species = [s for s in _SPECIES_KEYWORDS if s in text]
    modalities = [m for m in _MODALITY_KEYWORDS if m in text]
    tasks = [t for t in _TASK_KEYWORDS if t in text]
    return species, modalities, tasks


def _record_affordances(rec: dict) -> list[str]:
    aff = []
    if rec.get("has_behavior"):
        aff.append("has_behavior")
    if rec.get("has_trials"):
        aff.append("has_trials")
    if rec.get("has_raw_data"):
        aff.append("has_raw_data")
    if rec.get("has_processed_data"):
        aff.append("has_processed_data")
    return aff


def usefulness_retrieve(
    query: dict,
    bm25_top50: list[tuple[str, float, dict]],
    top_k: int,
) -> list[tuple[str, float, dict]]:
    """Score BM25 top-50 with structural usefulness scorer, return top-k."""
    try:
        from neural_search.retrieval.usefulness_scorer import (
            DatasetContext,
            UsefulnessIntent,
            score_usefulness,
        )
    except ImportError:
        return []

    query_text = query.get("query_text") or query.get("query") or ""
    intent_name = _INTENT_MAP.get(query.get("intent", ""), "EXPLORATION")
    try:
        intent = UsefulnessIntent[intent_name]
    except KeyError:
        intent = UsefulnessIntent.EXPLORATION

    q_species, q_modalities, q_tasks = _parse_query_context_fields(query_text)

    query_ctx = DatasetContext(
        dataset_id="__query__",
        modalities=q_modalities,
        tasks=q_tasks,
        species=q_species,
        brain_regions=[],
        affordances=[],
        data_standards=[],
    )

    scored: list[tuple[float, str, dict]] = []
    for rid, _s, rec in bm25_top50:
        candidate_ctx = DatasetContext(
            dataset_id=rid,
            modalities=list(rec.get("modalities") or []),
            tasks=list(rec.get("tasks") or []),
            species=list(rec.get("species") or []),
            brain_regions=list(rec.get("brain_regions") or []),
            affordances=_record_affordances(rec),
            data_standards=list(rec.get("data_standards") or []),
            has_timestamps=bool(rec.get("has_trials")),
        )
        try:
            result = score_usefulness(query_ctx, candidate_ctx, intent)
            scored.append((result.total_score, rid, rec))
        except Exception:
            scored.append((0.0, rid, rec))

    scored.sort(reverse=True)
    return [(rid, score, rec) for score, rid, rec in scored[:top_k]]


# ---------------------------------------------------------------------------
# Candidate builder
# ---------------------------------------------------------------------------

def _candidate_id(query_id: str, dataset_id: str) -> str:
    return f"qrels_candidate:{query_id}:{dataset_id}"


def _hard_negative_match(rec: dict, hard_negs: list[str]) -> bool:
    """Heuristic: check if dataset title/description matches any hard-negative pattern."""
    if not hard_negs:
        return False
    text = (_record_text(rec)).lower()
    for hn in hard_negs:
        if any(tok in text for tok in tokenize(hn) if len(tok) > 4):
            return True
    return False


def _affordance_matches(rec: dict, required_evidence: list[str]) -> list[str]:
    """Check which required_evidence fields are covered by the record's affordances."""
    matches = []
    aff_map = {
        "behavioral_metadata": rec.get("has_behavior"),
        "behavior": rec.get("has_behavior"),
        "trials": rec.get("has_trials"),
        "raw_data": rec.get("has_raw_data"),
        "processed_data": rec.get("has_processed_data"),
    }
    for evidence in required_evidence:
        key = evidence.lower().replace(" ", "_")
        if aff_map.get(key):
            matches.append(evidence)
    return matches


def build_pooled_candidate(
    query: dict,
    dataset_id: str,
    record: dict,
    systems_info: dict[str, Any],
) -> dict:
    query_id = query["query_id"]
    query_text = query.get("query_text") or query.get("query") or ""
    hard_negs: list[str] = query.get("known_failure_modes") or query.get("hard_negatives") or []
    required_evidence: list[str] = query.get("required_evidence") or query.get("must_have") or []

    retrieval_sources: list[str] = systems_info.get("sources", [])
    ranks_by_system: dict[str, int] = systems_info.get("ranks", {})
    usefulness_score: float = systems_info.get("usefulness_score", 0.0)
    best_score: float = systems_info.get("best_score", 0.0)

    aff_matches = _affordance_matches(record, required_evidence)
    is_hard_neg_warn = _hard_negative_match(record, hard_negs)

    return {
        "id": _candidate_id(query_id, dataset_id),
        "query_id": query_id,
        "query_text": query_text,
        "query_intent": query.get("intent", ""),
        "dataset_id": dataset_id,
        "dataset_title": record.get("title") or "",
        "dataset_source": record.get("source") or "",
        "dataset_source_url": record.get("url") or "",
        "dataset_description": (record.get("description") or "")[:800],
        "retrieval_sources": sorted(retrieval_sources),
        "ranks_by_system": ranks_by_system,
        "usefulness_score": round(usefulness_score, 4),
        "affordance_matches": aff_matches,
        "hard_negative_warning": is_hard_neg_warn,
        "retrieval_score": round(best_score, 4),
        "retrieval_method": "pooled",
        "metadata": {
            "pool": {
                "query_id": query_id,
                "record_id": dataset_id,
                "pooled_from": sorted(retrieval_sources),
                "n_systems": len(retrieval_sources),
                "status": "needs_annotation",
            },
            "query_known_failure_modes": hard_negs,
            "query_required_evidence": required_evidence,
            "record_species": record.get("species") or [],
            "record_modalities": record.get("modalities") or [],
            "record_tasks": record.get("tasks") or [],
            "record_brain_regions": record.get("brain_regions") or [],
            "record_data_standards": record.get("data_standards") or [],
        },
        "schema_version": SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# Main pooling logic
# ---------------------------------------------------------------------------

def _add_to_pool(
    pool: dict[str, dict],
    dataset_id: str,
    record: dict,
    system: str,
    rank: int,
    score: float,
    usefulness_score: float = 0.0,
) -> None:
    if dataset_id not in pool:
        pool[dataset_id] = {
            "record": record,
            "sources": [],
            "ranks": {},
            "best_score": 0.0,
            "usefulness_score": 0.0,
        }
    entry = pool[dataset_id]
    if system not in entry["sources"]:
        entry["sources"].append(system)
    entry["ranks"][system] = rank
    if score > entry["best_score"]:
        entry["best_score"] = score
    if usefulness_score > entry["usefulness_score"]:
        entry["usefulness_score"] = usefulness_score


def pool_for_query(
    query: dict,
    records: list[dict],
    corpus_by_id: dict[str, dict],
    idf: dict[str, float],
    doc_tf: dict[str, dict[str, float]],
    doc_lengths: dict[str, int],
    avg_dl: float,
    emb_index: dict[str, list[float]],
    top_k: int,
) -> list[dict]:
    query_text = query.get("query_text") or query.get("query") or ""

    # 1. BM25 top-k
    bm25_results = bm25_retrieve(query_text, records, idf, doc_tf, doc_lengths, avg_dl, top_k)

    # 2. Dense PRF top-k (needs BM25 top-5 as feedback)
    dense_results = dense_retrieve_prf(bm25_results, emb_index, top_k)

    # 3. Hybrid RRF
    hybrid_results = hybrid_rrf(bm25_results, dense_results, corpus_by_id, top_k)

    # 4. Usefulness-scored top-k (score BM25 top-50)
    bm25_top50 = bm25_retrieve(query_text, records, idf, doc_tf, doc_lengths, avg_dl, top_k * 3)
    usefulness_results = usefulness_retrieve(query, bm25_top50, top_k)

    # Merge into pool
    pool: dict[str, dict] = {}

    for rank, (rid, score, rec) in enumerate(bm25_results, start=1):
        _add_to_pool(pool, rid, rec, "bm25", rank, score)

    for rank, (rid, score) in enumerate(dense_results, start=1):
        rec = corpus_by_id.get(rid)
        if rec:
            _add_to_pool(pool, rid, rec, "dense_prf", rank, score)

    for rank, (rid, score, rec) in enumerate(hybrid_results, start=1):
        _add_to_pool(pool, rid, rec, "hybrid_rrf", rank, score)

    for rank, (rid, score, rec) in enumerate(usefulness_results, start=1):
        _add_to_pool(pool, rid, rec, "usefulness", rank, score, usefulness_score=score)

    # Build candidate records
    candidates = []
    for dataset_id, info in pool.items():
        candidate = build_pooled_candidate(
            query,
            dataset_id,
            info["record"],
            {
                "sources": info["sources"],
                "ranks": info["ranks"],
                "usefulness_score": info["usefulness_score"],
                "best_score": info["best_score"],
            },
        )
        candidates.append(candidate)

    # Sort: candidates surfaced by more systems first, then by best score
    candidates.sort(
        key=lambda c: (-len(c["retrieval_sources"]), -c["retrieval_score"])
    )
    return candidates


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-system pooled qrels candidates.")
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--embeddings", type=Path, default=EMBEDDINGS_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=20, metavar="K")
    parser.add_argument("--no-dense", action="store_true", help="Skip dense PRF retrieval")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path, name in [(args.queries, "queries"), (args.corpus, "corpus")]:
        if not path.exists():
            print(f"ERROR: {name} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading queries from {args.queries}...")
    queries = _load_jsonl(args.queries)
    print(f"  {len(queries)} queries")

    print(f"Loading corpus from {args.corpus}...")
    records = _load_jsonl(args.corpus)
    print(f"  {len(records)} records")

    corpus_by_id: dict[str, dict] = {_record_id(r): r for r in records}

    print("Building BM25 index...")
    idf, doc_tf, doc_lengths, avg_dl = build_bm25_index(records)
    print(f"  {len(idf)} unique tokens")

    emb_index: dict[str, list[float]] = {}
    if not args.no_dense and args.embeddings.exists():
        print(f"Loading {TARGET_FIELD} embeddings from {args.embeddings}...")
        emb_index = load_embeddings(args.embeddings, TARGET_FIELD)
        print(f"  {len(emb_index)} embedding records")
    else:
        if not args.no_dense:
            print("  Embeddings file not found, skipping dense PRF.")

    all_candidates: list[dict] = []
    stats: dict[str, Any] = {}

    for query in queries:
        query_id = query["query_id"]
        candidates = pool_for_query(
            query, records, corpus_by_id,
            idf, doc_tf, doc_lengths, avg_dl,
            emb_index, args.top_k,
        )
        n_by_system: dict[str, int] = defaultdict(int)
        for c in candidates:
            for s in c["retrieval_sources"]:
                n_by_system[s] += 1
        stats[query_id] = {
            "total": len(candidates),
            "by_system": dict(n_by_system),
            "multi_system": sum(1 for c in candidates if len(c["retrieval_sources"]) > 1),
        }
        all_candidates.extend(candidates)
        print(
            f"  {query_id}: {len(candidates)} candidates "
            f"({', '.join(f'{k}:{v}' for k,v in sorted(n_by_system.items()))})"
        )

    print(f"\nTotal candidates: {len(all_candidates)}")
    total_multi = sum(s["multi_system"] for s in stats.values())
    print(f"Multi-system candidates: {total_multi} ({100*total_multi/max(len(all_candidates),1):.0f}%)")

    if args.dry_run:
        print("DRY RUN — not writing output")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for c in all_candidates:
            f.write(json.dumps(c) + "\n")
    print(f"Wrote {len(all_candidates)} candidates to {args.output}")


if __name__ == "__main__":
    main()
