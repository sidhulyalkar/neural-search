#!/usr/bin/env python3
"""Run real BM25 and usefulness-reranked retrieval over benchmark queries.

Generates run files in JSONL format (one ranked result per line) that can be
pooled for human annotation and evaluated against adjudicated qrels.

Run files produced:
  - bm25.jsonl        BM25 sparse retrieval (field-weighted BM25 index)
  - usefulness.jsonl  BM25 candidates re-ranked by the usefulness scorer

Future variants (require dense index / GPU):
  - dense_bge.jsonl   BGE-large dense retrieval
  - bm25_bge_rrf.jsonl  RRF fusion of BM25 + dense
  - hybrid.jsonl      Full hybrid including ontology + graph

Usage:
    python scripts/eval/run_retrieval_baselines.py \
        --queries artifacts/benchmark_queries.jsonl \
        --corpus data/corpus/normalized/combined_corpus.jsonl \
        --out-dir reports/eval/runs/ \
        --top-k 100
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from neural_search.retrieval.dataset_context_bridge import dataset_context_from_record
from neural_search.retrieval.usefulness_scorer import DatasetContext, score_usefulness

# ---------------------------------------------------------------------------
# Imports — only use modules that don't require nbformat / GPU
# ---------------------------------------------------------------------------
from neural_search.search.sparse import SparseIndex

DEFAULT_CORPUS = Path("data/corpus/normalized/combined_corpus.jsonl")
DEFAULT_QUERIES = Path("artifacts/benchmark_queries.jsonl")
DEFAULT_OUT_DIR = Path("reports/eval/runs")
DEFAULT_TOP_K = 100


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def stable_record_id(record: dict[str, Any]) -> str:
    """Return a stable {source}:{source_id} compound ID for a corpus record."""
    source = str(record.get("source", "unknown"))
    source_id = str(
        record.get("source_id")
        or record.get("dataset_id")
        or record.get("id")
        or "unknown"
    )
    return f"{source}:{source_id}"


# ---------------------------------------------------------------------------
# Query context
# ---------------------------------------------------------------------------

_MODALITY_KEYWORDS: list[tuple[list[str], list[str]]] = [
    (["fmri", "bold", "mri"], ["fmri"]),
    (["calcium", "two-photon", "2p", "gcamp"], ["calcium_imaging", "two_photon"]),
    (["electrophysiology", "neuropixels", "ephys", "extracellular", "spike", "tetrode"],
     ["extracellular_electrophysiology"]),
    (["eeg"], ["eeg"]),
    (["fiber photometry", "photometry"], ["fiber_photometry"]),
    (["patch clamp", "whole.cell", "intracellular"], ["intracellular_electrophysiology"]),
    (["widefield", "wide.field"], ["widefield_imaging"]),
    (["meg"], ["meg"]),
    (["pet"], ["pet"]),
]

_TASK_KEYWORDS: list[tuple[list[str], list[str]]] = [
    (["reward", "reinforcement", "rl", "q-learning", "prediction error"],
     ["reinforcement_learning", "reward_learning"]),
    (["working memory", "n-back", "nback", "delayed response"],
     ["working_memory", "delayed_response"]),
    (["decision.making", "choice", "perceptual decision"],
     ["decision_making"]),
    (["visual stimul", "orientation", "grating", "natural image"],
     ["visual_stimulation", "orientation_tuning"]),
    (["spatial navigation", "maze", "place cell", "grid cell"],
     ["spatial_navigation", "maze_exploration"]),
    (["motor", "reaching", "movement", "locomotion"],
     ["motor_task"]),
    (["resting.state", "resting state"],
     ["resting_state"]),
    (["sleep", "slow oscillation", "spindle", "rem"],
     ["sleep"]),
]

_SPECIES_KEYWORDS: list[tuple[list[str], list[str]]] = [
    (["human", "participant", "subject", "patient", "healthy adult"],
     ["human"]),
    (["mouse", "mice", "mus musculus"],
     ["mouse"]),
    (["rat", "rattus"],
     ["rat"]),
    (["monkey", "primate", "macaque", "rhesus", "marmoset"],
     ["monkey", "rhesus_macaque"]),
    (["zebrafish"],
     ["zebrafish"]),
]


def _contains_any(text: str, keywords: list[str]) -> bool:
    for kw in keywords:
        if re.search(kw, text, re.IGNORECASE):
            return True
    return False


def query_to_context(query_text: str, query_id: str) -> DatasetContext:
    """Build a DatasetContext from query text via keyword matching."""
    modalities: list[str] = []
    tasks: list[str] = []
    species: list[str] = []

    for keywords, labels in _MODALITY_KEYWORDS:
        if _contains_any(query_text, keywords):
            modalities.extend(labels)
            break

    for keywords, labels in _TASK_KEYWORDS:
        if _contains_any(query_text, keywords):
            tasks.extend(labels)

    for keywords, labels in _SPECIES_KEYWORDS:
        if _contains_any(query_text, keywords):
            species.extend(labels)
            break

    return DatasetContext(
        dataset_id=f"query:{query_id}",
        modalities=modalities,
        tasks=tasks,
        species=species,
    )


# ---------------------------------------------------------------------------
# Run file I/O
# ---------------------------------------------------------------------------

def write_run(
    out_path: Path,
    query_id: str,
    ranked_results: list[tuple[str, float]],
) -> None:
    """Append ranked results for a query to a run JSONL file."""
    with out_path.open("a", encoding="utf-8") as handle:
        for rank, (record_id, score) in enumerate(ranked_results, start=1):
            handle.write(
                json.dumps({
                    "query_id": query_id,
                    "record_id": record_id,
                    "rank": rank,
                    "score": round(score, 6),
                }) + "\n"
            )


# ---------------------------------------------------------------------------
# BM25 retrieval
# ---------------------------------------------------------------------------

def run_bm25(
    index: SparseIndex,
    corpus_by_bm25_id: dict[str, dict],
    query_text: str,
    query_id: str,
    top_k: int,
    out_path: Path,
) -> list[tuple[str, float]]:
    candidates = index.search(query_text, top_k=top_k * 2)
    results: list[tuple[str, float]] = []
    seen: set[str] = set()
    for cand in candidates:
        record = corpus_by_bm25_id.get(cand.dataset_id)
        if record is None:
            continue
        rid = stable_record_id(record)
        if rid in seen:
            continue
        seen.add(rid)
        results.append((rid, cand.score))
        if len(results) >= top_k:
            break

    write_run(out_path, query_id, results)
    return results


# ---------------------------------------------------------------------------
# Usefulness reranking
# ---------------------------------------------------------------------------

def run_usefulness_rerank(
    bm25_results: list[tuple[str, float]],
    corpus_by_stable_id: dict[str, dict],
    query_text: str,
    query_id: str,
    intent_label: str,
    out_path: Path,
    top_k: int,
) -> None:
    from neural_search.retrieval.query_intent import UsefulnessIntent

    intent_map = {
        "META_ANALYSIS": UsefulnessIntent.META_ANALYSIS,
        "PIPELINE_REUSE": UsefulnessIntent.PIPELINE_REUSE,
        "REPLICATION": UsefulnessIntent.REPLICATION,
        "CROSS_DATASET_COMPARISON": UsefulnessIntent.CROSS_DATASET_COMPARISON,
        "METHOD_TRANSFER": UsefulnessIntent.METHOD_TRANSFER,
        "MODEL_VALIDATION": UsefulnessIntent.STRICT_LOOKUP,
        "REANALYSIS_FEASIBILITY": UsefulnessIntent.META_ANALYSIS,
        "EXPLORATION": UsefulnessIntent.EXPLORATION,
        "STRICT_LOOKUP": UsefulnessIntent.STRICT_LOOKUP,
    }
    intent = intent_map.get(intent_label.upper(), UsefulnessIntent.STRICT_LOOKUP)
    query_ctx = query_to_context(query_text, query_id)

    scored: list[tuple[str, float]] = []
    for record_id, _bm25_score in bm25_results:
        record = corpus_by_stable_id.get(record_id)
        if record is None:
            continue
        candidate_ctx = dataset_context_from_record(record)
        result = score_usefulness(query_ctx, candidate_ctx, intent)
        scored.append((record_id, result.total_score))

    scored.sort(key=lambda x: -x[1])
    write_run(out_path, query_id, scored[:top_k])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BM25 and usefulness retrieval baselines.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--variants", nargs="+", default=["bm25", "usefulness"],
                        choices=["bm25", "usefulness"])
    args = parser.parse_args(argv)

    if not args.queries.exists():
        raise SystemExit(f"Benchmark queries not found: {args.queries}")
    if not args.corpus.exists():
        raise SystemExit(f"Corpus not found: {args.corpus}")

    # Load queries
    queries: list[dict] = []
    with args.queries.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    print(f"Loaded {len(queries)} benchmark queries.")

    # Load corpus
    print(f"Loading corpus from {args.corpus} ...", flush=True)
    t0 = time.time()
    corpus: list[dict] = []
    with args.corpus.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus.append(json.loads(line))
    print(f"  {len(corpus)} records in {time.time() - t0:.1f}s")

    # Build lookup maps
    corpus_by_stable_id: dict[str, dict] = {stable_record_id(r): r for r in corpus}

    # Build BM25 index (maps source_id → record)
    corpus_by_bm25_id: dict[str, dict] = {}
    for record in corpus:
        bm25_id = str(
            record.get("dataset_id") or record.get("id") or record.get("source_id") or ""
        )
        if bm25_id:
            corpus_by_bm25_id[bm25_id] = record

    # Prepare output files (truncate existing)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variant_paths: dict[str, Path] = {}
    for variant in args.variants:
        path = args.out_dir / f"{variant}.jsonl"
        path.write_text("", encoding="utf-8")
        variant_paths[variant] = path

    # Build BM25 index if needed
    index: SparseIndex | None = None
    if "bm25" in args.variants or "usefulness" in args.variants:
        print("Building BM25 index ...", flush=True)
        t0 = time.time()
        index = SparseIndex()
        index.build(corpus)
        print(f"  BM25 index built in {time.time() - t0:.1f}s")

    # Run per-query
    run_stats: list[dict] = []
    for qi, query in enumerate(queries, start=1):
        qid = str(query["query_id"])
        qtext = str(query["query"])
        intent = str(query.get("intent", "STRICT_LOOKUP"))
        print(f"[{qi}/{len(queries)}] {qid}: {qtext[:70]}", flush=True)

        bm25_results: list[tuple[str, float]] = []

        if "bm25" in args.variants and index is not None:
            t0 = time.time()
            bm25_results = run_bm25(
                index, corpus_by_bm25_id, qtext, qid, args.top_k,
                variant_paths["bm25"]
            )
            print(f"  BM25: {len(bm25_results)} results in {time.time()-t0:.2f}s")

        if "usefulness" in args.variants:
            if not bm25_results and index is not None:
                bm25_results = run_bm25(
                    index, corpus_by_bm25_id, qtext, qid, args.top_k,
                    Path("/dev/null")
                )
            t0 = time.time()
            run_usefulness_rerank(
                bm25_results, corpus_by_stable_id, qtext, qid, intent,
                variant_paths["usefulness"], args.top_k,
            )
            print(f"  Usefulness rerank: {len(bm25_results)} candidates in {time.time()-t0:.2f}s")

        run_stats.append({
            "query_id": qid,
            "query": qtext,
            "intent": intent,
            "bm25_results": len(bm25_results),
        })

    # Write status report
    report = {
        "status": "completed",
        "query_count": len(queries),
        "corpus_size": len(corpus),
        "top_k": args.top_k,
        "variants": args.variants,
        "per_query": run_stats,
    }
    status_path = args.out_dir.parent / "retrieval_baselines_status.json"
    status_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "queries": len(queries), "variants": args.variants}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
