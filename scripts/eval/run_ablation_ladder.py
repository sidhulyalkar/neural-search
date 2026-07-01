#!/usr/bin/env python3
"""Additive ablation ladder for neural-search retrieval evaluation.

Runs eight retrieval systems in progressive order, writing a TREC-style JSONL
run file for each rung. Each rung adds one capability on top of the previous:

  rung 1 — bm25                BM25 sparse retrieval
  rung 2 — bm25_structured     BM25 + usefulness slot matching
  rung 3 — dense_bge           BGE-large dense retrieval (pre-computed embeddings)
  rung 4 — hybrid_rrf          RRF(BM25 + BGE-dense)
  rung 5 — hybrid_graph        hybrid_rrf + knowledge graph context score
  rung 6 — typed_kg            hybrid_rrf + typed finding-relationship score ONLY
                                (isolates the Phase 0-6b supports/contradicts layer
                                 from the aggregate graph signal in rung 5 — added
                                 2026-06-23 because hybrid_graph mixes linked-paper
                                 counts, affordances, and dataset-similarity edges
                                 together with the typed layer, so a gain or loss
                                 there couldn't be attributed to the typed layer)
  rung 7 — typed_kg_qualified  typed_kg + qualified-consensus region bonus
  rung 8 — full                hybrid_graph + source diversity reranking

Use --skip-rungs to exclude slow rungs (dense_bge requires sentence-transformers).

Outputs
-------
  reports/eval/runs/bm25.jsonl
  reports/eval/runs/bm25_structured.jsonl
  reports/eval/runs/dense_bge.jsonl
  reports/eval/runs/hybrid_rrf.jsonl
  reports/eval/runs/hybrid_graph.jsonl
  reports/eval/runs/typed_kg.jsonl
  reports/eval/runs/typed_kg_qualified.jsonl
  reports/eval/runs/full.jsonl
  reports/eval/ablation_ladder_report.json
  reports/eval/ablation_ladder_report.md

Usage
-----
    # Run all 6 rungs (requires sentence-transformers for dense_bge)
    python scripts/eval/run_ablation_ladder.py

    # Fast run — BM25-only rungs, skip BGE-dependent ones
    python scripts/eval/run_ablation_ladder.py --skip-rungs dense_bge hybrid_rrf hybrid_graph full

    # Custom query set
    python scripts/eval/run_ablation_ladder.py --queries data/eval/benchmark_queries_canonical.yaml

NOTE: Requires anaconda Python for dense rungs:
    /home/sid21/anaconda3/bin/python3 scripts/eval/run_ablation_ladder.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from neural_search.graph.typed_kg_features import TypedKGIndex, typed_kg_score
from neural_search.retrieval.dataset_context_bridge import dataset_context_from_record
from neural_search.retrieval.usefulness_scorer import DatasetContext, score_usefulness
from neural_search.search.sparse import SparseIndex

DEFAULT_CORPUS = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")
DEFAULT_QUERIES = Path("data/eval/benchmark_queries_canonical.yaml")
DEFAULT_EMBEDDINGS = Path("data/embeddings/real_all.dense.field_embeddings.jsonl")
DEFAULT_GRAPH = Path("data/graph/neural_search_graph.real_corpus.json")
DEFAULT_PAPER_LINKS = Path("artifacts/literature/paper_dataset_links.jsonl")
DEFAULT_FINDING_EDGES = Path("artifacts/literature/relationships/finding_edges.jsonl")
DEFAULT_QUALIFIED_CONSENSUS = Path("artifacts/literature/relationships/consensus_summaries_qualified.jsonl")
DEFAULT_OUT_DIR = Path("reports/eval/runs")
# Cross-platform discard sink for cache-miss recomputation that shouldn't be
# written to a run file (the original /dev/null literal only worked on POSIX).
NULL_PATH = Path(os.devnull)
DEFAULT_TOP_K = 100
RRF_K = 60
GRAPH_SCORE_WEIGHT = 0.05
TYPED_KG_SCORE_WEIGHT = 0.005  # same magnitude as GRAPH_SCORE_WEIGHT — fair comparison

ALL_RUNGS = [
    "bm25",
    "bm25_structured",
    "dense_bge",
    "hybrid_rrf",
    "hybrid_graph",
    "typed_kg",
    "typed_kg_qualified",
    "full",
]

FIELD_WEIGHTS: dict[str, float] = {
    "title": 2.0,
    "combined_scientific_summary": 2.0,
    "description": 1.0,
    "modalities": 1.0,
    "species": 1.0,
    "tasks": 1.0,
    "brain_regions": 1.0,
    "behaviors": 0.5,
    "data_standards": 0.5,
}

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

RunResult = tuple[str, float]  # (record_id, score)

_MODALITY_KW: list[tuple[list[str], list[str]]] = [
    (["fmri", "bold", "mri"], ["fmri"]),
    (["calcium", "two-photon", "2p", "gcamp"], ["calcium_imaging"]),
    (["electrophysiology", "neuropixels", "ephys", "extracellular", "spike"], ["extracellular_ephys"]),
    (["eeg"], ["eeg"]),
    (["fiber photometry", "photometry"], ["fiber_photometry"]),
    (["patch clamp", "whole.cell", "intracellular"], ["intracellular_ephys"]),
    (["widefield"], ["widefield_imaging"]),
    (["meg"], ["meg"]),
]

_TASK_KW: list[tuple[list[str], list[str]]] = [
    (["reward", "reinforcement", "q-learning"], ["reinforcement_learning"]),
    (["working memory", "n-back", "delayed response"], ["working_memory"]),
    (["decision.making", "choice", "perceptual decision"], ["decision_making"]),
    (["visual stim", "orientation", "grating"], ["visual_stimulation"]),
    (["navigation", "maze", "place cell"], ["spatial_navigation"]),
    (["motor", "reaching", "locomotion"], ["motor_task"]),
    (["sleep"], ["sleep"]),
]

_SPECIES_KW: list[tuple[list[str], list[str]]] = [
    (["human", "participant", "patient"], ["human"]),
    (["mouse", "mice", "mus musculus"], ["mouse"]),
    (["rat", "rattus"], ["rat"]),
    (["monkey", "macaque", "primate"], ["monkey"]),
    (["zebrafish"], ["zebrafish"]),
]


def _contains_any(text: str, kws: list[str]) -> bool:
    return any(re.search(kw, text, re.IGNORECASE) for kw in kws)


def _query_context(query_text: str, query_id: str) -> DatasetContext:
    """Extract structured DatasetContext from query text via keyword matching."""
    modalities: list[str] = []
    tasks: list[str] = []
    species: list[str] = []
    for kws, labels in _MODALITY_KW:
        if _contains_any(query_text, kws):
            modalities.extend(labels)
            break
    for kws, labels in _TASK_KW:
        if _contains_any(query_text, kws):
            tasks.extend(labels)
    for kws, labels in _SPECIES_KW:
        if _contains_any(query_text, kws):
            species.extend(labels)
            break
    return DatasetContext(dataset_id=f"query:{query_id}", modalities=modalities, tasks=tasks, species=species)


def _graph_context_dict(q: dict[str, Any]) -> dict[str, Any]:
    """Build query context dict for graph_context_score and concept_overlap_score."""
    text = str(q.get("query", ""))
    ctx = _query_context(text, str(q.get("id", "")))

    # Concept slugs from all query constraint fields for concept_overlap_score
    def _slugify(terms: list[str]) -> list[str]:
        return [t.casefold().replace(" ", "_").replace("-", "_") for t in terms if t]

    concepts: list[str] = []
    concepts.extend(_slugify(list(q.get("expected_modalities_any", []) or [])))
    concepts.extend(_slugify(list(q.get("expected_tasks", []) or [])))
    concepts.extend(_slugify(list(q.get("expected_behaviors", []) or [])))
    concepts.extend(_slugify(list(q.get("expected_regions_any", []) or [])))
    concepts.extend(_slugify(list(q.get("expected_species", []) or [])))
    # Deduplicate
    seen: set[str] = set()
    deduped_concepts = [c for c in concepts if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

    return {
        "tasks": ctx.tasks,
        "modalities": ctx.modalities,
        "species": ctx.species,
        "regions": list(q.get("expected_regions_any", []) or []),
        "analysis": list(q.get("expected_analysis_any", []) or []),
        "concepts": deduped_concepts,
    }


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _stable_id(record: dict[str, Any]) -> str:
    source = str(record.get("source", "unknown"))
    sid = str(record.get("source_id") or record.get("dataset_id") or record.get("id") or "unknown")
    return f"{source}:{sid}"


def _bm25_id(record: dict[str, Any]) -> str:
    return str(record.get("dataset_id") or record.get("id") or record.get("source_id") or "")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_queries(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], list) and data[key] and "query" in data[key][0]:
                return data[key]
    if isinstance(data, list):
        return data
    return []


def _load_corpus(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_run(out_path: Path, query_id: str, results: list[RunResult], rung: str) -> None:
    with out_path.open("a", encoding="utf-8") as f:
        for rank, (record_id, score) in enumerate(results, start=1):
            f.write(json.dumps({
                "query_id": query_id,
                "record_id": record_id,
                "rank": rank,
                "score": round(score, 6),
                "rung": rung,
            }) + "\n")


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def load_field_embeddings(path: Path) -> dict[str, list[float]]:
    """Load pre-computed BGE field embeddings and aggregate per record.

    Load priority (fastest to slowest):
    1. Pre-aggregated .npy + .ids.json cache (< 100ms)
    2. FAISS binary + meta.jsonl sidecar (covers full corpus even if JSONL truncated)
    3. Raw JSONL parsing (slow, may be partial)

    Returns dict mapping record_id → weighted-average normalized embedding.
    """
    import numpy as np

    print(f"Loading field embeddings from {path} ...", flush=True)
    t0 = time.time()

    # --- Fastest path: pre-aggregated .npy cache ---------------------------
    ids_path = path.with_name(path.stem + ".agg.ids.json")
    mat_path = path.with_name(path.stem + ".agg.mat.npy")
    if ids_path.exists() and mat_path.exists():
        try:
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
            mat = np.load(str(mat_path), mmap_mode="r")
            aggregated = {rid: mat[i].tolist() for i, rid in enumerate(ids)}
            print(f"  {len(aggregated)} record embeddings via .npy cache in {time.time() - t0:.2f}s")
            return aggregated
        except Exception as exc:
            print(f"  .npy cache failed ({exc}), trying FAISS ...")

    # --- Fast path: FAISS binary + meta.jsonl sidecar ----------------------
    faiss_path = path.with_suffix(".faiss")
    meta_path = path.with_name(path.stem + ".meta.jsonl")
    if faiss_path.exists() and meta_path.exists():
        try:
            from neural_search.embeddings.field_index import read_field_embedding_cache_faiss
            records = read_field_embedding_cache_faiss(faiss_path, meta_path)
            sums: dict[str, np.ndarray] = {}
            weights: dict[str, float] = {}
            for rec in records:
                rid = str(rec.record_id).removeprefix("dataset:")
                field = rec.field_name
                emb = np.array(rec.embedding, dtype=np.float32)
                w = FIELD_WEIGHTS.get(field, 0.5)
                if rid not in sums:
                    sums[rid] = np.zeros(len(emb), dtype=np.float32)
                    weights[rid] = 0.0
                sums[rid] += emb * w
                weights[rid] += w
            aggregated: dict[str, list[float]] = {}
            for rid, vec in sums.items():
                w = weights[rid]
                if w > 0:
                    v = vec / w
                    norm = float(np.linalg.norm(v))
                    if norm > 0:
                        v = v / norm
                    aggregated[rid] = v.tolist()
            print(f"  {len(aggregated)} record embeddings via FAISS in {time.time() - t0:.1f}s")
            return aggregated
        except Exception as exc:
            print(f"  FAISS fast-path failed ({exc}), falling back to JSONL ...")

    # --- Slow path: parse JSONL ---------------------------------------------
    sums2: dict[str, np.ndarray] = {}
    weights2: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # Strip 'dataset:' prefix so IDs match BM25 stable IDs (source:source_id)
        rid = str(rec["record_id"]).removeprefix("dataset:")
        field = str(rec.get("field_name", ""))
        emb = np.array(rec["embedding"], dtype=np.float32)
        w = FIELD_WEIGHTS.get(field, 0.5)
        if rid not in sums2:
            sums2[rid] = np.zeros(len(emb), dtype=np.float32)
            weights2[rid] = 0.0
        sums2[rid] += emb * w
        weights2[rid] += w

    aggregated2: dict[str, list[float]] = {}
    for rid, vec in sums2.items():
        w = weights2[rid]
        if w > 0:
            v = vec / w
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm
            aggregated2[rid] = v.tolist()

    print(f"  {len(aggregated2)} record embeddings via JSONL in {time.time() - t0:.1f}s")
    return aggregated2


def load_bge_encoder() -> Any:
    """Load BGE-large sentence-transformers model."""
    try:
        from neural_search.embeddings.dense_provider import DenseEmbeddingProvider
        print("Loading BGE-large model (sentence-transformers) ...", flush=True)
        t0 = time.time()
        enc = DenseEmbeddingProvider(device="cpu")
        print(f"  Model loaded in {time.time() - t0:.1f}s")
        return enc
    except Exception as exc:
        raise RuntimeError(f"Cannot load BGE model: {exc}") from exc


def encode_query_bge(encoder: Any, query_text: str) -> list[float]:
    import numpy as np
    vec = np.array(encoder.embed_text(query_text), dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


_DENSE_MAT_CACHE: tuple[int, list[str], Any] | None = None  # (dict_id, ids, mat)


def dense_retrieve(
    query_vec: list[float],
    embeddings: dict[str, list[float]],
    top_k: int,
) -> list[RunResult]:
    """Brute-force cosine similarity retrieval over pre-computed embeddings.

    Caches the numpy matrix across calls so repeated queries against the same
    embeddings dict don't rebuild the matrix each time.
    """
    import numpy as np
    global _DENSE_MAT_CACHE
    qv = np.array(query_vec, dtype=np.float32)
    dict_id = id(embeddings)
    if _DENSE_MAT_CACHE is not None and _DENSE_MAT_CACHE[0] == dict_id:
        ids, mat = _DENSE_MAT_CACHE[1], _DENSE_MAT_CACHE[2]
    else:
        ids = list(embeddings.keys())
        mat = np.array([embeddings[rid] for rid in ids], dtype=np.float32)
        _DENSE_MAT_CACHE = (dict_id, ids, mat)
    scores = mat @ qv
    top_indices = scores.argsort()[::-1][:top_k]
    return [(ids[i], float(scores[i])) for i in top_indices]


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def rrf_fuse(
    *ranked_lists: list[RunResult],
    k: int = RRF_K,
) -> list[RunResult]:
    """Reciprocal Rank Fusion over multiple ranked lists."""
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (record_id, _) in enumerate(ranked, start=1):
            scores[record_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])


# ---------------------------------------------------------------------------
# Retrieval rungs
# ---------------------------------------------------------------------------

def rung_bm25(
    index: SparseIndex,
    corpus_by_bm25_id: dict[str, dict],
    corpus_by_stable_id: dict[str, dict],
    query_text: str,
    query_id: str,
    top_k: int,
    out_path: Path,
) -> list[RunResult]:
    candidates = index.search(query_text, top_k=top_k * 2)
    results: list[RunResult] = []
    seen: set[str] = set()
    for cand in candidates:
        record = corpus_by_bm25_id.get(cand.dataset_id)
        if record is None:
            continue
        rid = _stable_id(record)
        if rid in seen:
            continue
        seen.add(rid)
        results.append((rid, cand.score))
        if len(results) >= top_k:
            break
    _write_run(out_path, query_id, results, "bm25")
    return results


def rung_bm25_structured(
    bm25_results: list[RunResult],
    corpus_by_stable_id: dict[str, dict],
    query_text: str,
    query_id: str,
    intent: str,
    top_k: int,
    out_path: Path,
) -> list[RunResult]:
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
    us_intent = intent_map.get(intent.upper(), UsefulnessIntent.STRICT_LOOKUP)
    q_ctx = _query_context(query_text, query_id)

    scored: list[RunResult] = []
    for record_id, _ in bm25_results:
        record = corpus_by_stable_id.get(record_id)
        if record is None:
            continue
        cand_ctx = dataset_context_from_record(record)
        result = score_usefulness(q_ctx, cand_ctx, us_intent)
        scored.append((record_id, result.total_score))
    scored.sort(key=lambda x: -x[1])
    results = scored[:top_k]
    _write_run(out_path, query_id, results, "bm25_structured")
    return results


def rung_dense_bge(
    encoder: Any,
    embeddings: dict[str, list[float]],
    query_text: str,
    query_id: str,
    top_k: int,
    out_path: Path,
) -> list[RunResult]:
    qvec = encode_query_bge(encoder, query_text)
    results = dense_retrieve(qvec, embeddings, top_k)
    _write_run(out_path, query_id, results, "dense_bge")
    return results


def rung_hybrid_rrf(
    bm25_results: list[RunResult],
    bge_results: list[RunResult],
    query_id: str,
    top_k: int,
    out_path: Path,
) -> list[RunResult]:
    fused = rrf_fuse(bm25_results, bge_results)[:top_k]
    _write_run(out_path, query_id, fused, "hybrid_rrf")
    return fused


def rung_hybrid_graph(
    rrf_results: list[RunResult],
    graph: Any,
    query: dict[str, Any],
    query_id: str,
    top_k: int,
    out_path: Path,
    graph_score_weight: float = GRAPH_SCORE_WEIGHT,
    concept_index: dict[str, list[str]] | None = None,
) -> list[RunResult]:
    from neural_search.graph.search_features import (
        concept_overlap_score,
        graph_context_score,
        region_hierarchy_score,
    )
    from neural_search.graph.search_features import _get_alias_index, _neighbor_labels

    q_ctx = _graph_context_dict(query)
    query_regions: list[str] = q_ctx.get("regions", [])

    alias_index = _get_alias_index(graph) if graph is not None else {}

    rescored: list[RunResult] = []
    for record_id, base_score in rrf_results:
        gscore = graph_context_score(graph, record_id, query_context=q_ctx)
        cscore = concept_overlap_score(record_id, q_ctx, concept_index) if concept_index else 0.0
        rh_score = 0.0
        if query_regions and graph is not None:
            node_id = alias_index.get(record_id, f"node:dataset:{record_id}")
            dataset_regions = _neighbor_labels(graph, node_id, "dataset_records_region")
            rh_score = region_hierarchy_score(dataset_regions, query_regions)
        total_graph = graph_score_weight * (gscore + cscore + rh_score)
        rescored.append((record_id, base_score + total_graph))
    rescored.sort(key=lambda x: -x[1])
    results = rescored[:top_k]
    _write_run(out_path, query_id, results, "hybrid_graph")
    return results


def rung_typed_kg(
    rrf_results: list[RunResult],
    typed_index: TypedKGIndex,
    corpus_by_stable_id: dict[str, dict],
    query_id: str,
    top_k: int,
    out_path: Path,
    *,
    qualified: bool,
    score_weight: float = TYPED_KG_SCORE_WEIGHT,
) -> list[RunResult]:
    """hybrid_rrf + typed finding-relationship score ONLY (no graph_context_score).

    Isolates the Phase 0-6b supports/contradicts/qualified-consensus layer from
    the aggregate hybrid_graph signal, so its retrieval contribution can be
    measured on its own against the same canonical qrels.
    """
    rung_name = "typed_kg_qualified" if qualified else "typed_kg"
    rescored: list[RunResult] = []
    for record_id, base_score in rrf_results:
        record = corpus_by_stable_id.get(record_id)
        tscore = typed_kg_score(record_id, typed_index, record=record, qualified=qualified)
        rescored.append((record_id, base_score + (score_weight * tscore)))
    rescored.sort(key=lambda x: -x[1])
    results = rescored[:top_k]
    _write_run(out_path, query_id, results, rung_name)
    return results


def rung_full(
    graph_results: list[RunResult],
    corpus_by_stable_id: dict[str, dict],
    query_id: str,
    top_k: int,
    out_path: Path,
    max_per_source: int = 25,
) -> list[RunResult]:
    """Add source diversity: cap each source at max_per_source results."""
    source_counts: dict[str, int] = defaultdict(int)
    results: list[RunResult] = []
    for record_id, score in graph_results:
        source = record_id.split(":")[0] if ":" in record_id else "unknown"
        if source_counts[source] >= max_per_source:
            continue
        source_counts[source] += 1
        results.append((record_id, score))
        if len(results) >= top_k:
            break
    _write_run(out_path, query_id, results, "full")
    return results


# ---------------------------------------------------------------------------
# Metrics against auto-labels
# ---------------------------------------------------------------------------

def _normalize_record_id(rid: str) -> str:
    """Strip 'dataset:' prefix to match run file record IDs."""
    return rid.removeprefix("dataset:")


def _load_auto_labels(path: Path) -> dict[tuple[str, str], str]:
    """Load annotation_candidates.jsonl → {(query_id, normalized_candidate_id): label}.

    Also returns a separate query-text→query_id mapping so canonical queries
    can be matched to label query IDs by text similarity.
    """
    if not path.exists():
        return {}
    labels: dict[tuple[str, str], str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = str(rec["query_id"])
        cid = _normalize_record_id(str(rec["candidate_id"]))
        labels[(qid, cid)] = str(rec.get("usefulness_label", ""))
    return labels


def _ndcg_at_k(results: list[RunResult], labels: dict[tuple[str, str], str], query_id: str, k: int) -> float:
    """Compute NDCG@k. Labels: highly_useful=2, useful=1, not_useful=0."""
    grade = {"highly_useful": 2, "useful": 1, "not_useful": 0}

    def dcg(gains: list[float]) -> float:
        return sum(g / (i + 1) for i, g in enumerate(gains[:k]))  # log2 approximation

    actual = [grade.get(labels.get((query_id, rid), ""), 0) for rid, _ in results[:k]]
    ideal = sorted(actual, reverse=True)
    idcg = dcg(ideal)  # type: ignore[arg-type]
    if idcg == 0:
        return 0.0
    return dcg(actual) / idcg  # type: ignore[arg-type]


def compute_metrics(
    run_file: Path,
    auto_labels: dict[tuple[str, str], str],
    k: int = 10,
) -> dict[str, float]:
    if not run_file.exists() or not auto_labels:
        return {}
    results_by_query: dict[str, list[RunResult]] = defaultdict(list)
    for line in run_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        results_by_query[rec["query_id"]].append((rec["record_id"], rec["score"]))

    ndcgs: list[float] = []
    for qid, results in results_by_query.items():
        ndcg = _ndcg_at_k(results, auto_labels, qid, k)
        ndcgs.append(ndcg)
    if not ndcgs:
        return {}
    return {f"ndcg@{k}": round(sum(ndcgs) / len(ndcgs), 4), "query_count": len(ndcgs)}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _write_report(out_dir: Path, rows: list[dict[str, Any]], *, partial: bool = False) -> None:
    report = {"rungs": rows}
    stem = "ablation_ladder_report.partial" if partial else "ablation_ladder_report"
    json_path = out_dir.parent / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = ["# Ablation Ladder Report\n", "| Rung | Queries | NDCG@10 |", "|------|---------|---------|"]
    for row in rows:
        m = row.get("metrics", {})
        ndcg = f"{m.get('ndcg@10', 'N/A')}"
        qc = m.get("query_count", row.get("queries_run", "?"))
        status = row.get("status", "ok")
        label = f"{row['rung']} ({'skipped' if status == 'skipped' else 'ok'})"
        md_lines.append(f"| {label} | {qc} | {ndcg} |")
    md_path = out_dir.parent / f"{stem}.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nReport: {json_path}")


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 50)
    print(f"{'Rung':<25} {'Queries':>8} {'NDCG@10':>10}")
    print("-" * 50)
    for row in rows:
        m = row.get("metrics", {})
        ndcg = m.get("ndcg@10", "-")
        qc = m.get("query_count", row.get("queries_run", "-"))
        status = "" if row.get("status") != "skipped" else " (skipped)"
        print(f"{row['rung'] + status:<25} {str(qc):>8} {str(ndcg):>10}")
    print("=" * 50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Additive ablation ladder for neural-search retrieval.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--paper-links", type=Path, default=DEFAULT_PAPER_LINKS)
    parser.add_argument("--finding-edges", type=Path, default=DEFAULT_FINDING_EDGES)
    parser.add_argument("--qualified-consensus", type=Path, default=DEFAULT_QUALIFIED_CONSENSUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--graph-score-weight",
        type=float,
        default=GRAPH_SCORE_WEIGHT,
        help="Global multiplier for graph_context_score in hybrid_graph/full rungs.",
    )
    parser.add_argument(
        "--typed-kg-score-weight",
        type=float,
        default=TYPED_KG_SCORE_WEIGHT,
        help="Global multiplier for typed_kg_score in typed_kg/typed_kg_qualified rungs.",
    )
    parser.add_argument(
        "--skip-rungs", nargs="+", default=[],
        choices=ALL_RUNGS, metavar="RUNG",
        help="Rungs to skip (e.g. --skip-rungs dense_bge hybrid_rrf)",
    )
    parser.add_argument(
        "--rungs", nargs="+", default=None,
        choices=ALL_RUNGS, metavar="RUNG",
        help="Only run these rungs (overrides --skip-rungs)",
    )
    args = parser.parse_args(argv)

    active_rungs = set(args.rungs or ALL_RUNGS) - set(args.skip_rungs)
    is_partial_run = active_rungs != set(ALL_RUNGS)
    needs_dense = bool(
        active_rungs
        & {"dense_bge", "hybrid_rrf", "hybrid_graph", "typed_kg", "typed_kg_qualified", "full"}
    )

    if not args.queries.exists():
        raise SystemExit(f"Query file not found: {args.queries}. Run merge_benchmark_queries.py first.")
    if not args.corpus.exists():
        raise SystemExit(f"Corpus not found: {args.corpus}")

    # -- Load resources ---------------------------------------------------
    print(f"Loading queries from {args.queries} ...", flush=True)
    queries = _load_queries(args.queries)
    print(f"  {len(queries)} queries")

    print("Loading corpus ...", flush=True)
    t0 = time.time()
    corpus = _load_corpus(args.corpus)
    print(f"  {len(corpus)} records in {time.time() - t0:.1f}s")

    corpus_by_stable_id: dict[str, dict] = {_stable_id(r): r for r in corpus}
    corpus_by_bm25_id: dict[str, dict] = {}
    for r in corpus:
        bid = _bm25_id(r)
        if bid:
            corpus_by_bm25_id[bid] = r

    print("Building BM25 index ...", flush=True)
    t0 = time.time()
    index = SparseIndex()
    _bm25_cache_path = DEFAULT_CORPUS.parent / (DEFAULT_CORPUS.name + ".bm25.pkl")
    _bm25_loaded = False
    if _bm25_cache_path.exists():
        corpus_mtime = DEFAULT_CORPUS.stat().st_mtime
        cache_mtime = _bm25_cache_path.stat().st_mtime
        if cache_mtime > corpus_mtime:
            try:
                import pickle
                with _bm25_cache_path.open("rb") as _pf:
                    index = pickle.load(_pf)
                print(f"  Loaded cached BM25 index in {time.time() - t0:.1f}s")
                _bm25_loaded = True
            except Exception as _exc:
                print(f"  BM25 cache load failed ({_exc}), rebuilding ...")
    if not _bm25_loaded:
        index.build(corpus)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")
        try:
            import pickle
            with _bm25_cache_path.open("wb") as _pf:
                pickle.dump(index, _pf, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  Saved BM25 cache to {_bm25_cache_path.name}")
        except Exception as _exc:
            print(f"  BM25 cache save failed ({_exc})")

    embeddings: dict[str, list[float]] = {}
    encoder: Any = None
    if needs_dense:
        if not args.embeddings.exists():
            print(f"  Warning: embeddings not found at {args.embeddings} -- dense rungs will be skipped.")
            active_rungs -= {
                "dense_bge", "hybrid_rrf", "hybrid_graph", "typed_kg", "typed_kg_qualified", "full",
            }
        else:
            embeddings = load_field_embeddings(args.embeddings)
            encoder = load_bge_encoder()

    graph: Any = None
    concept_index: dict[str, list[str]] | None = None
    if active_rungs & {"hybrid_graph", "full"}:
        try:
            from neural_search.graph.search_features import load_graph_if_exists
            graph = load_graph_if_exists(args.graph)
            if graph:
                node_count = len(graph.nodes) if hasattr(graph, "nodes") else "?"
                print(f"  Graph loaded: {node_count} nodes")
            else:
                print(f"  Graph not found at {args.graph} -- hybrid_graph/full will run without graph signal")
        except Exception as exc:
            print(f"  Warning: graph load failed: {exc}")
        try:
            from neural_search.ingestion.corpus_kg_linker import load_dataset_concept_index
            concept_index_path = Path("data/kg/dataset_concept_index.jsonl")
            if concept_index_path.exists():
                concept_index = load_dataset_concept_index(concept_index_path)
                print(f"  Concept index loaded: {len(concept_index)} datasets")
            else:
                print("  Concept index not found -- concept_overlap_score disabled")
        except Exception as exc:
            print(f"  Warning: concept index load failed: {exc}")

    typed_index: TypedKGIndex | None = None
    if active_rungs & {"typed_kg", "typed_kg_qualified"}:
        typed_index = TypedKGIndex.from_files(
            args.paper_links, args.finding_edges, args.qualified_consensus
        )
        n_linked = len(typed_index.dataset_to_papers)
        n_qualified = len(typed_index.region_to_qualified_consensus)
        print(
            f"  Typed KG index: {n_linked} datasets with confidently-linked papers, "
            f"{n_qualified} regions with qualified consensus (n_papers>=2)"
        )

    auto_labels = _load_auto_labels(Path("data/eval/annotation_candidates.jsonl"))
    print(f"  Auto-labels loaded: {len(auto_labels)} pairs")

    # -- Run per rung ------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_paths: dict[str, Path] = {}
    for rung in ALL_RUNGS:
        p = args.out_dir / f"{rung}.jsonl"
        if rung in active_rungs:
            p.write_text("", encoding="utf-8")
        run_paths[rung] = p

    report_rows: list[dict[str, Any]] = []
    bm25_cache: dict[str, list[RunResult]] = {}
    dense_cache: dict[str, list[RunResult]] = {}
    rrf_cache: dict[str, list[RunResult]] = {}
    graph_cache: dict[str, list[RunResult]] = {}

    for rung in ALL_RUNGS:
        if rung not in active_rungs:
            report_rows.append({"rung": rung, "status": "skipped", "metrics": {}})
            continue

        print(f"\n-- Rung: {rung} --------------------------------------")
        t_rung = time.time()
        queries_run = 0

        for qi, q in enumerate(queries, start=1):
            qid = str(q.get("id", q.get("query_id", f"q_{qi:04d}")))
            qtext = str(q.get("query", ""))
            intent = str(q.get("intent", "EXPLORATION"))
            if qi % 20 == 1:
                print(f"  [{qi}/{len(queries)}] {qid}: {qtext[:60]}", flush=True)

            if rung == "bm25":
                res = rung_bm25(index, corpus_by_bm25_id, corpus_by_stable_id,
                                qtext, qid, args.top_k, run_paths["bm25"])
                bm25_cache[qid] = res

            elif rung == "bm25_structured":
                bm25 = bm25_cache.get(qid) or rung_bm25(
                    index, corpus_by_bm25_id, corpus_by_stable_id,
                    qtext, qid, args.top_k, NULL_PATH,
                )
                rung_bm25_structured(bm25, corpus_by_stable_id, qtext, qid,
                                     intent, args.top_k, run_paths["bm25_structured"])

            elif rung == "dense_bge":
                res = rung_dense_bge(encoder, embeddings, qtext, qid,
                                     args.top_k, run_paths["dense_bge"])
                dense_cache[qid] = res

            elif rung == "hybrid_rrf":
                bm25 = bm25_cache.get(qid) or rung_bm25(
                    index, corpus_by_bm25_id, corpus_by_stable_id,
                    qtext, qid, args.top_k, NULL_PATH,
                )
                dense = dense_cache.get(qid) or rung_dense_bge(
                    encoder, embeddings, qtext, qid, args.top_k, NULL_PATH,
                )
                res = rung_hybrid_rrf(bm25, dense, qid, args.top_k, run_paths["hybrid_rrf"])
                rrf_cache[qid] = res

            elif rung == "hybrid_graph":
                rrf = rrf_cache.get(qid)
                if rrf is None:
                    bm25 = bm25_cache.get(qid) or rung_bm25(
                        index, corpus_by_bm25_id, corpus_by_stable_id,
                        qtext, qid, args.top_k, NULL_PATH,
                    )
                    dense = dense_cache.get(qid) or rung_dense_bge(
                        encoder, embeddings, qtext, qid, args.top_k, NULL_PATH,
                    )
                    rrf = rung_hybrid_rrf(bm25, dense, qid, args.top_k, NULL_PATH)
                res = rung_hybrid_graph(
                    rrf,
                    graph,
                    q,
                    qid,
                    args.top_k,
                    run_paths["hybrid_graph"],
                    graph_score_weight=args.graph_score_weight,
                    concept_index=concept_index,
                )
                graph_cache[qid] = res

            elif rung == "typed_kg":
                rrf = rrf_cache.get(qid)
                if rrf is None:
                    bm25 = bm25_cache.get(qid) or rung_bm25(
                        index, corpus_by_bm25_id, corpus_by_stable_id,
                        qtext, qid, args.top_k, NULL_PATH,
                    )
                    dense = dense_cache.get(qid) or rung_dense_bge(
                        encoder, embeddings, qtext, qid, args.top_k, NULL_PATH,
                    )
                    rrf = rung_hybrid_rrf(bm25, dense, qid, args.top_k, NULL_PATH)
                rung_typed_kg(
                    rrf, typed_index, corpus_by_stable_id, qid, args.top_k,
                    run_paths["typed_kg"], qualified=False,
                    score_weight=args.typed_kg_score_weight,
                )

            elif rung == "typed_kg_qualified":
                rrf = rrf_cache.get(qid)
                if rrf is None:
                    bm25 = bm25_cache.get(qid) or rung_bm25(
                        index, corpus_by_bm25_id, corpus_by_stable_id,
                        qtext, qid, args.top_k, NULL_PATH,
                    )
                    dense = dense_cache.get(qid) or rung_dense_bge(
                        encoder, embeddings, qtext, qid, args.top_k, NULL_PATH,
                    )
                    rrf = rung_hybrid_rrf(bm25, dense, qid, args.top_k, NULL_PATH)
                rung_typed_kg(
                    rrf, typed_index, corpus_by_stable_id, qid, args.top_k,
                    run_paths["typed_kg_qualified"], qualified=True,
                    score_weight=args.typed_kg_score_weight,
                )

            elif rung == "full":
                gr = graph_cache.get(qid)
                if gr is None:
                    rrf = rrf_cache.get(qid)
                    if rrf is None:
                        bm25 = bm25_cache.get(qid) or rung_bm25(
                            index, corpus_by_bm25_id, corpus_by_stable_id,
                            qtext, qid, args.top_k, NULL_PATH,
                        )
                        dense = dense_cache.get(qid) or rung_dense_bge(
                            encoder, embeddings, qtext, qid, args.top_k, NULL_PATH,
                        )
                        rrf = rung_hybrid_rrf(bm25, dense, qid, args.top_k, NULL_PATH)
                    gr = rung_hybrid_graph(
                        rrf,
                        graph,
                        q,
                        qid,
                        args.top_k,
                        NULL_PATH,
                        graph_score_weight=args.graph_score_weight,
                        concept_index=concept_index,
                    )
                rung_full(gr, corpus_by_stable_id, qid, args.top_k, run_paths["full"])

            queries_run += 1

        elapsed = time.time() - t_rung
        metrics = compute_metrics(run_paths[rung], auto_labels)
        print(f"  Rung '{rung}' done: {queries_run} queries in {elapsed:.1f}s")
        if metrics:
            print(f"  Metrics: {metrics}")
        report_rows.append({
            "rung": rung,
            "status": "ok",
            "queries_run": queries_run,
            "elapsed_s": round(elapsed, 1),
            "metrics": metrics,
        })

    # -- Report -----------------------------------------------------------
    _print_table(report_rows)
    _write_report(args.out_dir, report_rows, partial=is_partial_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
