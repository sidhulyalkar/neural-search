#!/usr/bin/env python3
"""Diagnose retrieval loading bottlenecks and measure per-stage latency.

Usage::

    python scripts/eval/diagnose_retrieval_loading.py \\
        --corpus data/corpus/normalized/combined_corpus.jsonl \\
        --out-dir reports/eval

Outputs:
    reports/eval/retrieval_loading_diagnostics.md
    reports/eval/retrieval_loading_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("diagnose_retrieval_loading")

DEFAULT_CORPUS = Path("data/corpus/normalized/combined_corpus.jsonl")
DEFAULT_EMBEDDING_CACHE = Path("data/embeddings/real_all.dense.field_embeddings.jsonl")
DEFAULT_INDEX = Path("data/index/turbovec_dense_1024.index")
DEFAULT_GRAPH = Path("data/graph/neural_search_graph.demo_v05.json")
DEFAULT_NODES = Path("artifacts/field_state/memory_graph_nodes.jsonl")
DEFAULT_EDGES = Path("artifacts/field_state/memory_graph_edges.jsonl")
DEFAULT_OUT_DIR = Path("reports/eval")

TEST_QUERIES = [
    "Neuropixels recordings of hippocampal place cells in mice",
    "Two-photon calcium imaging of visual cortex in mice",
    "fMRI during decision making in humans",
    "Spike sorting from silicon probes in frontal cortex",
    "EEG motor imagery for BCI",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDING_CACHE)
    p.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    p.add_argument("--memory-graph-nodes", type=Path, default=DEFAULT_NODES)
    p.add_argument("--memory-graph-edges", type=Path, default=DEFAULT_EDGES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--n-queries", type=int, default=5)
    return p.parse_args()


def measure(label: str, fn) -> tuple[Any, float]:
    t0 = time.monotonic()
    result = fn()
    elapsed = time.monotonic() - t0
    return result, elapsed


def main() -> None:  # noqa: C901
    args = parse_args()
    timings: dict[str, float | None] = {}
    sizes: dict[str, int] = {}
    warnings: list[str] = []

    # --- 1. Corpus load ---
    log.info("Measuring corpus load time...")
    try:
        t0 = time.monotonic()
        records = []
        if args.corpus.exists():
            with args.corpus.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        records.append(json.loads(line))
        timings["corpus_load_s"] = time.monotonic() - t0
        sizes["corpus_record_count"] = len(records)
        log.info("  corpus: %d records in %.3fs", len(records), timings["corpus_load_s"])
    except Exception as exc:
        warnings.append(f"corpus load failed: {exc}")
        timings["corpus_load_s"] = None
        records = []

    # --- 2. Embedding cache load ---
    log.info("Measuring embedding cache load time...")
    try:
        t0 = time.monotonic()
        emb_rows = 0
        if args.embedding_cache.exists():
            with args.embedding_cache.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        emb_rows += 1
        timings["embedding_cache_load_s"] = time.monotonic() - t0
        sizes["embedding_cache_rows"] = emb_rows
        log.info("  embedding cache: %d rows in %.3fs", emb_rows, timings["embedding_cache_load_s"])
    except Exception as exc:
        warnings.append(f"embedding cache load failed: {exc}")
        timings["embedding_cache_load_s"] = None

    # --- 3. TurboVec index load ---
    log.info("Measuring TurboVec index load time...")
    if args.index.exists():
        try:
            from neural_search.embeddings.turbovec_index import NeuralSearchTurboIndex
            t0 = time.monotonic()
            NeuralSearchTurboIndex.load(str(args.index))
            timings["turbovec_index_load_s"] = time.monotonic() - t0
            log.info("  turbovec index loaded in %.3fs", timings["turbovec_index_load_s"])
        except Exception as exc:
            warnings.append(f"turbovec index load failed: {exc}")
            timings["turbovec_index_load_s"] = None
    else:
        warnings.append(f"turbovec index not found at {args.index}")
        timings["turbovec_index_load_s"] = None

    # --- 4. Knowledge graph (legacy) ---
    log.info("Measuring legacy graph load time...")
    if args.graph.exists():
        try:
            from neural_search.graph.schema import read_graph_json
            t0 = time.monotonic()
            kg = read_graph_json(str(args.graph))
            timings["legacy_graph_load_s"] = time.monotonic() - t0
            sizes["legacy_graph_nodes"] = len(kg.nodes)
            sizes["legacy_graph_edges"] = len(kg.edges)
            log.info("  legacy graph: %d nodes, %d edges in %.3fs", len(kg.nodes), len(kg.edges), timings["legacy_graph_load_s"])
        except Exception as exc:
            warnings.append(f"legacy graph load failed: {exc}")
            timings["legacy_graph_load_s"] = None
    else:
        timings["legacy_graph_load_s"] = None

    # --- 5. Field-state memory graph ---
    log.info("Measuring field-state memory graph load time...")
    if args.memory_graph_nodes.exists():
        try:
            from neural_search.field_state.graph_store import FieldStateGraphStore
            t0 = time.monotonic()
            store = FieldStateGraphStore.from_jsonl(args.memory_graph_nodes, args.memory_graph_edges)
            timings["memory_graph_load_s"] = time.monotonic() - t0
            sizes["memory_graph_nodes"] = store.node_count
            sizes["memory_graph_edges"] = store.edge_count
            log.info("  memory graph: %s in %.3fs", store, timings["memory_graph_load_s"])
        except Exception as exc:
            warnings.append(f"memory graph load failed: {exc}")
            timings["memory_graph_load_s"] = None
    else:
        warnings.append("Memory graph not built yet — run build_memory_graph.py first")
        timings["memory_graph_load_s"] = None

    # --- 6. Query parse time ---
    log.info("Measuring query parse time...")
    try:
        from neural_search.core.query import parse_and_plan_query
        query_times: list[float] = []
        for q in TEST_QUERIES[:args.n_queries]:
            t0 = time.monotonic()
            parse_and_plan_query(q)
            query_times.append(time.monotonic() - t0)
        timings["query_parse_avg_s"] = sum(query_times) / len(query_times) if query_times else None
        log.info("  query parse avg: %.4fs", timings["query_parse_avg_s"] or 0)
    except Exception as exc:
        warnings.append(f"query parse failed: {exc}")
        timings["query_parse_avg_s"] = None

    # --- 7. Search latency (sparse BM25 path) ---
    log.info("Measuring search latency (sparse path)...")
    if records:
        try:
            from neural_search.search.core import search_datasets
            search_times: list[float] = []
            for q in TEST_QUERIES[:args.n_queries]:
                t0 = time.monotonic()
                search_datasets(q, records[:500], config={"graph": {"enabled": False}})
                search_times.append(time.monotonic() - t0)
            timings["search_sparse_avg_s"] = sum(search_times) / len(search_times) if search_times else None
            log.info("  sparse search avg: %.4fs", timings["search_sparse_avg_s"] or 0)
        except Exception as exc:
            warnings.append(f"sparse search failed: {exc}")
            timings["search_sparse_avg_s"] = None

    # --- Report ---
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timings_seconds": dict(timings),
        "artifact_sizes": sizes,
        "warnings": warnings,
        "test_queries": TEST_QUERIES[:args.n_queries],
    }
    json_path = out_dir / "retrieval_loading_diagnostics.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    # Markdown
    lines = ["# Retrieval Loading Diagnostics", ""]
    lines += ["## Timings"]
    for label, val in timings.items():
        display = f"{val*1000:.1f} ms" if val is not None else "N/A"
        lines.append(f"- `{label}`: {display}")
    lines += ["", "## Artifact Sizes"]
    for label, val in sizes.items():
        lines.append(f"- `{label}`: {val:,}")
    if warnings:
        lines += ["", "## Warnings"]
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
    md_path = out_dir / "retrieval_loading_diagnostics.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    log.info("Wrote diagnostics → %s and %s", json_path, md_path)
    print("\nRetrieval Loading Diagnostics")
    for label, val in timings.items():
        display = f"{val*1000:.1f} ms" if val is not None else "N/A"
        print(f"  {label:<40} {display}")
    if warnings:
        print(f"\n  {len(warnings)} warning(s) — see {json_path}")


# Type hint fix for Python 3.9
from typing import Any  # noqa: E402

if __name__ == "__main__":
    main()
