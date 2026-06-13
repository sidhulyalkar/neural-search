#!/usr/bin/env python3
"""Build the field-state memory graph from all available corpus inputs.

Usage::

    python scripts/field_state/build_memory_graph.py \\
        --corpus data/corpus/normalized/combined_corpus.jsonl \\
        --out-dir artifacts/field_state

Outputs:
    artifacts/field_state/memory_graph_nodes.jsonl
    artifacts/field_state/memory_graph_edges.jsonl
    artifacts/field_state/memory_graph_manifest.json
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Allow running from repo root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.field_state.memory_graph import MemoryGraphBuilder

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("build_memory_graph")

# Default artifact locations
DEFAULT_CORPUS = Path("data/corpus/normalized/combined_corpus.jsonl")
DEFAULT_EVIDENCE_PACKETS = Path("artifacts/field_state/neuro_judge_evidence_packets.jsonl")
DEFAULT_JUDGMENTS = Path("artifacts/field_state/neuro_qrels_consensus.jsonl")
DEFAULT_FEEDBACK = Path("artifacts/frontend/retrieval_feedback.jsonl")
DEFAULT_CONCEPTS = Path("artifacts/field_state/concept_memory/concepts.jsonl")
DEFAULT_OUT_DIR = Path("artifacts/field_state")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--evidence-packets", type=Path, default=DEFAULT_EVIDENCE_PACKETS)
    p.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    p.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK)
    p.add_argument("--concepts", type=Path, default=DEFAULT_CONCEPTS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--skip-evidence", action="store_true", help="Skip evidence packets and judgments"
    )
    p.add_argument("--skip-feedback", action="store_true", help="Skip feedback signals")
    p.add_argument("--skip-concepts", action="store_true", help="Skip concept memory")
    p.add_argument("--dry-run", action="store_true", help="Build but do not write outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.monotonic()

    out_dir: Path = args.out_dir
    nodes_path = out_dir / "memory_graph_nodes.jsonl"
    edges_path = out_dir / "memory_graph_edges.jsonl"
    manifest_path = out_dir / "memory_graph_manifest.json"

    corpus_path = args.corpus if args.corpus.exists() else None
    if corpus_path is None:
        log.warning("Corpus not found at %s — building from empty set", args.corpus)

    builder = MemoryGraphBuilder()
    store = builder.build(
        corpus_path=corpus_path,
        evidence_packets_path=None if args.skip_evidence else args.evidence_packets,
        judgments_path=None if args.skip_evidence else args.judgments,
        feedback_path=None if args.skip_feedback else args.feedback,
        concept_memory_path=None if args.skip_concepts else args.concepts,
    )

    elapsed = time.monotonic() - t0
    log.info("Graph built in %.2fs: %s", elapsed, store)

    if args.dry_run:
        log.info("--dry-run: not writing outputs")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    store.export_jsonl(nodes_path, edges_path)
    log.info("Wrote nodes → %s", nodes_path)
    log.info("Wrote edges → %s", edges_path)

    build_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = store.write_manifest(manifest_path, build_id=build_id)
    log.info("Wrote manifest → %s", manifest_path)

    # Print summary
    print(f"\nMemory graph built in {elapsed:.2f}s")
    print(f"  Total nodes : {manifest['total_nodes']}")
    print(f"  Total edges : {manifest['total_edges']}")
    print(f"  Node types  : {len(manifest['node_counts_by_type'])}")
    for ntype, count in sorted(manifest["node_counts_by_type"].items(), key=lambda x: -x[1])[:10]:
        print(f"    {ntype:<35} {count}")


if __name__ == "__main__":
    main()
