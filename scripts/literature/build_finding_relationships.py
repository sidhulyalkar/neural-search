#!/usr/bin/env python3
"""Build cross-finding and region co-occurrence relationships from normalized findings.

Produces three JSONL artifacts:
  - finding_edges.jsonl         (supports / contradicts edges between findings)
  - region_cooccurrence.jsonl   (region_co_occurs_with edges)
  - consensus_summaries.jsonl   (per-region direction consensus records)

Usage:
    python scripts/literature/build_finding_relationships.py \\
        --findings artifacts/literature/findings_tier1_normalized.jsonl \\
        --out-dir  artifacts/literature/relationships
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural_search.literature.relationship_builder import (  # noqa: E402
    build_consensus_summaries,
    build_cross_finding_edges,
    build_region_cooccurrence_edges,
    write_edges_jsonl,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--findings", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--min-cooccurrences", type=int, default=2,
        help="Minimum co-occurrences for a region pair edge (default: 2)",
    )
    p.add_argument(
        "--min-papers", type=int, default=2,
        help="Minimum papers for a consensus record (default: 2)",
    )
    p.add_argument(
        "--max-finding-edges", type=int, default=200_000,
        help="Safety cap on finding-level edges (default: 200,000)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.findings.exists():
        logger.error("Findings file not found: %s", args.findings)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building cross-finding edges (supports / contradicts)…")
    finding_edges = build_cross_finding_edges(
        args.findings,
        min_shared_regions=1,
        max_edges=args.max_finding_edges,
    )
    finding_edges_path = args.out_dir / "finding_edges.jsonl"
    write_edges_jsonl(finding_edges, finding_edges_path)

    logger.info("Building region co-occurrence edges…")
    region_edges = build_region_cooccurrence_edges(
        args.findings,
        min_cooccurrences=args.min_cooccurrences,
    )
    region_path = args.out_dir / "region_cooccurrence.jsonl"
    write_edges_jsonl(region_edges, region_path)

    logger.info("Building consensus summaries…")
    consensus = build_consensus_summaries(
        args.findings,
        min_papers=args.min_papers,
    )
    consensus_path = args.out_dir / "consensus_summaries.jsonl"
    with consensus_path.open("w") as fh:
        for r in consensus:
            fh.write(json.dumps(asdict(r)) + "\n")
    logger.info("Wrote %d consensus records to %s", len(consensus), consensus_path)

    # --- Print summary ---
    n_supports = sum(1 for e in finding_edges if e.edge_type == "supports")
    n_contradicts = sum(1 for e in finding_edges if e.edge_type == "contradicts")
    strong_consensus = [r for r in consensus if r.consensus_strength >= 0.8 and r.n_papers >= 3]

    print("\n=== Relationship Build Summary ===")
    print(f"  Finding edges total:        {len(finding_edges):,}")
    print(f"    supports:                 {n_supports:,}")
    print(f"    contradicts:              {n_contradicts:,}")
    print(f"  Region co-occurrence edges: {len(region_edges):,}")
    print(f"  Consensus records:          {len(consensus):,}")
    print(f"  Strong consensus (≥0.8, ≥3 papers): {len(strong_consensus):,}")

    if strong_consensus:
        print("\n  Top established findings (strong consensus):")
        for r in strong_consensus[:10]:
            task_str = f" during {r.task}" if r.task else ""
            print(
                f"    {r.region}{task_str}: {r.direction} "
                f"({r.n_papers} papers, strength={r.consensus_strength:.2f})"
            )

    if region_edges:
        print("\n  Top region co-occurrence pairs:")
        for edge in region_edges[:8]:
            print(f"    {edge.region_a} + {edge.region_b}: {edge.n_findings} findings")


if __name__ == "__main__":
    main()
