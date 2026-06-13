#!/usr/bin/env python3
"""Run the v2.0 integration point: rebuild graph, re-embed, rebuild index.

Run only after Track 1 and Track 2 both pass their exit criteria.

Usage:
    python scripts/run_integration.py
    python scripts/run_integration.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _run(cmd: list[str], *, dry_run: bool = False) -> int:
    if dry_run:
        print(f"  [dry-run] would run: {' '.join(cmd)}")
        return 0
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"  ERROR: command failed with exit code {result.returncode}")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph rebuild")
    parser.add_argument("--skip-embed", action="store_true", help="Skip embedding recompute")
    parser.add_argument("--skip-index", action="store_true", help="Skip turbovec index rebuild")
    args = parser.parse_args(argv)

    print(f"=== Neural Search v2.0 Integration Point ({'DRY RUN' if args.dry_run else 'LIVE'}) ===")
    print(f"Started: {datetime.now(UTC).isoformat()}")

    steps = []

    if not args.skip_graph:
        print("\n[1/3] Rebuilding knowledge graph from expanded corpus...")
        rc = _run([sys.executable, "scripts/rebuild_corpus_graph.py"], dry_run=args.dry_run)
        steps.append(("rebuild_graph", rc == 0))

    if not args.skip_embed:
        print("\n[2/3] Recomputing dense embeddings (BGE-large-en-v1.5)...")
        rc = _run(
            [sys.executable, "scripts/recompute_embeddings.py", "--provider", "dense"],
            dry_run=args.dry_run,
        )
        steps.append(("recompute_embeddings", rc == 0))

    if not args.skip_index:
        print("\n[3/3] Building turbovec index from new embeddings...")
        rc = _run([sys.executable, "scripts/build_turbovec_index.py"], dry_run=args.dry_run)
        steps.append(("build_turbovec_index", rc == 0))

    print("\n=== Integration Summary ===")
    all_ok = True
    for step, passed in steps:
        status = "OK" if passed else "FAILED"
        print(f"  {step}: {status}")
        if not passed:
            all_ok = False

    if not args.dry_run:
        record = {
            "integration_run_at": datetime.now(UTC).isoformat(),
            "steps": dict(steps),
            "all_passed": all_ok,
        }
        Path("reports").mkdir(exist_ok=True)
        Path("reports/integration_run.json").write_text(json.dumps(record, indent=2))

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
