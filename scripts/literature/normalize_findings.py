#!/usr/bin/env python3
"""Post-process a findings JSONL file: normalize species/regions/tasks, dedup, quality-flag.

Usage:
    python scripts/literature/normalize_findings.py \\
        --in  artifacts/literature/findings_tier1_ollama.jsonl \\
        --out artifacts/literature/findings_tier1_normalized.jsonl \\
        --report artifacts/eval/normalization_report.json

The output file is a drop-in replacement for the raw extraction output.
Each record gains two optional fields:
  _normalized   — dict of original values for any field that was changed
  quality_flags — list of flag strings (generic_region_only, no_species, …)

Records are not deleted: all findings pass through.  Consumers can
filter by quality_flags as needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from neural_search.literature.normalizer import deduplicate_findings, normalize_finding


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="input", required=True, type=Path)
    p.add_argument("--out", dest="output", required=True, type=Path)
    p.add_argument("--report", dest="report", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.input.exists():
        logger.error("Input file not found: %s", args.input)
        sys.exit(1)

    logger.info("Loading findings from %s", args.input)
    raw: list[dict] = []
    with args.input.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed line: %s", exc)

    logger.info("Loaded %d findings", len(raw))

    # --- Pass 1: normalize each finding ---
    normalized = [normalize_finding(r) for r in raw]

    # --- Pass 2: deduplicate within each paper ---
    deduped, n_removed = deduplicate_findings(normalized)
    logger.info("Removed %d exact-duplicate findings", n_removed)

    # --- Write output ---
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for r in deduped:
            fh.write(json.dumps(r) + "\n")
    logger.info("Wrote %d findings to %s", len(deduped), args.output)

    # --- Build report ---
    all_flags: list[str] = []
    n_normalized = 0
    species_changed = 0
    region_changed = 0
    task_changed = 0

    for r in deduped:
        norm = r.get("_normalized", {})
        if norm:
            n_normalized += 1
        if "species_original" in norm:
            species_changed += 1
        if "regions_original" in norm:
            region_changed += 1
        if "tasks_original" in norm:
            task_changed += 1
        all_flags.extend(r.get("quality_flags", []))

    flag_counts = dict(Counter(all_flags).most_common())
    n_clean = sum(1 for r in deduped if not r.get("quality_flags"))

    report = {
        "input_findings": len(raw),
        "duplicates_removed": n_removed,
        "output_findings": len(deduped),
        "records_normalized": n_normalized,
        "species_changed": species_changed,
        "region_changed": region_changed,
        "task_changed": task_changed,
        "quality_flag_counts": flag_counts,
        "clean_findings": n_clean,
        "clean_pct": round(n_clean / len(deduped) * 100, 1) if deduped else 0.0,
    }

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        logger.info("Report written to %s", args.report)

    # Print summary
    print("\n=== Normalization Summary ===")
    print(f"  Input findings:       {report['input_findings']:,}")
    print(f"  Duplicates removed:   {report['duplicates_removed']:,}")
    print(f"  Output findings:      {report['output_findings']:,}")
    print(f"  Records normalized:   {report['records_normalized']:,}")
    print(f"  Species changes:      {report['species_changed']:,}")
    print(f"  Region changes:       {report['region_changed']:,}")
    print(f"  Task changes:         {report['task_changed']:,}")
    print(f"  Clean findings:       {report['clean_findings']:,} ({report['clean_pct']}%)")
    print("\n  Quality flag breakdown:")
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        pct = count / len(deduped) * 100
        print(f"    {flag}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
