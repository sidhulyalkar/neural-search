#!/usr/bin/env python3
"""Corpus quality dashboard — validate corpus completeness and provenance.

Produces both pass/fail output and a Markdown table of per-source metrics.

Exit criteria:
  - Total usable records >= 4000
  - No records without a persistent identifier
  - Tier 2 rejection log exists and is non-empty

Usage:
    python scripts/validate_corpus.py
    python scripts/validate_corpus.py --output reports/corpus_quality.md
    python scripts/validate_corpus.py --min-records 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CORPUS_DIR = Path("data/corpus/normalized")
REJECTION_LOG = Path("data/corpus/rejected/tier2_rejected.jsonl")
REPORT_PATH = Path("reports/corpus_quality.md")


def _has_modality(rec: dict) -> bool:
    mods = rec.get("modalities", [])
    return bool(mods) and any(
        (m.get("label") if isinstance(m, dict) else m) for m in mods
    )


def _has_doi(rec: dict) -> bool:
    doi = rec.get("doi") or (rec.get("metadata_json") or {}).get("doi")
    return bool(doi)


def _has_accession(rec: dict) -> bool:
    accession = rec.get("source_id") or rec.get("dataset_id")
    return bool(accession and str(accession).strip())


def _source_from_file(filepath: Path) -> str:
    name = filepath.stem
    if name.startswith("real_"):
        return name[len("real_"):]
    return name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(REPORT_PATH))
    parser.add_argument("--min-records", type=int, default=4000)
    args = parser.parse_args(argv)

    files = sorted(CORPUS_DIR.glob("real_*.jsonl"))
    rows: list[dict] = []
    total_usable = 0
    no_id_count = 0

    for f in files:
        source = _source_from_file(f)
        recs = []
        with f.open() as fh:
            for line in fh:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        raw = len(recs)
        with_modality = sum(1 for r in recs if _has_modality(r))
        with_id = sum(1 for r in recs if _has_doi(r) or _has_accession(r))
        no_id = raw - with_id
        no_id_count += no_id
        total_usable += raw
        modality_pct = round(100 * with_modality / raw, 1) if raw else 0.0

        rows.append({
            "source": source,
            "raw": raw,
            "usable": raw,
            "modality_pct": modality_pct,
            "no_id": no_id,
        })

    tier2_rejections = 0
    if REJECTION_LOG.exists():
        tier2_rejections = sum(1 for _ in REJECTION_LOG.open())

    header = "| Source | Raw | Usable | Modality% | No-ID |\n|--------|-----|--------|-----------|-------|"
    table_lines = [header]
    for row in rows:
        table_lines.append(
            f"| {row['source']} | {row['raw']} | {row['usable']} "
            f"| {row['modality_pct']}% | {row['no_id']} |"
        )
    table_lines.append(
        f"| **TOTAL** | {sum(r['raw'] for r in rows)} | {total_usable} "
        f"| — | {no_id_count} |"
    )
    table = "\n".join(table_lines)

    checks = {
        f"total_usable >= {args.min_records}": total_usable >= args.min_records,
        "tier2_rejection_log_exists": REJECTION_LOG.exists(),
        "tier2_rejection_log_non_empty": tier2_rejections > 0,
        "no_records_without_identifier": no_id_count == 0,
    }

    all_pass = all(checks.values())

    report = f"""# Corpus Quality Report

{table}

## Checks
{"".join(f'- [{"x" if v else " "}] {k}\n' for k, v in checks.items())}
**Tier 2 rejections logged:** {tier2_rejections}

**Status: {"PASS" if all_pass else "FAIL"}**
"""

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(report)
    print(report)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
