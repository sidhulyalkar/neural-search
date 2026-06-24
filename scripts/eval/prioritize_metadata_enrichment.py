#!/usr/bin/env python3
"""Convert qrels failure analysis into metadata enrichment priorities."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_FAILURES = Path("reports/eval/failure_analysis.json")
DEFAULT_OUT = Path("reports/eval/metadata_enrichment_priorities.json")
DEFAULT_MD = Path("reports/eval/metadata_enrichment_priorities.md")

FIELD_TO_ACTION = {
    "modality": "Add or normalize modality labels from source metadata and file manifests.",
    "modalities": "Add or normalize modality labels from source metadata and file manifests.",
    "species": "Normalize species aliases and source-specific organism fields.",
    "task": "Extract task and behavioral paradigm labels from titles, descriptions, and protocol files.",
    "tasks": "Extract task and behavioral paradigm labels from titles, descriptions, and protocol files.",
    "brain_region": "Extract anatomical regions from repository metadata, paper links, and file annotations.",
    "raw_data": "Verify raw-data availability and expose raw/processed evidence separately.",
    "behavior": "Detect behavioral events, trial tables, and task timing fields.",
    "data_standard": "Normalize NWB, BIDS, and source-specific file-standard metadata.",
}


def build_priorities(failure_report: dict[str, Any]) -> dict[str, Any]:
    field_counts: Counter[str] = Counter()
    mismatch_counts: Counter[str] = Counter()
    intent_field_counts: dict[str, Counter[str]] = defaultdict(Counter)
    source_counts: Counter[str] = Counter()

    for variant, stats in (failure_report.get("variants") or {}).items():
        field_counts.update(stats.get("fp_metadata_missing_counts") or {})
        field_counts.update(stats.get("fn_metadata_missing_counts") or {})
        mismatch_counts.update(stats.get("fp_mismatch_counts") or {})
        source_counts.update(stats.get("source_fp_counts") or {})
        for intent, count in (stats.get("intent_fp_counts") or {}).items():
            for field, field_count in (stats.get("fp_metadata_missing_counts") or {}).items():
                intent_field_counts[intent][field] += min(int(count), int(field_count))

    priorities: list[dict[str, Any]] = []
    for field, count in field_counts.most_common():
        normalized = str(field).casefold()
        action = FIELD_TO_ACTION.get(normalized, f"Audit and normalize `{field}` evidence.")
        related_mismatches = [
            name
            for name, _ in mismatch_counts.most_common()
            if normalized.split("_")[0] in name
        ][:5]
        priorities.append(
            {
                "field": field,
                "failure_count": count,
                "recommended_action": action,
                "related_mismatches": related_mismatches,
                "top_intents": [
                    {"intent": intent, "count": fields[field]}
                    for intent, fields in sorted(
                        intent_field_counts.items(),
                        key=lambda item: -item[1].get(field, 0),
                    )
                    if fields.get(field, 0)
                ][:5],
            }
        )

    return {
        "priority_count": len(priorities),
        "priorities": priorities,
        "top_false_positive_sources": dict(source_counts.most_common(10)),
        "top_mismatch_modes": dict(mismatch_counts.most_common(10)),
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Metadata Enrichment Priorities",
        "",
        "Priorities are derived from the current qrels failure-analysis false positives and false negatives.",
        "",
        "| Rank | Field | Failure count | Recommended action | Related mismatch modes |",
        "|---:|---|---:|---|---|",
    ]
    for rank, row in enumerate(report["priorities"], start=1):
        mismatches = ", ".join(f"`{m}`" for m in row["related_mismatches"]) or "none"
        lines.append(
            f"| {rank} | `{row['field']}` | {row['failure_count']} | "
            f"{row['recommended_action']} | {mismatches} |"
        )
    lines.extend([
        "",
        "## Top False-Positive Sources",
        "",
        "| Source | Count |",
        "|---|---:|",
    ])
    for source, count in report["top_false_positive_sources"].items():
        lines.append(f"| `{source}` | {count} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failures", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    report = build_priorities(json.loads(args.failures.read_text(encoding="utf-8")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.md)
    print(f"Wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
