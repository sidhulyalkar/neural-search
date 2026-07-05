#!/usr/bin/env python3
"""Report datasets that are promising candidates for reanalysis."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.eval.reanalysis_affordance_lib import analyze_record, load_jsonl

DEFAULT_CORPUS = Path("data/eval/ablation_corpus_from_packets.jsonl")
DEFAULT_OUT = Path("reports/eval/reanalysis_affordance_report.json")
DEFAULT_MD = Path("reports/eval/reanalysis_affordance_report.md")


def build_report(records: list[dict[str, Any]], top_n: int = 50) -> dict[str, Any]:
    rows = [analyze_record(record) for record in records]
    rows.sort(key=lambda row: (-row["best_method_score"], row["source"], row["record_id"]))
    source_counts = Counter(row["source"] for row in rows)
    method_counts: Counter[str] = Counter()
    missing_counts: Counter[str] = Counter()
    for row in rows:
        if row["top_methods"] and row["top_methods"][0]["score"] >= 0.55:
            method_counts[row["top_methods"][0]["method"]] += 1
        missing_counts.update(row["missing_metadata"])
    return {
        "dataset_count": len(rows),
        "high_affordance_count": sum(1 for row in rows if row["best_method_score"] >= 0.7),
        "medium_affordance_count": sum(1 for row in rows if 0.55 <= row["best_method_score"] < 0.7),
        "source_counts": dict(sorted(source_counts.items())),
        "top_method_counts": dict(method_counts.most_common()),
        "metadata_gap_counts": dict(missing_counts.most_common()),
        "top_datasets": rows[:top_n],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Reanalysis Affordance Report",
        "",
        f"- Datasets analyzed: {report['dataset_count']}",
        f"- High-affordance candidates: {report['high_affordance_count']}",
        f"- Medium-affordance candidates: {report['medium_affordance_count']}",
        "",
        "## Top Method Opportunities",
        "",
        "| Method | Candidate datasets |",
        "|---|---:|",
    ]
    for method, count in report["top_method_counts"].items():
        lines.append(f"| `{method}` | {count} |")
    lines.extend([
        "",
        "## Metadata Gaps Blocking Reanalysis",
        "",
        "| Gap | Count |",
        "|---|---:|",
    ])
    for gap, count in report["metadata_gap_counts"].items():
        lines.append(f"| `{gap}` | {count} |")
    lines.extend([
        "",
        "## Top Dataset Candidates",
        "",
        "| Rank | Dataset | Source | Best method | Score | Missing requirements |",
        "|---:|---|---|---|---:|---|",
    ])
    for rank, row in enumerate(report["top_datasets"], start=1):
        best = row["top_methods"][0] if row["top_methods"] else {}
        missing = ", ".join(best.get("missing_requirements", [])) or "none"
        title = str(row["title"]).replace("|", "\\|")[:90]
        lines.append(
            f"| {rank} | `{row['record_id']}`<br>{title} | {row['source']} | "
            f"`{best.get('method', 'n/a')}` | {best.get('score', 0):.3f} | {missing} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args(argv)

    records = load_jsonl(args.corpus)
    report = build_report(records, top_n=args.top_n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.md)
    print(f"Wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
