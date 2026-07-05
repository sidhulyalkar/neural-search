#!/usr/bin/env python3
"""Match newer analysis methods to older or date-unknown datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.eval.reanalysis_affordance_lib import analyze_record, load_jsonl

DEFAULT_CORPUS = Path("data/eval/ablation_corpus_from_packets.jsonl")
DEFAULT_OUT = Path("reports/eval/new_method_dataset_matches.json")
DEFAULT_MD = Path("reports/eval/new_method_dataset_matches.md")


def build_matches(records: list[dict[str, Any]], min_score: float = 0.55, top_n: int = 75) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for record in records:
        analyzed = analyze_record(record)
        year = analyzed["year"]
        for method in analyzed["top_methods"]:
            is_old = year is not None and year < int(method["novelty_year"])
            date_unknown = year is None
            if method["score"] < min_score or not (is_old or date_unknown):
                continue
            matches.append(
                {
                    "record_id": analyzed["record_id"],
                    "title": analyzed["title"],
                    "source": analyzed["source"],
                    "year": year,
                    "method": method["method"],
                    "display_name": method["display_name"],
                    "method_novelty_year": method["novelty_year"],
                    "score": method["score"],
                    "date_status": "older_than_method" if is_old else "date_unknown",
                    "missing_requirements": method["missing_requirements"],
                    "reinterpretation": method["reinterpretation"],
                }
            )
    matches.sort(
        key=lambda row: (
            -float(row["score"]),
            row["date_status"] != "older_than_method",
            row["source"],
            row["record_id"],
            row["method"],
        )
    )
    return {
        "match_count": len(matches),
        "min_score": min_score,
        "matches": matches[:top_n],
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# New Method To Old Dataset Matches",
        "",
        f"- Matches above threshold: {report['match_count']}",
        f"- Minimum score: {report['min_score']}",
        "",
        "| Rank | Dataset | Method | Score | Date status | Missing requirements | Rationale |",
        "|---:|---|---|---:|---|---|---|",
    ]
    for rank, row in enumerate(report["matches"], start=1):
        title = str(row["title"]).replace("|", "\\|")[:80]
        missing = ", ".join(row["missing_requirements"]) or "none"
        rationale = str(row["reinterpretation"]).replace("|", "\\|")
        year = row["year"] if row["year"] is not None else "unknown"
        lines.append(
            f"| {rank} | `{row['record_id']}`<br>{title} | `{row['method']}` "
            f"({row['method_novelty_year']}) | {row['score']:.3f} | "
            f"{row['date_status']} ({year}) | {missing} | {rationale} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--top-n", type=int, default=75)
    args = parser.parse_args(argv)

    report = build_matches(load_jsonl(args.corpus), min_score=args.min_score, top_n=args.top_n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.md)
    print(f"Wrote {args.out} and {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
