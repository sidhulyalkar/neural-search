"""List datasets still waiting for corpus QA review.

Usage:
    python -m neural_search.qa.list_unreviewed
"""

from __future__ import annotations

import argparse
import json

from neural_search.qa import list_unreviewed_records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.qa.list_unreviewed",
        description="List datasets with unreviewed, auto-generated, or needs-review QA status.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    records = list_unreviewed_records()
    rows = [
        {
            "dataset_id": record["dataset"].get("source_id", record["dataset"].get("id")),
            "qa_status": record["dataset"]["qa_status"],
            "title": record["dataset"].get("title", "Untitled"),
            "missing_fields": record.get("extraction").missing_fields
            if record.get("extraction") is not None
            else [],
        }
        for record in records
    ]

    if args.json:
        print(json.dumps(rows, indent=2))
        return 0

    if not rows:
        print("No unreviewed datasets found.")
        return 0

    print(f"{'Dataset ID':<38} {'Status':<16} Title")
    print("-" * 88)
    for row in rows:
        print(f"{row['dataset_id']:<38} {row['qa_status']:<16} {row['title']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
