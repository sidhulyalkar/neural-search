"""Shared CLI helpers for changing corpus QA status."""

from __future__ import annotations

import argparse
import json

from neural_search.qa import DatasetQAStatus, update_dataset_status


def mark_main(status: DatasetQAStatus, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Mark a dataset as {status} in the corpus QA workflow."
    )
    parser.add_argument("--dataset-id", required=True, help="Stable dataset/source ID to update.")
    parser.add_argument("--notes", default=None, help="Optional reviewer notes to persist.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)

    record = update_dataset_status(args.dataset_id, status, notes=args.notes)
    if args.json:
        print(json.dumps(record, indent=2))
    else:
        print(f"{args.dataset_id} marked {record['qa_status']}.")
    return 0
