"""CLI for listing notebook templates."""

from __future__ import annotations

import argparse
import json

from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.notebooks.templates import (
    available_templates_for_dataset,
    load_notebook_templates,
)


def _find_dataset(dataset_id: str) -> dict | None:
    for record in build_demo_seed():
        dataset = record["dataset"]
        if dataset.get("source_id") == dataset_id or dataset.get("id") == dataset_id:
            return dataset
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.notebooks.list_templates")
    parser.add_argument("--dataset-id", default=None, help="Optionally show availability for a dataset.")
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    args = parser.parse_args(argv)

    if args.dataset_id:
        dataset = _find_dataset(args.dataset_id)
        if dataset is None:
            print(json.dumps({"error": f"Dataset not found: {args.dataset_id}"}, indent=2))
            return 1
        templates = available_templates_for_dataset(dataset)
    else:
        templates = load_notebook_templates()

    if args.json:
        print(json.dumps({"templates": templates}, indent=2))
        return 0

    for template in templates:
        status = ""
        if "available" in template:
            status = " available" if template["available"] else " unavailable"
        print(f"{template['id']}: {template.get('title', template['id'])}{status}")
        description = template.get("description")
        if description:
            print(f"  {description}")
        missing = template.get("missing_requirements", [])
        if missing:
            print("  Missing: " + "; ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
