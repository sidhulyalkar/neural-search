"""CLI for generating NWB starter notebooks.

Usage:
    python -m neural_search.notebooks.generate --dataset-id <DATASET_ID>
    python -m neural_search.notebooks.generate --dataset-id <DATASET_ID> --asset-id <ASSET_ID>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_search.cards import generate_dataset_card_json
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.notebooks.generator import generate_nwb_starter_notebook
from neural_search.notebooks.templates import (
    evaluate_template_for_dataset,
    get_notebook_template,
)
from neural_search.recipes import get_recipe


GENERATED_DIR = Path(__file__).resolve().parents[2] / "data" / "notebooks" / "generated"


def find_dataset_and_asset(
    records: list[dict],
    dataset_id: str,
    asset_id: str | None = None,
) -> tuple[dict, dict, dict | None]:
    """Find dataset and asset from demo seed records.

    Args:
        records: List of demo seed records.
        dataset_id: Dataset ID to find.
        asset_id: Optional asset ID. If None, uses first asset.

    Returns:
        Tuple of (dataset, asset, card) dicts.

    Raises:
        ValueError: If dataset or asset not found.
    """
    # Find dataset
    record = None
    for candidate in records:
        ds = candidate["dataset"]
        if str(ds.get("source_id")) == dataset_id or str(ds.get("id")) == dataset_id:
            record = candidate
            break

    if record is None:
        available = [r["dataset"].get("source_id", r["dataset"].get("id")) for r in records]
        raise ValueError(f"Dataset '{dataset_id}' not found. Available: {available}")

    dataset = record["dataset"]
    assets = record.get("assets", [])

    if not assets:
        raise ValueError(f"Dataset '{dataset_id}' has no assets.")

    # Find asset
    asset = assets[0]  # Default to first
    if asset_id:
        for candidate_asset in assets:
            if (
                str(candidate_asset.get("id")) == asset_id
                or str(candidate_asset.get("path")) == asset_id
            ):
                asset = candidate_asset
                break
        else:
            available_assets = [a.get("id", a.get("path")) for a in assets]
            raise ValueError(f"Asset '{asset_id}' not found. Available: {available_assets}")

    # Generate card
    card = None
    extraction = record.get("extraction")
    papers = record.get("papers", [])
    if extraction:
        try:
            card = generate_dataset_card_json(dataset, extraction, papers, record.get("assets", []))
        except Exception:
            pass

    return dataset, asset, card


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for notebook generation."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.notebooks.generate",
        description="Generate an NWB starter notebook for a dataset.",
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help="Dataset ID (source_id) to generate notebook for.",
    )
    parser.add_argument(
        "--asset-id",
        default=None,
        help="Asset ID or path. If not specified, uses the first asset.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output path for the notebook. Defaults to data/notebooks/generated/<dataset_id>.ipynb",
    )
    parser.add_argument(
        "--recipe",
        action="append",
        default=[],
        help="Analysis recipe ID to include. Can be provided multiple times.",
    )
    parser.add_argument(
        "--template",
        default="generic_nwb_inspection",
        help="Notebook template ID. Defaults to generic_nwb_inspection.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_datasets",
        help="List available datasets and exit.",
    )
    args = parser.parse_args(argv)

    # Load demo seed
    records = build_demo_seed()

    # List mode
    if args.list_datasets:
        print("Available datasets:")
        for record in records:
            ds = record["dataset"]
            ds_id = ds.get("source_id", ds.get("id"))
            title = ds.get("title", "Untitled")[:50]
            assets = record.get("assets", [])
            print(f"  {ds_id}: {title} ({len(assets)} assets)")
        return 0

    # Require dataset-id if not listing
    if not args.dataset_id:
        parser.error("--dataset-id is required (use --list to see available datasets)")

    # Find dataset and asset
    try:
        dataset, asset, card = find_dataset_and_asset(records, args.dataset_id, args.asset_id)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Determine output path
    if args.output_path:
        output_path = Path(args.output_path)
    else:
        GENERATED_DIR.mkdir(parents=True, exist_ok=True)
        safe_id = args.dataset_id.replace("/", "_").replace(":", "_")
        output_path = GENERATED_DIR / f"{safe_id}.ipynb"

    # Select notebook template and associated recipes
    template = get_notebook_template(args.template)
    if template is None:
        print(f"Error: template '{args.template}' not found")
        return 1
    template_status = evaluate_template_for_dataset(template, dataset)

    recipes = []
    recipe_ids = [*template.get("recipes", []), *args.recipe]
    for recipe_id in dict.fromkeys(recipe_ids):
        recipe = get_recipe(recipe_id)
        if recipe is None:
            print(f"Error: recipe '{recipe_id}' not found")
            return 1
        recipes.append(recipe)

    response = generate_nwb_starter_notebook(
        dataset,
        asset,
        output_path,
        card=card,
        recipes=recipes,
        notebook_template=template,
        template_warnings=template_status["missing_requirements"],
    )

    # Output result
    print(json.dumps(response.model_dump(mode="json"), indent=2))

    if response.warnings:
        print("\nWarnings:")
        for warning in response.warnings:
            print(f"  - {warning}")

    print(f"\nNotebook written to: {response.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
