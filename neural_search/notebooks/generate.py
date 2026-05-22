"""CLI for generating a demo NWB starter notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.notebooks import generate_nwb_starter_notebook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.notebooks.generate")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--output-path", default="data/notebooks/demo_starter.ipynb")
    args = parser.parse_args(argv)

    records = build_demo_seed()
    record = records[0]
    asset = record["assets"][0]
    for candidate in records:
        if str(candidate["dataset"].get("source_id")) == args.dataset_id or str(
            candidate["dataset"].get("id")
        ) == args.dataset_id:
            record = candidate
            break
    for candidate_asset in record["assets"]:
        if str(candidate_asset.get("id")) == args.asset_id or str(
            candidate_asset.get("path")
        ) == args.asset_id:
            asset = candidate_asset
            break

    response = generate_nwb_starter_notebook(record["dataset"], asset, Path(args.output_path))
    print(json.dumps(response.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

