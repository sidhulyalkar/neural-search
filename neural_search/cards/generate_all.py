"""Demo CLI for generating cards from the built-in seed datasets."""

from __future__ import annotations

import json

from neural_search.cards import generate_dataset_card_json
from neural_search.ingestion.demo_seed import build_demo_seed


def main() -> int:
    records = build_demo_seed()
    cards = [
        generate_dataset_card_json(record["dataset"], record["extraction"], record["papers"]).model_dump(
            mode="json"
        )
        for record in records
    ]
    print(json.dumps(cards, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

