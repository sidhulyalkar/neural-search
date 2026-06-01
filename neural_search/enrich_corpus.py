"""CLI for enriching normalized corpus records with labels and affordances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_search.analysis_affordances import enrich_record_with_affordances
from neural_search.normalized import load_normalized_records, write_jsonl
from neural_search.schemas import NormalizedDatasetRecord
from neural_search.scientific_labels import enrich_record_with_scientific_labels


def enrich_records(input_path: str | Path) -> list[object]:
    """Load normalized records and attach scientific labels and affordances."""

    enriched = []
    for record in load_normalized_records(input_path):
        labeled = enrich_record_with_scientific_labels(record)
        if isinstance(labeled, NormalizedDatasetRecord):
            labeled = enrich_record_with_affordances(labeled)
        enriched.append(labeled)
    return enriched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m neural_search.enrich_corpus")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    output = args.out
    if output.suffix != ".jsonl":
        output.mkdir(parents=True, exist_ok=True)
        output = output / "normalized_enriched.jsonl"
    records = enrich_records(args.input)
    write_jsonl(records, output)
    print(json.dumps({"records": len(records), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
