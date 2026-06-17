#!/usr/bin/env python3
"""CLI: link corpus datasets to OpenAlex papers via DOI/title matching."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.literature.linking import (
    DatasetPaperLink,
    _iter_corpus_records,
    _resolve_link,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link corpus datasets to OpenAlex papers."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Path to corpus JSONL file or directory of JSONL shards.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL path for DatasetPaperLink records.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit number of corpus records to process.",
    )
    parser.add_argument(
        "--skip-without-doi",
        action="store_true",
        help="Skip records that have no DOI (no title-based fallback).",
    )
    return parser.parse_args()


def _summarize(links: list[DatasetPaperLink]) -> None:
    by_method: dict[str, int] = {}
    for link in links:
        by_method[link.match_method] = by_method.get(link.match_method, 0) + 1
    print(f"\nSummary ({len(links)} records processed):")
    for method, count in sorted(by_method.items()):
        print(f"  {method}: {count}")


def main() -> None:
    args = _parse_args()
    corpus_path: Path = args.corpus
    out_path: Path = args.out

    if not corpus_path.exists():
        print(f"ERROR: corpus path does not exist: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = _iter_corpus_records(corpus_path)
    if args.max_records is not None:
        records = itertools.islice(records, args.max_records)

    links: list[DatasetPaperLink] = []
    import dataclasses

    with out_path.open("w") as out_fh:
        for i, rec in enumerate(records, start=1):
            record_id = f"{rec['source']}:{rec['source_id']}"
            doi = rec.get("doi") or None
            title = rec.get("title") or ""

            link = _resolve_link(record_id, doi, title, args.skip_without_doi)
            out_fh.write(json.dumps(dataclasses.asdict(link)) + "\n")
            links.append(link)

            if i % 50 == 0:
                print(f"  Processed {i} records...")

    _summarize(links)
    print(f"Output written to: {out_path}")


if __name__ == "__main__":
    main()
