#!/usr/bin/env python3
"""Check retraction/correction status for every DOI resolved by the
literature-linking sources, via Crossref's `update-to` field.

Selects unique paper DOIs across all `paper_dataset_links*.jsonl` files
(OpenAlex, DataCite, Crossref, Semantic Scholar, PubMed/bioRxiv) with a real
match, checks each via `neural_search.literature.crossref.fetch_retraction_status`,
and writes results to `artifacts/literature/paper_retraction_status.jsonl` --
a durable artifact consumed by `neural_search.graph.paper_node_builder.attach_retraction_status`
on the next production graph build. This script itself is not run on every
build (it's a live-network, Crossref-rate-limited pass), matching the same
pattern as `scripts/validate_top_reanalysis_suggestions.py`.

Run: python scripts/check_paper_retraction_status.py [--max-dois N]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from neural_search.literature.api_client import TransientLookupError
from neural_search.literature.crossref import fetch_retraction_status

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
LITERATURE_DIR = PROJECT_ROOT / "artifacts" / "literature"
LINK_FILES = [
    LITERATURE_DIR / "paper_dataset_links.jsonl",
    LITERATURE_DIR / "paper_dataset_links.datacite.jsonl",
    LITERATURE_DIR / "paper_dataset_links.crossref.jsonl",
    LITERATURE_DIR / "paper_dataset_links.semantic_scholar.jsonl",
    LITERATURE_DIR / "paper_dataset_links.pubmed.jsonl",
]
OUT_PATH = LITERATURE_DIR / "paper_retraction_status.jsonl"
_NOT_REAL_MATCH_METHODS = {"not_found", "not_applicable_no_dataset_doi"}


def collect_unique_dois() -> list[str]:
    dois: set[str] = set()
    for path in LINK_FILES:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row.get("match_method") in _NOT_REAL_MATCH_METHODS:
                    continue
                doi = row.get("paper_doi")
                if doi:
                    dois.add(doi)
    return sorted(dois)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-dois", type=int, default=0, help="Limit to N DOIs (0 = no limit)")
    args = parser.parse_args(argv)

    dois = collect_unique_dois()
    if args.max_dois:
        dois = dois[: args.max_dois]
    print(f"Checking retraction status for {len(dois)} unique DOIs...")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checked = 0
    retracted = 0
    corrected = 0
    with OUT_PATH.open("w", encoding="utf-8") as out_fh:
        for i, doi in enumerate(dois, start=1):
            try:
                status = fetch_retraction_status(doi)
            except TransientLookupError as exc:
                print(
                    f"Aborting after {i - 1}/{len(dois)} DOIs: transient Crossref "
                    f"lookup failure ({exc}). Results so far are saved; re-run "
                    f"later rather than trusting the remaining DOIs as genuinely "
                    f"unretracted."
                )
                break
            status["doi"] = doi
            out_fh.write(json.dumps(status) + "\n")
            checked += 1
            if status["status"] == "retracted":
                retracted += 1
            elif status["status"] == "corrected":
                corrected += 1
            if i % 50 == 0:
                print(f"  Checked {i}/{len(dois)}...")

    print(f"Checked {checked}/{len(dois)} DOIs: {retracted} retracted, {corrected} corrected.")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
