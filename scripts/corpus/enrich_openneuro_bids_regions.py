"""Enrich OpenNeuro corpus records with brain regions from BIDS electrode files.

OpenNeuro stores iEEG/sEEG/ECoG electrode coordinates and anatomical labels in
BIDS `*_electrodes.tsv` files. This script:
  1. Finds OpenNeuro datasets in corpus with empty brain_regions
  2. Fetches electrode TSV files via the OpenNeuro BIDS file listing API
  3. Parses the `name` and `group` columns for region labels
  4. Maps to canonical region IDs via the ontology matcher
  5. Updates corpus records in place

For fMRI datasets, also parses `*_events.json` task descriptions and
`dataset_description.json` for BIDSVersion and task context.

Rate limit: 0.3s between requests to avoid hammering OpenNeuro API.

Usage
-----
    python scripts/corpus/enrich_openneuro_bids_regions.py [--dry-run] [--limit N]

    # Debug one dataset
    python scripts/corpus/enrich_openneuro_bids_regions.py --accession ds003505

    # Only process datasets that have no regions
    python scripts/corpus/enrich_openneuro_bids_regions.py --only-empty
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("openneuro_bids_regions")

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")
OPENNEURO_API = "https://openneuro.org/crn/datasets"
RATE_LIMIT_DELAY = 0.3

_ELECTRODE_REGION_COLUMNS = ("group", "name", "hemisphere", "destrieux_label", "fs_label")


def _get(url: str, timeout: int = 10) -> bytes | None:
    req = request.Request(url, headers={"Accept": "application/json, text/plain"})
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except error.URLError as e:
        log.debug("GET %s failed: %s", url, e)
        return None


def _list_bids_files(accession: str) -> list[dict[str, Any]]:
    """Fetch the file tree for an OpenNeuro dataset (top-level only)."""
    url = f"{OPENNEURO_API}/{accession}/files"
    data = _get(url)
    if not data:
        return []
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return []


def _download_tsv(accession: str, path: str) -> list[dict[str, str]]:
    """Download a BIDS TSV file and return as list of row dicts."""
    encoded_path = path.replace("/", "%2F")
    url = f"{OPENNEURO_API}/{accession}/files/{encoded_path}"
    data = _get(url, timeout=15)
    if not data:
        return []
    try:
        text = data.decode("utf-8", errors="replace")
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        return [dict(row) for row in reader]
    except Exception as e:
        log.debug("TSV parse failed %s: %s", path, e)
        return []


def _find_electrode_files(file_tree: list[dict]) -> list[str]:
    """Return paths to *_electrodes.tsv files in a BIDS file tree."""
    paths: list[str] = []

    def _walk(nodes: list[dict], prefix: str = "") -> None:
        for node in nodes:
            name = node.get("filename", "")
            full = f"{prefix}/{name}".lstrip("/")
            if name.endswith("_electrodes.tsv"):
                paths.append(full)
            children = node.get("files") or node.get("children") or []
            if children:
                _walk(children, full.rsplit("/", 1)[0] if "/" in full else full)

    _walk(file_tree)
    return paths


def _extract_regions_from_electrodes(rows: list[dict[str, str]]) -> list[str]:
    """Extract brain region text from electrode TSV rows."""
    raw_values: set[str] = set()
    for row in rows:
        for col in _ELECTRODE_REGION_COLUMNS:
            val = str(row.get(col) or "").strip()
            if val and val.lower() not in {"n/a", "na", "none", "", "unknown"}:
                raw_values.add(val.lower().replace("-", "_").replace(" ", "_"))
    return sorted(raw_values)


def _match_regions(raw_values: list[str]) -> list[str]:
    """Map raw region strings to canonical ontology IDs."""
    try:
        from neural_search.ontology import match_brain_regions
        haystack = " ".join(raw_values)
        matches = match_brain_regions(haystack)
        return sorted({m.id for m in matches})
    except Exception as e:
        log.debug("Ontology match failed: %s", e)
        return []


def enrich_one(accession: str) -> list[str]:
    """Return canonical brain region IDs for one OpenNeuro dataset."""
    log.debug("Fetching file tree: %s", accession)
    file_tree = _list_bids_files(accession)
    time.sleep(RATE_LIMIT_DELAY)

    electrode_files = _find_electrode_files(file_tree)
    if not electrode_files:
        log.debug("No electrode files: %s", accession)
        return []

    all_raw: list[str] = []
    for path in electrode_files[:3]:  # sample up to 3 files
        rows = _download_tsv(accession, path)
        raw = _extract_regions_from_electrodes(rows)
        all_raw.extend(raw)
        time.sleep(RATE_LIMIT_DELAY)

    if not all_raw:
        return []

    regions = _match_regions(list(dict.fromkeys(all_raw)))
    log.debug("%s: electrode regions → %s", accession, regions)
    return regions


def load_corpus(path: Path) -> list[dict]:
    if not path.exists():
        log.error("Corpus not found: %s", path)
        return []
    records = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    log.info("Loaded %d records from %s", len(records), path)
    return records


def save_corpus(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    log.info("Saved %d records → %s", len(records), path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--limit", type=int, default=200,
                        help="Max datasets to process")
    parser.add_argument("--only-empty", action="store_true",
                        help="Only process datasets with empty brain_regions")
    parser.add_argument("--accession", type=str, default=None,
                        help="Debug: process a single dataset by accession ID")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.accession:
        regions = enrich_one(args.accession)
        print(f"{args.accession}: {regions}")
        return 0

    records = load_corpus(args.corpus)
    openneuro = [
        r for r in records
        if str(r.get("source", "")).lower() == "openneuro"
    ]
    log.info("OpenNeuro records: %d", len(openneuro))

    if args.only_empty:
        openneuro = [r for r in openneuro if not r.get("brain_regions")]
        log.info("Without brain_regions: %d", len(openneuro))

    openneuro = openneuro[: args.limit]

    n_enriched = 0
    n_skipped = 0
    for i, rec in enumerate(openneuro):
        accession = str(rec.get("source_id") or rec.get("id") or "")
        if not accession:
            n_skipped += 1
            continue

        log.info("[%d/%d] %s", i + 1, len(openneuro), accession)
        if args.dry_run:
            log.info("  (dry run — would fetch electrodes)")
            continue

        try:
            new_regions = enrich_one(accession)
        except Exception as e:
            log.warning("Failed %s: %s", accession, e)
            continue

        if not new_regions:
            n_skipped += 1
            continue

        existing = list(rec.get("brain_regions") or [])
        merged = list(dict.fromkeys(existing + new_regions))
        if merged != existing:
            rec["brain_regions"] = merged
            n_enriched += 1
            log.info("  → %s", merged)

    log.info("Enriched: %d | Skipped/no-electrode: %d", n_enriched, n_skipped)

    if not args.dry_run and n_enriched > 0:
        save_corpus(records, args.corpus)
        log.info("Corpus updated")

    return 0


if __name__ == "__main__":
    sys.exit(main())
