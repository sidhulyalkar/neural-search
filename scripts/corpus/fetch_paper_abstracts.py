"""CrossRef paper abstract mining for corpus enrichment.

For corpus records without brain_regions, extracts DOIs from their descriptions
and URLs, fetches abstracts from CrossRef (free API), then runs rule-based
region extraction on the abstract text.

Focuses on DANDI records first — ephys/imaging papers name specific regions in
their abstracts. OpenNeuro whole-brain fMRI abstracts rarely name specific
regions and are skipped by default.

Provenance: "crossref_abstract_rule_based_silver_not_human_gold"

Usage:
    python scripts/corpus/fetch_paper_abstracts.py [--dry-run] [--limit N]
    python scripts/corpus/fetch_paper_abstracts.py --source openneuro
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("paper_abstract_miner")

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")
ABSTRACT_CACHE_PATH = Path("data/raw/paper_abstract_cache.jsonl")

_PROVENANCE = "crossref_abstract_rule_based_silver_not_human_gold"

# JATS XML tag stripper
_JATS_TAG_RE = re.compile(r"<[^>]+>")

# CrossRef polite pool requires an email in User-Agent
_CROSSREF_HEADERS = {
    "User-Agent": "neural-search-enrichment/1.0 (sid.soccer.21@gmail.com)",
}


def _load_abstract_cache() -> dict[str, str | None]:
    """doi → abstract text (None means 'fetched but no abstract found')."""
    cache: dict[str, str | None] = {}
    if ABSTRACT_CACHE_PATH.exists():
        for line in ABSTRACT_CACHE_PATH.read_text().splitlines():
            if line.strip():
                try:
                    entry = json.loads(line)
                    cache[entry["doi"]] = entry.get("abstract")
                except Exception:
                    pass
    return cache


def _save_abstract_cache(doi: str, abstract: str | None) -> None:
    with ABSTRACT_CACHE_PATH.open("a") as f:
        f.write(json.dumps({"doi": doi, "abstract": abstract}) + "\n")


def _strip_jats(text: str) -> str:
    """Remove JATS XML tags from CrossRef abstract."""
    return _JATS_TAG_RE.sub(" ", text).strip()


def _fetch_crossref_abstract(doi: str, client: httpx.Client) -> str | None:
    """Fetch abstract from CrossRef for a given DOI. Returns None on failure."""
    url = f"https://api.crossref.org/works/{doi}"
    try:
        resp = client.get(url, headers=_CROSSREF_HEADERS, timeout=15.0)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        msg = resp.json().get("message", {})
        raw_abstract = msg.get("abstract")
        if not raw_abstract:
            return None
        abstract = _strip_jats(raw_abstract)
        return abstract if len(abstract) > 30 else None
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("CrossRef rate limit — sleeping 10s")
            time.sleep(10)
        else:
            logger.debug("CrossRef HTTP %s for doi:%s", e.response.status_code, doi)
        return None
    except Exception as e:
        logger.debug("CrossRef fetch failed for doi:%s — %s", doi, e)
        return None


def _extract_dois_from_record(rec: dict) -> list[str]:
    """Extract DOIs from a corpus record's description, URL, and metadata."""
    from neural_search.ingestion.doi_utils import (
        extract_dois_from_dandi_metadata,
        extract_dois_from_openneuro_metadata,
        extract_dois_from_text,
    )
    found: list[str] = []
    seen: set[str] = set()

    def _add(doi: str) -> None:
        if doi and doi not in seen:
            seen.add(doi)
            found.append(doi)

    source = rec.get("source", "")
    description = rec.get("description") or ""
    url = rec.get("url") or ""

    # Source-specific structured extraction
    metadata = rec.get("metadata_json") or {}
    if source == "dandi":
        for doi in extract_dois_from_dandi_metadata(metadata):
            _add(doi)
    elif source == "openneuro":
        for doi in extract_dois_from_openneuro_metadata(metadata):
            _add(doi)

    # Fallback: text scan on description and URL
    for doi in extract_dois_from_text(description):
        _add(doi)
    for doi in extract_dois_from_text(url):
        _add(doi)

    return found


def _canonical_ids() -> frozenset[str]:
    from neural_search.ontology.loader import get_brain_regions
    return frozenset(r.id for r in get_brain_regions())


def _filter_canonical(regions: list[str]) -> list[str]:
    valid = _canonical_ids()
    return [r for r in regions if r in valid]


def _flatten_ids(br: list) -> list[str]:
    return [(v.get("id") if isinstance(v, dict) else v) for v in (br or []) if v]


def _has_region(record: dict) -> bool:
    return any(_flatten_ids(record.get("brain_regions") or []))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", choices=["dandi", "openneuro", "all"], default="dandi",
                        help="Which source to target (default: dandi, as fMRI abstracts rarely name regions)")
    args = parser.parse_args(argv)

    corpus = [json.loads(line) for line in CORPUS_PATH.read_text().strip().splitlines() if line.strip()]
    abstract_cache = _load_abstract_cache()

    targets = [
        r for r in corpus
        if not _has_region(r)
        and (args.source == "all" or r.get("source") == args.source)
    ]
    logger.info("Records without regions (%s): %d", args.source, len(targets))
    if args.limit:
        targets = targets[: args.limit]

    from neural_search.extraction import extract_dataset_labels

    enriched_map: dict[str, list[str]] = {}
    dois_fetched = 0
    abstracts_found = 0

    with httpx.Client(follow_redirects=True) as client:
        for i, rec in enumerate(targets, 1):
            sid = rec["source_id"]
            dois = _extract_dois_from_record(rec)
            if not dois:
                continue

            logger.info("[%d/%d] %s: DOIs %s", i, len(targets), sid, dois[:2])
            dois_fetched += 1

            for doi in dois:
                # Check cache
                if doi in abstract_cache:
                    abstract = abstract_cache[doi]
                else:
                    abstract = _fetch_crossref_abstract(doi, client)
                    _save_abstract_cache(doi, abstract)
                    time.sleep(0.1)  # polite rate limiting

                if not abstract:
                    continue

                abstracts_found += 1
                logger.info("  doi:%s abstract (%d chars)", doi, len(abstract))

                result = extract_dataset_labels(
                    title=rec.get("title"),
                    description=rec.get("description"),
                    linked_paper_abstracts=[abstract],
                )
                new_regions = _filter_canonical([item.id for item in result.brain_regions])
                if new_regions:
                    logger.info("  → regions: %s", new_regions)
                    existing = _flatten_ids(rec.get("brain_regions") or [])
                    enriched_map[sid] = list(dict.fromkeys(existing + new_regions))
                    break  # One abstract with regions is enough

    logger.info("Records with DOIs: %d, abstracts found: %d, regions extracted: %d",
                dois_fetched, abstracts_found, len(enriched_map))

    if args.dry_run:
        logger.info("[dry-run] skipping corpus write")
        return 0

    if not enriched_map:
        logger.info("No new regions — corpus unchanged")
        return 0

    output = []
    updated = 0
    for rec in corpus:
        sid = rec.get("source_id", "")
        new_regions = enriched_map.get(sid)
        if new_regions:
            rec = {**rec, "brain_regions": new_regions, "brain_regions_provenance": _PROVENANCE}
            updated += 1
        output.append(json.dumps(rec))

    CORPUS_PATH.write_text("\n".join(output) + "\n")
    logger.info("Updated %d records → %s", updated, CORPUS_PATH)

    refreshed = [json.loads(line) for line in CORPUS_PATH.read_text().strip().splitlines() if line.strip()]
    total = len(refreshed)
    with_region = sum(1 for r in refreshed if _has_region(r))
    logger.info("Brain region coverage: %d/%d = %d%%", with_region, total,
                round(100 * with_region / total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
