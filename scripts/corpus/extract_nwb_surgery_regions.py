"""Extract brain region labels from NWB surgery/experiment_description fields.

Reads free-text NWB metadata fields that often name recording locations:
  - general/surgery           e.g. "craniotomy over left somatosensory cortex"
  - general/experiment_description e.g. "Single-unit recordings from CA1"
  - general/notes             fallback free-text field

This is complementary to extract_nwb_electrode_regions.py (which reads the
structured electrode table). Surgery text applies to ALL DANDI records, not
just ephys, so this catches calcium imaging and opto experiments too.

All extracted labels are silver-tier with provenance tracking.

Usage:
    python scripts/corpus/extract_nwb_surgery_regions.py [--dry-run] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("nwb_surgery_extractor")

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")
SURGERY_CACHE_PATH = Path("data/raw/dandi/nwb_surgery_cache.jsonl")

_PROVENANCE = "nwb_surgery_text_silver_not_human_gold"

# NWB fields to read, in priority order
_NWB_TEXT_FIELDS = [
    "general/surgery",
    "general/experiment_description",
    "general/notes",
    "general/session_description",  # sometimes has recording site info
]


def _load_surgery_cache() -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    if SURGERY_CACHE_PATH.exists():
        for line in SURGERY_CACHE_PATH.read_text().splitlines():
            if line.strip():
                try:
                    entry = json.loads(line)
                    if "source_id" in entry:
                        cache[entry["source_id"]] = entry.get("regions", [])
                except Exception:
                    pass
    return cache


def _save_surgery_cache(source_id: str, regions: list[str]) -> None:
    with SURGERY_CACHE_PATH.open("a") as f:
        f.write(json.dumps({"source_id": source_id, "regions": regions}) + "\n")


def _canonical_ids() -> frozenset[str]:
    """Return the set of valid region IDs from the YAML ontology."""
    from neural_search.ontology.loader import get_brain_regions
    return frozenset(r.id for r in get_brain_regions())


def _filter_canonical(regions: list[str]) -> list[str]:
    valid = _canonical_ids()
    filtered = [r for r in regions if r in valid]
    dropped = [r for r in regions if r not in valid]
    if dropped:
        logger.debug("Dropped non-canonical region IDs: %s", dropped)
    return filtered


def _flatten_ids(br: list) -> list[str]:
    return [(v.get("id") if isinstance(v, dict) else v) for v in (br or []) if v]


def _has_region(record: dict) -> bool:
    return any(_flatten_ids(record.get("brain_regions") or []))


def _decode_h5_scalar(val) -> str:
    """Decode an h5py scalar dataset value to a plain string."""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    if isinstance(val, (list, tuple)) and val:
        first = val[0]
        return first.decode("utf-8", errors="replace") if isinstance(first, bytes) else str(first)
    return str(val)


def _extract_text_regions(source_id: str) -> list[str]:
    """Stream one NWB asset and read free-text metadata fields for region names.

    Returns canonical region IDs from the ontology, or [] on failure.
    """
    try:
        import h5py
        import remfile
        from dandi.dandiapi import DandiAPIClient

        from neural_search.extraction import extract_dataset_labels

        with DandiAPIClient() as client:
            dandiset = client.get_dandiset(source_id, "draft")
            assets = list(dandiset.get_assets_by_glob("*.nwb"))
            if not assets:
                assets = [a for a in dandiset.get_assets() if a.path.endswith(".nwb")]
            if not assets:
                logger.debug("%s: no NWB assets", source_id)
                return []

            asset = assets[0]
            url = asset.get_content_url(follow_redirects=1, strip_query=True)
            logger.info("  %s: streaming %s", source_id, asset.path)

            rfile = remfile.File(url)
            collected_text: list[str] = []
            with h5py.File(rfile, "r") as f:
                for field in _NWB_TEXT_FIELDS:
                    try:
                        if field in f:
                            raw = f[field][()]
                            text = _decode_h5_scalar(raw)
                            if text and len(text.strip()) > 4:
                                collected_text.append(text.strip())
                                logger.debug("  %s [%s]: %s…", source_id, field, text[:80])
                    except Exception:
                        pass

            if not collected_text:
                logger.debug("%s: no text fields found", source_id)
                return []

            combined = " ".join(collected_text)
            result = extract_dataset_labels(
                title=None,
                description=combined,
                file_paths=[],
                source_metadata={},
                linked_paper_abstracts=[],
            )
            regions = _filter_canonical([item.id for item in result.brain_regions])
            if regions:
                logger.info("  %s: text regions → %s", source_id, regions)
            return regions

    except ImportError as e:
        logger.warning("Missing dependency (%s); skipping surgery extraction", e)
        return []
    except Exception as e:
        logger.debug("%s: NWB surgery extraction failed: %s", source_id, type(e).__name__)
        return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    corpus = [json.loads(line) for line in CORPUS_PATH.read_text().strip().splitlines() if line.strip()]
    surgery_cache = _load_surgery_cache()

    targets = [
        r for r in corpus
        if r.get("source") == "dandi"
        and not _has_region(r)
    ]
    logger.info("DANDI records without brain regions: %d", len(targets))
    if args.limit:
        targets = targets[: args.limit]

    enriched_map: dict[str, list[str]] = {}

    for i, rec in enumerate(targets, 1):
        sid = rec["source_id"]
        if sid in surgery_cache:
            regions = _filter_canonical(surgery_cache[sid])
            if regions:
                enriched_map[sid] = regions
            continue

        logger.info("[%d/%d] Extracting surgery text for %s: %s", i, len(targets), sid,
                    (rec.get("title") or "")[:60])
        regions = _extract_text_regions(sid)

        _save_surgery_cache(sid, regions)
        if regions:
            enriched_map[sid] = regions

        time.sleep(0.5)

    logger.info("Records enriched via surgery text: %d", len(enriched_map))

    if args.dry_run:
        logger.info("[dry-run] skipping corpus write")
        return 0

    output = []
    updated = 0
    for rec in corpus:
        sid = rec.get("source_id", "")
        new_regions = enriched_map.get(sid)
        if new_regions:
            existing = _flatten_ids(rec.get("brain_regions") or [])
            merged = list(dict.fromkeys(existing + new_regions))
            rec = {**rec, "brain_regions": merged, "brain_regions_provenance": _PROVENANCE}
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
