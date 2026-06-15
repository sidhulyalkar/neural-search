"""Shared helpers for corpus fetch scripts."""
from __future__ import annotations
from functools import lru_cache


@lru_cache(maxsize=1)
def canonical_region_ids() -> frozenset[str]:
    from neural_search.ontology.loader import get_brain_regions
    return frozenset(r.id for r in get_brain_regions())


def filter_regions(regions: list) -> list:
    valid = canonical_region_ids()
    out = []
    for v in (regions or []):
        rid = v.get("id") if isinstance(v, dict) else v
        if rid and rid in valid:
            out.append(rid)
    return out


def has_region(record: dict) -> bool:
    return bool(filter_regions(record.get("brain_regions") or []))


def append_to_corpus(corpus_path, corpus: list, new_records: list, source_name: str,
                     logger, dry_run: bool = False) -> int:
    import json
    from pathlib import Path

    # Apply canonical region filter
    for r in new_records:
        r["brain_regions"] = filter_regions(r.get("brain_regions") or [])

    with_region = sum(1 for r in new_records if has_region(r))
    pct = round(100 * with_region / len(new_records)) if new_records else 0
    logger.info("New records with brain regions: %d/%d = %d%%", with_region, len(new_records), pct)

    for r in new_records[:5]:
        logger.info("  %s | regions=%s | mods=%s",
                    r.get("source_id", "")[:35],
                    (r.get("brain_regions") or [])[:3],
                    (r.get("modalities") or [])[:2])

    if dry_run:
        logger.info("[dry-run] would append %d records", len(new_records))
        return 0

    all_records = corpus + new_records
    Path(corpus_path).write_text("\n".join(json.dumps(r) for r in all_records) + "\n")

    total = len(all_records)
    total_with = sum(1 for r in all_records if has_region(r))
    logger.info("Corpus: %d records (+%d %s)", total, len(new_records), source_name)
    logger.info("Brain region coverage: %d/%d = %d%%", total_with, total,
                round(100 * total_with / total))
    return len(new_records)
