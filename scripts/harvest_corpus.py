"""Central corpus harvest orchestrator.

Runs all registered ingestion adapters, writes per-source JSONL checkpoints,
then deduplicates into combined_corpus.jsonl.

Usage:
    python scripts/harvest_corpus.py [--dry-run] [--sources dandi openneuro ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS_DIR = Path("data/corpus/normalized")

SOURCE_OUTPUTS: dict[str, Path] = {
    # Tier 1 - curated neuroscience archives
    "dandi": CORPUS_DIR / "real_dandi.jsonl",
    "openneuro": CORPUS_DIR / "real_openneuro.jsonl",
    "bluebrain": CORPUS_DIR / "real_bluebrain.jsonl",
    "brain_image_library": CORPUS_DIR / "real_brain_image_library.jsonl",
    "allen": CORPUS_DIR / "real_allen.jsonl",
    "nemo": CORPUS_DIR / "real_nemo.jsonl",
    "ibl": CORPUS_DIR / "real_ibl.jsonl",
    "crcns": CORPUS_DIR / "real_crcns.jsonl",
    # Tier 2 - broader neuroscience repositories
    "neurovault": CORPUS_DIR / "real_neurovault.jsonl",
    "gin": CORPUS_DIR / "real_gin.jsonl",
    "zenodo": CORPUS_DIR / "real_zenodo.jsonl",
    "physionet": CORPUS_DIR / "real_physionet.jsonl",
    # Tier 2 - morphology database
    "neuromorpho": CORPUS_DIR / "real_neuromorpho.jsonl",
    # Tier 3 - general repositories (low yield, staged for review)
    "ebrains": CORPUS_DIR / "real_ebrains.jsonl",
    "osf": CORPUS_DIR / "real_osf.jsonl",
    "figshare": CORPUS_DIR / "real_figshare.jsonl",
    "harvard_dataverse": CORPUS_DIR / "real_harvard_dataverse.jsonl",
}

COMBINED_OUTPUT = CORPUS_DIR / "combined_corpus.jsonl"

SOURCE_LIMITS: dict[str, int] = {
    "dandi": 1000,
    "openneuro": 2000,
    "bluebrain": 300,
    "brain_image_library": 300,
    "allen": 500,
    "nemo": 100,
    "ibl": 200,
    "crcns": 300,
    "neurovault": 1500,
    "gin": 700,
    "ebrains": 300,
    "zenodo": 2000,
    "physionet": 200,
    "neuromorpho": 500,
    "osf": 200,
    "figshare": 400,
    "harvard_dataverse": 500,
}


def load_seen_ids(path: Path) -> set[str]:
    """Return set of source_id values already in a JSONL file."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if sid := rec.get("source_id"):
                    seen.add(str(sid))
            except json.JSONDecodeError:
                pass
    return seen


def append_new_records(
    path: Path,
    records: list[dict[str, Any]],
    seen_ids: set[str],
) -> int:
    """Write records whose source_id is not in seen_ids. Returns count added."""
    new = [r for r in records if str(r.get("source_id", "")) not in seen_ids]
    if not new:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in new:
            f.write(json.dumps(rec) + "\n")
    return len(new)


def write_records(path: Path, records: list[dict[str, Any]]) -> int:
    """Replace a source checkpoint with the provided records."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return len(records)


def deduplicate_combined(sources: list[Path], output: Path) -> int:
    """Read all source JSONL files, deduplicate by (source, source_id), write combined."""
    seen: set[tuple[str, str]] = set()
    records: list[dict[str, Any]] = []

    for src in sources:
        if not src.exists():
            continue
        with open(src, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    key = (str(rec.get("source", "")), str(rec.get("source_id", "")))
                    if key[1] and key not in seen:
                        seen.add(key)
                        records.append(rec)
                except json.JSONDecodeError:
                    pass

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return len(records)


def run_harvest(
    sources: list[str],
    dry_run: bool = False,
    refresh: bool = False,
) -> dict[str, int]:
    """Run specified adapters and return {source: new_records_added} mapping."""
    import neural_search.ingestion.dandi  # noqa: F401
    import neural_search.ingestion.openneuro  # noqa: F401
    import neural_search.ingestion.bluebrain  # noqa: F401
    import neural_search.ingestion.brain_image_library  # noqa: F401
    import neural_search.ingestion.allen_brain  # noqa: F401
    import neural_search.ingestion.nemo_archive  # noqa: F401
    import neural_search.ingestion.ibl  # noqa: F401
    import neural_search.ingestion.crcns  # noqa: F401
    import neural_search.ingestion.neurovault  # noqa: F401
    import neural_search.ingestion.gin  # noqa: F401
    import neural_search.ingestion.ebrains  # noqa: F401
    import neural_search.ingestion.zenodo  # noqa: F401
    import neural_search.ingestion.physionet  # noqa: F401
    import neural_search.ingestion.neuromorpho  # noqa: F401
    import neural_search.ingestion.osf  # noqa: F401
    import neural_search.ingestion.figshare  # noqa: F401
    import neural_search.ingestion.harvard_dataverse  # noqa: F401
    from neural_search.ingestion.registry import _REGISTRY, run_adapter  # type: ignore[attr-defined]

    results: dict[str, int] = {}

    for source in sources:
        if source not in _REGISTRY:
            logger.warning("Source '%s' not in registry - skipping", source)
            continue

        output_path = SOURCE_OUTPUTS.get(source, CORPUS_DIR / f"real_{source}.jsonl")
        limit = SOURCE_LIMITS.get(source, 200)
        seen_ids = load_seen_ids(output_path)

        logger.info(
            "Running %s (seen=%d, limit=%d, refresh=%s)...",
            source,
            len(seen_ids),
            limit,
            refresh,
        )

        try:
            records: list[dict[str, Any]] = run_adapter(source, limit=limit)
            logger.info("%s: fetched %d records", source, len(records))
        except Exception as exc:
            logger.error("%s: adapter failed - %s", source, exc)
            results[source] = 0
            continue

        if dry_run:
            count = len(records) if refresh else len([
                r for r in records if str(r.get("source_id", "")) not in seen_ids
            ])
            action = "replace with" if refresh else "add"
            logger.info("[DRY-RUN] %s: would %s %d records", source, action, count)
            results[source] = count
        else:
            if refresh:
                if not records and seen_ids:
                    logger.error(
                        "%s: refresh returned 0 records; keeping existing checkpoint %s",
                        source,
                        output_path,
                    )
                    results[source] = 0
                    continue
                written = write_records(output_path, records)
                logger.info("%s: refreshed %d records in %s", source, written, output_path)
                results[source] = written
            else:
                added = append_new_records(output_path, records, seen_ids)
                logger.info("%s: added %d new records to %s", source, added, output_path)
                results[source] = added

    if not dry_run:
        # Always deduplicate across all known source checkpoint files so the combined
        # corpus accumulates data from every prior run, not just the current sources.
        total = deduplicate_combined(list(SOURCE_OUTPUTS.values()), COMBINED_OUTPUT)
        logger.info("Combined corpus: %d total unique records → %s", total, COMBINED_OUTPUT)

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Harvest neuroscience datasets from all sources")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(SOURCE_OUTPUTS.keys()),
        help="Sources to run (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Replace selected source checkpoint files instead of appending only new IDs",
    )
    args = parser.parse_args(argv)

    results = run_harvest(args.sources, dry_run=args.dry_run, refresh=args.refresh)
    action = "would refresh" if args.dry_run and args.refresh else (
        "refreshed" if args.refresh else ("(dry-run)" if args.dry_run else "added")
    )
    for source, count in results.items():
        print(f"{source}: {count} {action}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
