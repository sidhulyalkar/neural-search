#!/usr/bin/env python3
"""NLP task enrichment: second-pass match against expanded task aliases.

Usage:
    PYTHONPATH=. python scripts/enrich_task_coverage.py [--dry-run] [--limit N]

For each dataset with no task coverage, matches its title + description against
all task aliases in behavioral_task_ontology.yaml and inserts matches with
confidence=0.60 (marked silver/inferred).
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_search.coverage.duckdb_store import CoverageStore

TASK_ONTOLOGY = Path("data/ontology/behavioral_task_ontology.yaml")
CONFIDENCE = 0.60
_MIN_ALIAS_LEN = 4  # discard very short strings that cause too many false matches

log = logging.getLogger(__name__)


def _build_alias_map(task_path: Path = TASK_ONTOLOGY) -> dict[str, str]:
    """Return {normalized_alias: task_id} for all tasks."""
    with task_path.open() as f:
        data = yaml.safe_load(f)
    alias_map: dict[str, str] = {}
    for task in data.get("tasks", []):
        task_id = task.get("id", "")
        if not task_id:
            log.warning("Skipping task entry with missing id: %s", task)
            continue
        candidates = [
            task.get("label", ""),
            task_id.replace("_", " "),
        ]
        for key in ("aliases", "synonyms"):
            candidates.extend(task.get(key, []))
        for alias in candidates:
            if not alias:
                continue
            normalized = _normalize_text(str(alias)).strip()
            if len(normalized) < _MIN_ALIAS_LEN:
                continue
            if normalized in alias_map and alias_map[normalized] != task_id:
                log.warning(
                    "Alias collision: %r maps to both %s and %s; keeping %s",
                    normalized,
                    alias_map[normalized],
                    task_id,
                    alias_map[normalized],
                )
            else:
                alias_map[normalized] = task_id
    return alias_map


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _match_tasks(text: str, alias_map: dict[str, str]) -> list[tuple[str, float]]:
    """Return (task_id, confidence) for all aliases found in text."""
    normalized = _normalize_text(text)
    found: set[str] = set()
    for alias, task_id in alias_map.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, normalized):
            found.add(task_id)
    return [(task_id, CONFIDENCE) for task_id in found]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="NLP task coverage enrichment")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--limit", type=int, default=0, help="Max datasets to process")
    args = parser.parse_args()

    alias_map = _build_alias_map()
    log.info("Loaded %d task aliases from %d tasks", len(alias_map), len(set(alias_map.values())))

    with CoverageStore() as store:
        uncovered = store.get_uncovered_datasets(dimension="tasks")

        if args.limit > 0:
            uncovered = uncovered[: args.limit]

        log.info("Datasets without task coverage: %d", len(uncovered))

        entries: list[dict[str, Any]] = []
        matched_count = 0

        for dataset_id, title in uncovered:
            text = title or ""
            matches = _match_tasks(text, alias_map)
            for task_id, conf in matches:
                entries.append({
                    "dataset_id": dataset_id,
                    "dimension": "tasks",
                    "value_id": task_id,
                    "confidence": conf,
                    "provenance": "nlp_enrichment_v1",
                })
            if matches:
                matched_count += 1

        log.info("Datasets matched: %d/%d", matched_count, len(uncovered))
        log.info("New task entries to insert: %d", len(entries))

        if args.dry_run:
            log.info("[dry-run] No changes written.")
        else:
            inserted = store.add_coverage_entries_batch(entries)
            log.info("Inserted: %d new entries", inserted)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
