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
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_search.coverage.duckdb_store import CoverageStore

TASK_ONTOLOGY = Path("data/ontology/behavioral_task_ontology.yaml")
CONFIDENCE = 0.60


def _build_alias_map(task_path: Path = TASK_ONTOLOGY) -> dict[str, str]:
    """Return {normalized_alias: task_id} for all tasks."""
    with task_path.open() as f:
        data = yaml.safe_load(f)
    alias_map: dict[str, str] = {}
    for task in data.get("tasks", []):
        task_id = task["id"]
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
            if len(normalized) >= 4:
                alias_map[normalized] = task_id
    return alias_map


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _match_tasks(text: str, alias_map: dict[str, str]) -> list[tuple[str, float]]:
    """Return (task_id, confidence) for all aliases found in text."""
    normalized = _normalize_text(text)
    found: dict[str, int] = {}
    for alias, task_id in alias_map.items():
        pattern = r"\b" + re.escape(alias) + r"\b"
        if re.search(pattern, normalized):
            found[task_id] = max(found.get(task_id, 0), len(alias))
    return [(task_id, CONFIDENCE) for task_id in found]


def main() -> int:
    parser = argparse.ArgumentParser(description="NLP task coverage enrichment")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to DB")
    parser.add_argument("--limit", type=int, default=0, help="Max datasets to process")
    args = parser.parse_args()

    alias_map = _build_alias_map()
    print(f"Loaded {len(alias_map)} task aliases from {len(set(alias_map.values()))} tasks")

    store = CoverageStore()

    uncovered = store._conn.execute("""
        SELECT d.dataset_id, d.title
        FROM datasets d
        WHERE d.dataset_id NOT IN (
            SELECT DISTINCT dataset_id FROM coverage_entries WHERE dimension = 'tasks'
        )
        ORDER BY d.dataset_id
    """).fetchall()

    if args.limit > 0:
        uncovered = uncovered[: args.limit]

    print(f"Datasets without task coverage: {len(uncovered)}")

    entries: list[dict] = []
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

    print(f"Datasets matched: {matched_count}/{len(uncovered)}")
    print(f"New task entries to insert: {len(entries)}")

    if args.dry_run:
        print("[dry-run] No changes written.")
    else:
        inserted = store.add_coverage_entries_batch(entries)
        print(f"Inserted: {inserted} new entries")

    store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
