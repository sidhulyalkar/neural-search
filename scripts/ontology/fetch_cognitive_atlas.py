"""Fetch the Cognitive Atlas task list and build a task_atlas.yaml bridge.

Maps our internal behavioral task IDs (behavioral_task_ontology.yaml) to
stable Cognitive Atlas identifiers via fuzzy label matching.

Output: data/ontology/task_atlas.yaml

Usage:
  python scripts/ontology/fetch_cognitive_atlas.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
TASK_ONTOLOGY_PATH = ROOT / "data" / "ontology" / "behavioral_task_ontology.yaml"
TASK_ATLAS_PATH = ROOT / "data" / "ontology" / "task_atlas.yaml"
COGAT_CACHE_PATH = ROOT / "data" / "ontology" / "cogat_tasks_cache.json"

COGAT_API = "https://www.cognitiveatlas.org/api/v-alpha/task"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("cogat_fetch")


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _fetch_cogat_tasks() -> list[dict]:
    if COGAT_CACHE_PATH.exists():
        log.info("Loading Cognitive Atlas tasks from cache")
        return json.loads(COGAT_CACHE_PATH.read_text())

    log.info("Fetching Cognitive Atlas task list…")
    # The CogAt API returns ALL tasks on every page — it doesn't actually paginate.
    # Fetch once, deduplicate by ID.
    try:
        resp = requests.get(
            COGAT_API,
            params={"format": "json", "count": 1000},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        log.error("CogAt API error: %s", exc)
        return []

    items = data if isinstance(data, list) else data.get("tasks", [])
    seen: set[str] = set()
    all_tasks: list[dict] = []
    for item in items:
        tid = item.get("id", "")
        if tid and tid not in seen:
            seen.add(tid)
            all_tasks.append(item)
    log.info("  fetched %d unique tasks", len(all_tasks))

    COGAT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    COGAT_CACHE_PATH.write_text(json.dumps(all_tasks, indent=2))
    log.info("Cached %d Cognitive Atlas tasks", len(all_tasks))
    return all_tasks


def _best_match(
    our_label: str,
    our_synonyms: list[str],
    cogat_tasks: list[dict],
) -> tuple[str, str, str] | None:
    """Return (cogat_id, cogat_label, match_type) or None."""
    needle_norm = _normalize(our_label)
    needle_syns = {_normalize(s) for s in our_synonyms}
    needle_syns.add(needle_norm)

    # Exact label match
    for task in cogat_tasks:
        cogat_norm = _normalize(task.get("name", ""))
        if cogat_norm in needle_syns:
            return task["id"], task["name"], "exact"

    # Substring containment: our label inside cogat label or vice versa
    for task in cogat_tasks:
        cogat_norm = _normalize(task.get("name", ""))
        if needle_norm in cogat_norm or cogat_norm in needle_norm:
            return task["id"], task["name"], "substring"

    # Synonym match against cogat aliases (when available)
    for task in cogat_tasks:
        aliases = [_normalize(a) for a in task.get("alias", [])]
        if needle_syns & set(aliases):
            return task["id"], task["name"], "alias"

    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--refresh", action="store_true",
                        help="Ignore cache and re-fetch from API")
    args = parser.parse_args(argv)

    if args.refresh and COGAT_CACHE_PATH.exists():
        COGAT_CACHE_PATH.unlink()

    cogat_tasks = _fetch_cogat_tasks()
    log.info("Cognitive Atlas: %d tasks loaded", len(cogat_tasks))

    raw = yaml.safe_load(TASK_ONTOLOGY_PATH.read_text())
    our_tasks: list[dict] = raw.get("tasks", [])

    mappings: list[dict] = []
    matched = 0
    unmatched_ids: list[str] = []

    for task in our_tasks:
        tid = task["id"]
        label = task.get("label", tid)
        synonyms = task.get("synonyms", [])

        result = _best_match(label, synonyms, cogat_tasks)
        if result:
            cogat_id, cogat_label, match_type = result
            mappings.append({
                "our_id": tid,
                "cogat_id": cogat_id,
                "cogat_label": cogat_label,
                "match_type": match_type,
            })
            matched += 1
        else:
            mappings.append({
                "our_id": tid,
                "cogat_id": None,
                "cogat_label": None,
                "match_type": "none",
            })
            unmatched_ids.append(tid)

    log.info("Matched: %d/%d", matched, len(our_tasks))
    if unmatched_ids:
        log.info("Unmatched: %s", ", ".join(unmatched_ids))

    output = {
        "_meta": {
            "source": "https://www.cognitiveatlas.org",
            "cogat_task_count": len(cogat_tasks),
            "our_task_count": len(our_tasks),
            "matched": matched,
        },
        "task_mappings": mappings,
    }

    if not args.dry_run:
        TASK_ATLAS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TASK_ATLAS_PATH.write_text(
            "# Cognitive Atlas bridge for neural-search behavioral task IDs\n"
            "# cogat_id = None means no match found — manual curation needed\n\n"
            + yaml.dump(output, default_flow_style=False, allow_unicode=True, sort_keys=False)
        )
        log.info("Written: %s", TASK_ATLAS_PATH)
    else:
        log.info("[dry-run] Would write %s", TASK_ATLAS_PATH)

    return 0


if __name__ == "__main__":
    sys.exit(main())
