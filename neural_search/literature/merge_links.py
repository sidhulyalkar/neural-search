"""Merge per-source literature-linking output files into one combined file.

Each source (`neural_search.literature.linking` for OpenAlex, `datacite.py`,
`crossref.py`, `semantic_scholar.py`, `pubmed.py`) writes its own raw JSONL
for provenance/debuggability. This module concatenates them, keeping **one
row per `(dataset_record_id, paper_source)` pair** -- a dataset can
legitimately have real matches from multiple sources at once (e.g. both an
OpenAlex row and a DataCite row), so this is a union across sources, not a
dedup down to a single winner per dataset. Within the same source, if a
source file happens to contain more than one row for the same dataset
(shouldn't normally happen -- each `link_corpus_to_*` orchestrator emits
exactly one row per corpus record), the highest-confidence row wins.

Deliberately does NOT overwrite the legacy `paper_dataset_links.jsonl`
(OpenAlex master) -- several existing consumers
(`reanalysis_bridge_builder.py`, `reinterpretation_candidate_builder.py`,
`scripts/build_artifact_manifest.py`'s `PAPER_LINKS_PATH`) assume that file
stays OpenAlex-shaped for backward compatibility. This writes a new,
separate combined file instead.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
LITERATURE_DIR = PROJECT_ROOT / "artifacts" / "literature"
MERGED_LINKS_PATH = LITERATURE_DIR / "paper_dataset_links.merged.jsonl"

DEFAULT_SOURCE_PATHS: dict[str, Path] = {
    "openalex": LITERATURE_DIR / "paper_dataset_links.jsonl",
    "datacite": LITERATURE_DIR / "paper_dataset_links.datacite.jsonl",
    "crossref": LITERATURE_DIR / "paper_dataset_links.crossref.jsonl",
    "semantic_scholar": LITERATURE_DIR / "paper_dataset_links.semantic_scholar.jsonl",
    "pubmed": LITERATURE_DIR / "paper_dataset_links.pubmed.jsonl",
}


def merge_link_sources(
    source_paths: dict[str, Path] = DEFAULT_SOURCE_PATHS,
    out_path: Path = MERGED_LINKS_PATH,
) -> dict[str, int]:
    """Merge source files into `out_path`. Returns {source_name: row_count}
    for whichever sources were actually found (missing files are skipped,
    not fatal -- matches the house style of the orphaned_layers builders)."""

    merged: dict[tuple[str, str], dict] = {}
    counts: dict[str, int] = {}

    for source_name, path in source_paths.items():
        if not path.exists():
            continue
        count = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                count += 1
                key = (row["dataset_record_id"], row.get("paper_source", source_name))
                existing = merged.get(key)
                if existing is None or row.get("confidence", 0.0) > existing.get("confidence", 0.0):
                    merged[key] = row
        counts[source_name] = count

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out_fh:
        for row in merged.values():
            out_fh.write(json.dumps(row) + "\n")

    return counts
