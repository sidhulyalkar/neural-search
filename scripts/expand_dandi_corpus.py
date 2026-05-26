"""Expand DANDI corpus with diverse neuroscience datasets."""

from __future__ import annotations

import json
from pathlib import Path

from neural_search.ingestion.dandi import fetch_dandi, normalize_dandiset_record

# Diverse search queries to cover different neuroscience domains
DANDI_QUERIES = [
    "neuropixels",
    "calcium imaging",
    "electrophysiology",
    "hippocampus",
    "visual cortex",
    "prefrontal cortex",
    "decision making",
    "motor cortex",
    "behavior",
    "two-photon",
    "ecog",
    "ieeg",
    "fiber photometry",
    "patch clamp",
    "optogenetics",
    "reward",
    "learning",
    "working memory",
    "attention",
    "sleep",
]

OUTPUT_PATH = Path("data/corpus/normalized/real_dandi.jsonl")


def main() -> None:
    seen_ids: set[str] = set()
    all_records: list[dict] = []

    # Load existing records to avoid duplicates
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                source_id = record.get("source_id", "")
                if source_id:
                    seen_ids.add(source_id)
        print(f"Loaded {len(seen_ids)} existing records")

    for query in DANDI_QUERIES:
        print(f"Fetching: {query}...")
        try:
            payload = fetch_dandi(query, limit=15)
            results = payload.get("results", [])
            for raw in results:
                source_id = str(raw.get("identifier", raw.get("id", "")))
                if source_id in seen_ids:
                    continue
                seen_ids.add(source_id)
                try:
                    record = normalize_dandiset_record(raw)
                    all_records.append(record.model_dump())
                    print(f"  + {source_id}: {record.title[:60]}...")
                except Exception as e:
                    print(f"  ! Failed to normalize {source_id}: {e}")
        except Exception as e:
            print(f"  ! Query failed: {e}")

    print(f"\nTotal new records: {len(all_records)}")

    # Append to file
    if all_records:
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record) + "\n")
        print(f"Appended to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
