"""Expand OpenNeuro corpus: harvest all 1,754+ datasets via cursor pagination."""
from __future__ import annotations

import json
from pathlib import Path

from neural_search.ingestion.openneuro import fetch_all_openneuro

OUTPUT_PATH = Path("data/corpus/normalized/real_openneuro.jsonl")


def main() -> None:
    seen_ids: set[str] = set()

    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if sid := rec.get("source_id"):
                        seen_ids.add(sid)
                except json.JSONDecodeError:
                    pass
        print(f"Loaded {len(seen_ids)} existing OpenNeuro source IDs")

    print("Fetching all OpenNeuro datasets via cursor pagination…")
    all_records = fetch_all_openneuro()
    print(f"Fetched {len(all_records)} total records from OpenNeuro")

    new_records = [r for r in all_records if r.get("source_id") not in seen_ids]
    print(f"New records (not yet in corpus): {len(new_records)}")

    if new_records:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for rec in new_records:
                f.write(json.dumps(rec) + "\n")
        print(f"Appended {len(new_records)} records to {OUTPUT_PATH}")
    else:
        print("No new records to add.")


if __name__ == "__main__":
    main()
