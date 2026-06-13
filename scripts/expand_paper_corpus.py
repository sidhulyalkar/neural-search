#!/usr/bin/env python3
"""Expand paper corpus by fetching OpenAlex papers related to datasets.

This script:
1. Loads normalized dataset records
2. Searches OpenAlex for related papers based on dataset metadata
3. Normalizes and saves paper records
4. Reports on linking potential
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from neural_search.ingestion.openalex import fetch_openalex, normalize_work_record

CORPUS_DIR = project_root / "data" / "corpus" / "normalized"
OUTPUT_PATH = CORPUS_DIR / "real_papers.jsonl"

# Search queries to find neuroscience papers
PAPER_QUERIES = [
    # Dataset-specific searches
    "Neuropixels visual cortex mouse",
    "calcium imaging prefrontal cortex",
    "fMRI resting state connectivity",
    "ECoG motor cortex decoding",
    "hippocampus place cells recording",
    # Task-specific searches
    "decision making neural activity",
    "reversal learning dopamine",
    "motor imagery BCI",
    "working memory prefrontal cortex",
    "visual discrimination task",
    # Modality searches
    "two-photon imaging neural",
    "patch clamp recording",
    "optogenetics behavior",
    "fiber photometry dopamine",
    "MEG decoding",
    # Analysis-focused searches
    "neural decoding population",
    "latent state dynamics neural",
    "encoding model visual cortex",
    "event-related neural activity",
    "spike sorting analysis",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand paper corpus from OpenAlex")
    parser.add_argument("--limit-per-query", type=int, default=10)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--dry-run", action="store_true", help="Don't save records")
    args = parser.parse_args()

    # Load existing papers to avoid duplicates
    seen_ids: set[str] = set()
    output_path = Path(args.output)
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                source_id = record.get("source_id", "")
                if source_id:
                    seen_ids.add(source_id)
        print(f"Loaded {len(seen_ids)} existing paper records")

    all_records: list[dict] = []

    for query in PAPER_QUERIES:
        print(f"Searching: {query}...")
        try:
            payload = fetch_openalex(query, args.limit_per_query)
            results = payload.get("results", [])
            for work in results:
                openalex_id = work.get("id", "")
                source_id = str(openalex_id).replace("https://openalex.org/", "")
                if source_id in seen_ids:
                    continue
                seen_ids.add(source_id)

                try:
                    record = normalize_work_record(work)
                    all_records.append(record.model_dump())
                    title = record.title[:50] if record.title else "No title"
                    print(f"  + {source_id}: {title}...")
                except Exception as e:
                    print(f"  ! Failed to normalize {source_id}: {e}")

        except Exception as e:
            print(f"  ! Query failed: {e}")

    print(f"\nTotal new papers: {len(all_records)}")

    # Save records
    if all_records and not args.dry_run:
        with open(output_path, "a", encoding="utf-8") as f:
            for record in all_records:
                f.write(json.dumps(record) + "\n")
        print(f"Appended to {output_path}")

    # Report statistics
    print("\n--- Paper Corpus Statistics ---")
    total_papers = len(seen_ids)
    print(f"Total papers: {total_papers}")

    # Analyze label distribution
    label_counts: dict[str, int] = {}
    for record in all_records:
        for label in record.get("extracted_labels", []):
            label_type = label.get("label_type", "unknown")
            label_counts[label_type] = label_counts.get(label_type, 0) + 1

    if label_counts:
        print("Label distribution:")
        for label_type, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {label_type}: {count}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
