"""Expand OpenNeuro corpus with diverse neuroscience datasets."""

from __future__ import annotations

import json
from pathlib import Path

from neural_search.ingestion.openneuro import fetch_openneuro, normalize_openneuro_record

# BIDS modalities available on OpenNeuro
# See: https://bids-specification.readthedocs.io/en/stable/04-modality-specific-files/
OPENNEURO_MODALITIES = [
    "eeg",      # EEG
    "func",     # fMRI functional
    "anat",     # MRI anatomical
    "meg",      # MEG
    "ieeg",     # intracranial EEG
    "dwi",      # diffusion weighted imaging
    "pet",      # PET
    "beh",      # behavioral
    "fmap",     # field maps
    "perf",     # perfusion
    "micr",     # microscopy
    "nirs",     # near-infrared spectroscopy
    None,       # all public datasets (no filter)
]

OUTPUT_PATH = Path("data/corpus/normalized/real_openneuro.jsonl")


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

    for modality in OPENNEURO_MODALITIES:
        label = modality if modality else "all"
        print(f"Fetching modality: {label}...")
        try:
            payload = fetch_openneuro(modality, limit=100)
            edges = payload.get("data", {}).get("datasets", {}).get("edges", [])
            for edge in edges:
                node = edge.get("node", {})
                source_id = node.get("id", "")
                if source_id in seen_ids:
                    continue
                seen_ids.add(source_id)
                try:
                    record = normalize_openneuro_record(node)
                    all_records.append(record.model_dump())
                    title = record.title[:60] if record.title else "No title"
                    print(f"  + {source_id}: {title}...")
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
