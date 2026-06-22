"""Normalize brain region strings in findings using Allen CCF.

Usage: python scripts/literature/normalize_regions.py
Input:  artifacts/literature/findings_tier1_ollama.jsonl
Output: artifacts/claims/findings_normalized.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.literature.region_normalizer import (
    build_name_index,
    fetch_allen_ccf,
    normalize_finding,
)

INPUT_PATH = REPO_ROOT / "artifacts/literature/findings_tier1_ollama.jsonl"
OUTPUT_PATH = REPO_ROOT / "artifacts/claims/findings_normalized.jsonl"
CACHE_PATH = REPO_ROOT / "data/ontology/allen_ccf.json"


def main() -> None:
    print("Fetching/loading Allen CCF...")
    structures = fetch_allen_ccf(CACHE_PATH)
    name_index = build_name_index(structures)
    print(f"  {len(structures)} structures loaded")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with INPUT_PATH.open(encoding="utf-8") as fin, OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            finding = json.loads(line)
            normalized = normalize_finding(finding, name_index, structures)
            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1
            if count % 10000 == 0:
                print(f"  {count} findings normalized...")

    print(f"Done. {count} findings written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
