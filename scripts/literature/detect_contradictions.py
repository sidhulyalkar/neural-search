"""Detect and mark contradicting claim pairs.

Usage: python scripts/literature/detect_contradictions.py
Input:  artifacts/claims/claims_raw.jsonl
Output: artifacts/claims/claims_validated.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.literature.claim_synthesizer import detect_contradictions

INPUT_PATH = REPO_ROOT / "artifacts/claims/claims_raw.jsonl"
OUTPUT_PATH = REPO_ROOT / "artifacts/claims/claims_validated.jsonl"


def main() -> None:
    claims = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                claims.append(json.loads(line))
    print(f"Loaded {len(claims)} claims")

    validated = detect_contradictions(claims)
    contested = sum(1 for c in validated if c["status"] == "contested")
    print(f"  {contested} contested claims found")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for claim in validated:
            f.write(json.dumps(claim, ensure_ascii=False) + "\n")
    print(f"Done. {len(validated)} claims written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
