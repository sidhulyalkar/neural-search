"""Synthesize one consensus claim per finding cluster via Claude Haiku.

Usage: python scripts/literature/synthesize_claims.py [--config PATH] [--max-clusters N]
Input:  artifacts/claims/finding_clusters.jsonl
Output: artifacts/claims/claims_raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import anthropic
import yaml

from neural_search.literature.claim_synthesizer import synthesize_claim

INPUT_PATH = REPO_ROOT / "artifacts/claims/finding_clusters.jsonl"
OUTPUT_PATH = REPO_ROOT / "artifacts/claims/claims_raw.jsonl"
DEFAULT_CONFIG = REPO_ROOT / "configs/literature/synthesis_v1.yaml"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-clusters", type=int, default=None)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    client = anthropic.Anthropic(api_key=api_key)

    clusters = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                clusters.append(json.loads(line))
    if args.max_clusters:
        clusters = clusters[: args.max_clusters]
    print(f"Synthesizing {len(clusters)} clusters...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    success = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for i, cluster in enumerate(clusters):
            try:
                claim = synthesize_claim(cluster, client, config)
                fout.write(json.dumps(claim, ensure_ascii=False) + "\n")
                success += 1
            except Exception as e:
                print(f"  ERROR cluster {i}: {e}", file=sys.stderr)
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(clusters)} done ({success} succeeded)...")

    print(f"Done. {success}/{len(clusters)} claims written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
