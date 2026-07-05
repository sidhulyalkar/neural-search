"""Cluster normalized findings by (region, direction, species).

Usage: python scripts/literature/cluster_findings.py [--min-size N]
Input:  artifacts/claims/findings_normalized.jsonl
Output: artifacts/claims/finding_clusters.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from neural_search.literature.claim_synthesizer import cluster_findings

INPUT_PATH = REPO_ROOT / "artifacts/claims/findings_normalized.jsonl"
OUTPUT_PATH = REPO_ROOT / "artifacts/claims/finding_clusters.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-size", type=int, default=3)
    args = parser.parse_args()

    print(f"Loading findings from {INPUT_PATH}...")
    findings = []
    with INPUT_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                findings.append(json.loads(line))
    print(f"  {len(findings)} findings loaded")

    print(f"Clustering (min_size={args.min_size})...")
    clusters = cluster_findings(findings, min_size=args.min_size)
    print(f"  {len(clusters)} clusters formed")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for cluster in clusters:
            # Strip individual findings to reduce file size (keep IDs only)
            slim = {k: v for k, v in cluster.items() if k != "findings"}
            slim["finding_ids"] = [fi.get("finding_id") for fi in cluster["findings"]]
            slim["findings"] = cluster["findings"]  # keep full for synthesis
            f.write(json.dumps(slim, ensure_ascii=False) + "\n")

    print(f"Done. Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
