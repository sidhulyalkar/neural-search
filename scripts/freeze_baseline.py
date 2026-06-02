#!/usr/bin/env python3
"""Freeze a snapshot of current metrics as the v0.9 comparison baseline.

Usage:
    python scripts/freeze_baseline.py
    python scripts/freeze_baseline.py --output reports/baseline_v09.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/baseline_v09.json")
    args = parser.parse_args(argv)

    # Count corpus records
    corpus_sizes: dict[str, int] = {}
    normalized_dir = Path("data/corpus/normalized")
    for f in sorted(normalized_dir.glob("real_*.jsonl")):
        count = sum(1 for _ in f.open())
        corpus_sizes[f.stem] = count

    # Count seed pairs
    seed_path = Path("data/eval/usefulness_seed_pairs.jsonl")
    n_pairs = sum(1 for _ in seed_path.open()) if seed_path.exists() else 0

    # Count embedding vectors
    embed_path = Path("data/embeddings/real_all.field_embeddings.jsonl")
    n_embeddings = sum(1 for _ in embed_path.open()) if embed_path.exists() else 0

    # Read last correlation report if exists
    corr_path = Path("reports/usefulness_correlation_v09.json")
    spearman_r = None
    if corr_path.exists():
        try:
            data = json.loads(corr_path.read_text())
            spearman_r = data.get("spearman_r")
        except Exception:
            pass

    baseline = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "corpus_record_counts": corpus_sizes,
        "total_corpus_records": sum(corpus_sizes.values()),
        "seed_pairs": n_pairs,
        "field_embedding_vectors": n_embeddings,
        "embedding_provider": "hashing-64",
        "spearman_r": spearman_r,
        "notes": "v0.9 baseline before Track 1 embedding upgrade",
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline, indent=2))
    print(f"Baseline frozen → {out}")
    for k, v in baseline.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
