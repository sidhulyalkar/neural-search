"""Run aperiodic spectral parameterization for a batch of signals.

Reads a JSONL manifest where each line describes one signal to fit:

    {"dataset_id": "dataset:dandi:000001", "signal_npy": "path/to/ch0.npy",
     "sample_rate": 1000.0, "channel_id": "ch0", "region_id": "V1"}

Writes one ``SpectralFeatureBundle`` per line to ``--out`` (JSONL). Failures
on individual lines are logged and skipped rather than aborting the batch.

Usage:
    python scripts/reanalysis/run_aperiodic_batch.py \
        --manifest path/to/manifest.jsonl \
        --out artifacts/spectral/bundles.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.reanalysis.run_aperiodic_one import run_one  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--backend", default="mock", choices=["mock", "specparam", "fooof", "irasa"])
    parser.add_argument("--freq-low", type=float, default=1.0)
    parser.add_argument("--freq-high", type=float, default=40.0)
    parser.add_argument("--aperiodic-mode", default="fixed", choices=["fixed", "knee"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_lines = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_ok, n_failed = 0, 0
    with args.out.open("w", encoding="utf-8") as handle:
        for entry in manifest_lines:
            dataset_id = entry["dataset_id"]
            try:
                signal = np.load(entry["signal_npy"])
                bundle = run_one(
                    dataset_id=dataset_id,
                    signal=signal,
                    sample_rate=float(entry["sample_rate"]),
                    backend=entry.get("backend", args.backend),
                    freq_range_hz=(args.freq_low, args.freq_high),
                    aperiodic_mode=entry.get("aperiodic_mode", args.aperiodic_mode),
                    channel_id=entry.get("channel_id"),
                    region_id=entry.get("region_id"),
                    task_state_id=entry.get("task_state_id"),
                )
            except Exception as exc:  # noqa: BLE001 - batch jobs must not abort on one bad row
                print(f"[skip] {dataset_id}: {exc}", file=sys.stderr)
                n_failed += 1
                continue
            handle.write(json.dumps(bundle.model_dump(mode="json")) + "\n")
            n_ok += 1

    print(f"Wrote {n_ok} bundles to {args.out} ({n_failed} skipped)", file=sys.stderr)


if __name__ == "__main__":
    main()
