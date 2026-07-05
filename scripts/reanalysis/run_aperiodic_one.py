"""Run aperiodic spectral parameterization for one dataset/signal.

Usage:
    python scripts/reanalysis/run_aperiodic_one.py \
        --dataset-id dataset:dandi:000001 \
        --signal-npy path/to/signal.npy \
        --sample-rate 1000 \
        --backend mock \
        --out artifacts/spectral/dataset_dandi_000001.json

``--signal-npy`` must point to a 1-D numpy array (``.npy``) containing a
single channel's raw or LFP-band time series. This script does not download
or read NWB/BIDS assets itself — pair it with a small project-specific
extraction step that writes out the channel(s) you want analyzed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neural_search.spectral.features import (  # noqa: E402
    build_feature_bundle,
    compute_spectral_estimate,
)
from neural_search.spectral.schemas import (  # noqa: E402
    AperiodicEligibility,
    SpectralRunConfig,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--signal-npy", required=True, type=Path)
    parser.add_argument("--sample-rate", required=True, type=float)
    parser.add_argument("--backend", default="mock", choices=["mock", "specparam", "fooof", "irasa"])
    parser.add_argument("--freq-low", type=float, default=1.0)
    parser.add_argument("--freq-high", type=float, default=40.0)
    parser.add_argument("--aperiodic-mode", default="fixed", choices=["fixed", "knee"])
    parser.add_argument("--channel-id", default=None)
    parser.add_argument("--region-id", default=None)
    parser.add_argument("--task-state-id", default=None)
    parser.add_argument("--out", type=Path, default=None, help="Write the bundle JSON here")
    return parser.parse_args()


def run_one(
    *,
    dataset_id: str,
    signal: np.ndarray,
    sample_rate: float,
    backend: str = "mock",
    freq_range_hz: tuple[float, float] = (1.0, 40.0),
    aperiodic_mode: str = "fixed",
    channel_id: str | None = None,
    region_id: str | None = None,
    task_state_id: str | None = None,
):
    """Fit one signal and return its ``SpectralFeatureBundle``.

    Eligibility is reported as ``"unknown"`` here since this function only
    has the raw signal, not a ``NormalizedDatasetRecord``; pair with
    ``neural_search.spectral.eligibility.detect_aperiodic_eligibility`` (or
    the ``/api/datasets/{id}/aperiodic/eligibility`` endpoint) for a real
    eligibility determination against the dataset's full metadata.
    """

    run_config = SpectralRunConfig(
        run_id=f"run:{dataset_id}:{backend}",
        backend=backend,
        freq_range_hz=freq_range_hz,
        aperiodic_mode=aperiodic_mode,
        sample_rate_hz=sample_rate,
    )

    estimate = compute_spectral_estimate(
        signal,
        sample_rate,
        dataset_id=dataset_id,
        estimate_id=f"estimate:{dataset_id}:{channel_id or 'ch0'}",
        run_config=run_config,
        channel_id=channel_id,
        region_id=region_id,
        task_state_id=task_state_id,
        channel_metadata_present=channel_id is not None,
        region_metadata_present=region_id is not None,
        task_state_present=task_state_id is not None,
    )

    eligibility = AperiodicEligibility(
        dataset_id=dataset_id,
        support_level="unknown",
        confidence=0.5,
        reasons=["Eligibility was not evaluated by this script; only the signal was fit directly."],
    )
    return build_feature_bundle(
        [estimate],
        bundle_id=f"bundle:{dataset_id}:{backend}",
        dataset_id=dataset_id,
        eligibility=eligibility,
        run_config=run_config,
    )


def main() -> None:
    args = _parse_args()
    signal = np.load(args.signal_npy)

    bundle = run_one(
        dataset_id=args.dataset_id,
        signal=signal,
        sample_rate=args.sample_rate,
        backend=args.backend,
        freq_range_hz=(args.freq_low, args.freq_high),
        aperiodic_mode=args.aperiodic_mode,
        channel_id=args.channel_id,
        region_id=args.region_id,
        task_state_id=args.task_state_id,
    )

    payload = bundle.model_dump(mode="json")
    print(json.dumps(payload, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
