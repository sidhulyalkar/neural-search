"""CLI: python -m neural_search.decoder_generalization --source synthetic."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from neural_search.decoder_generalization.sources import (
    load_bnci2014_001,
    load_npz,
    repair_known_synthetic_channel_map,
    synthetic_dataset,
)
from neural_search.decoder_generalization.study import PROTOCOLS, run_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Leakage-aware neural decoder transfer pilot; local only by default."
    )
    parser.add_argument("--source", choices=("synthetic", "npz", "moabb"), default="synthetic")
    parser.add_argument("--npz", type=Path, help="NPZ with X, y, subject, session")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--protocol", choices=(*PROTOCOLS, "all"), default="all")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--synthetic-shift", choices=("gain", "channel_swap"), default="gain"
    )
    parser.add_argument(
        "--synthetic-oracle-repair", action="store_true",
        help="Oracle-known channel-map repair on synthetic channel_swap data only",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/decoder_generalization/pilot.json"),
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.synthetic_oracle_repair and not (
        args.source == "synthetic" and args.synthetic_shift == "channel_swap"
    ):
        parser.error("Oracle repair requires synthetic channel_swap data")
    if args.source == "npz":
        if args.npz is None:
            parser.error("--source npz requires --npz PATH")
        dataset = load_npz(args.npz)
    elif args.source == "moabb":
        dataset = load_bnci2014_001(args.subjects)
    else:
        dataset = synthetic_dataset(seed=args.seed, shift=args.synthetic_shift)
        if args.synthetic_oracle_repair:
            dataset = repair_known_synthetic_channel_map(dataset)
    source_loading_seconds = time.perf_counter() - started
    protocols = PROTOCOLS if args.protocol == "all" else (args.protocol,)
    report = run_study(dataset, protocols=protocols, seed=args.seed)
    report["execution"] = {
        "source_loading_seconds": round(source_loading_seconds, 3),
        "evaluation_seconds": report["elapsed_seconds"],
        "total_before_artifact_write_seconds": round(
            time.perf_counter() - started, 3
        ),
        "paid_model_calls": 0,
        "paid_gpu_jobs": 0,
        "billing_cost_usd": None,
        "billing_note": (
            "Measured wall time is not a monetary cost estimate. "
            "Local and GitHub Actions compute/network charges are not metered."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace prevents a killed run leaving a partial JSON result.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent,
        prefix=".decoder-", suffix=".json", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, args.output)
    print(json.dumps({
        "output": str(args.output),
        "source": report["dataset"]["source"],
        "summary": report["summary"],
        "elapsed_seconds": report["elapsed_seconds"],
        "execution": report["execution"],
        "evidence_status": report["evidence_status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
