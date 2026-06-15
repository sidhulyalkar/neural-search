#!/usr/bin/env python3
"""Initialize or summarize ablation reports from variant eval reports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ABLATIONS = [
    "no_dense",
    "no_sparse",
    "no_graph",
    "no_provenance",
    "no_affordance",
    "no_field_weighting",
    "no_missingness_penalty",
    "no_usefulness_reranker",
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=Path("reports/eval/ablations"))
    parser.add_argument("--out", type=Path, default=Path("reports/eval/ablation_report.json"))
    args = parser.parse_args(argv)

    rows = []
    for ablation in ABLATIONS:
        path = args.eval_dir / f"{ablation}.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            metrics = data.get("metrics", {})
            status = data.get("status", "computed")
        else:
            metrics = {}
            status = "Pending benchmark artifact"
        rows.append({"ablation": ablation, "status": status, "metrics": metrics})

    report = {"ablations": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ablations": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
