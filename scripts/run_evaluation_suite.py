#!/usr/bin/env python3
"""Run the complete 5-layer evaluation suite and produce a summary report.

Layer 1: Retrieval quality (NDCG@10, MRR, P@5, Recall@10) via existing benchmark
Layer 2: Latent usefulness quality (Spearman r, pairwise accuracy)
Layer 3: Corpus quality dashboard
Layer 4: Index quality (turbovec recall, latency)
Layer 5: Graph contribution (ablation)

Usage:
    python scripts/run_evaluation_suite.py
    python scripts/run_evaluation_suite.py --layers 1,2 --n-queries 30
    python scripts/run_evaluation_suite.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def _run_and_capture(cmd: list[str]) -> tuple[int, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode, result.stdout + result.stderr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", default="1,2,3,4,5", help="Comma-separated layer numbers")
    parser.add_argument("--n-queries", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    layers = [int(x) for x in args.layers.split(",")]

    if args.dry_run:
        print(f"DRY RUN — would run layers {layers} with {args.n_queries} queries")
        return 0

    results: dict = {
        "run_at": datetime.now(UTC).isoformat(),
        "n_queries": args.n_queries,
        "layers": {},
    }

    if 1 in layers:
        print("\n[Layer 1] Retrieval quality benchmark...")
        rc, out = _run_and_capture([
            sys.executable, "-m", "neural_search.evaluation",
            "--suite", "real_corpus", "--k", "10",
        ])
        results["layers"]["1_retrieval"] = {"returncode": rc, "summary": out[-500:]}
        print(f"  Layer 1: {'OK' if rc == 0 else 'FAILED'}")

    if 2 in layers:
        print("\n[Layer 2] Latent usefulness correlation...")
        rc, out = _run_and_capture([
            sys.executable, "scripts/evaluate_usefulness_correlation.py",
            "--n-queries", str(args.n_queries),
        ])
        spearman_r = None
        for line in out.split("\n"):
            if "spearman" in line.lower() and "=" in line:
                try:
                    spearman_r = float(line.split("=")[1].split()[0])
                except Exception:
                    pass
        results["layers"]["2_usefulness"] = {
            "returncode": rc,
            "spearman_r": spearman_r,
        }
        print(f"  Layer 2: Spearman r = {spearman_r}")

    if 3 in layers:
        print("\n[Layer 3] Corpus quality...")
        rc, out = _run_and_capture([sys.executable, "scripts/validate_corpus.py"])
        results["layers"]["3_corpus"] = {"returncode": rc, "summary": out[-500:]}
        print(f"  Layer 3: {'PASS' if rc == 0 else 'FAIL'}")

    if 4 in layers:
        print("\n[Layer 4] Index quality (turbovec recall)...")
        rc, out = _run_and_capture([
            sys.executable, "scripts/validate_turbovec_recall.py", "--k", "50",
        ])
        recall = None
        try:
            for line in out.split("\n"):
                if "mean_recall" in line:
                    recall = float(line.split(":")[1].strip().rstrip(","))
                    break
        except Exception:
            pass
        results["layers"]["4_index"] = {"returncode": rc, "recall": recall}
        print(f"  Layer 4: recall@50 = {recall}")

    if 5 in layers:
        print("\n[Layer 5] Graph contribution (ablation)...")
        rc, out = _run_and_capture([
            sys.executable, "scripts/ablate_graph_proximity.py",
            "--n-queries", str(args.n_queries),
        ])
        pct_changed = None
        try:
            report = json.loads(Path("reports/graph_ablation.json").read_text())
            pct_changed = report.get("pct_pairs_changed")
        except Exception:
            pass
        results["layers"]["5_graph"] = {"returncode": rc, "pct_pairs_changed": pct_changed}
        print(f"  Layer 5: {pct_changed}% pairs changed with real graph")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/evaluation_suite_v2.json").write_text(json.dumps(results, indent=2))
    print("\nReport → reports/evaluation_suite_v2.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
