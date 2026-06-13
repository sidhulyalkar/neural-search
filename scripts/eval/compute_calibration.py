#!/usr/bin/env python3
"""Compute Expected Calibration Error (ECE) for usefulness score buckets.

ECE measures whether predicted usefulness scores match empirical relevance
rates. A score of 0.8 is well-calibrated if ~80% of records at that score
bin are truly useful (label >= 2) in the adjudicated qrels.

Usage (once qrels are available):
    python scripts/eval/compute_calibration.py \
        --qrels artifacts/qrels.jsonl \
        --run artifacts/runs/usefulness.jsonl \
        --out reports/eval/calibration_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    if not path.exists():
        return qrels
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("record_id") or row.get("dataset_id")
            label = row.get("label")
            if label is None or isinstance(label, str):
                label = row.get("relevance")
            if record_id is None or label is None:
                continue
            qrels[str(row["query_id"])][str(record_id)] = int(label)
    return qrels


def load_scored_run(path: Path) -> list[dict]:
    """Load run file rows that include a 'score' field."""
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def compute_ece(
    scores: list[float],
    labels: list[int],
    n_bins: int = 10,
    relevant_threshold: int = 2,
) -> dict:
    """Compute ECE and per-bin calibration statistics.

    Args:
        scores: predicted usefulness scores in [0, 1]
        labels: corresponding graded relevance labels (0-3)
        n_bins: number of equal-width bins
        relevant_threshold: minimum label to count as relevant (default 2)

    Returns:
        dict with ece, mean_confidence, mean_accuracy, and per_bin breakdown
    """
    if not scores:
        return {"status": "no_data", "ece": None, "per_bin": []}

    bin_width = 1.0 / n_bins
    bins: dict[int, list[tuple[float, int]]] = defaultdict(list)

    for score, label in zip(scores, labels, strict=False):
        bin_idx = min(int(score / bin_width), n_bins - 1)
        bins[bin_idx].append((score, label))

    per_bin = []
    total = len(scores)
    ece = 0.0

    for bin_idx in range(n_bins):
        bin_items = bins[bin_idx]
        if not bin_items:
            per_bin.append(
                {
                    "bin": bin_idx,
                    "lower": round(bin_idx * bin_width, 3),
                    "upper": round((bin_idx + 1) * bin_width, 3),
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                }
            )
            continue

        bin_scores = [s for s, _ in bin_items]
        bin_labels = [label for _, label in bin_items]
        mean_conf = sum(bin_scores) / len(bin_scores)
        accuracy = sum(1 for label in bin_labels if label >= relevant_threshold) / len(bin_labels)
        weight = len(bin_items) / total
        ece += weight * abs(mean_conf - accuracy)

        per_bin.append(
            {
                "bin": bin_idx,
                "lower": round(bin_idx * bin_width, 3),
                "upper": round((bin_idx + 1) * bin_width, 3),
                "count": len(bin_items),
                "mean_confidence": round(mean_conf, 4),
                "accuracy": round(accuracy, 4),
                "calibration_gap": round(abs(mean_conf - accuracy), 4),
            }
        )

    return {
        "ece": round(ece, 4),
        "n_bins": n_bins,
        "total_pairs": total,
        "relevant_threshold": relevant_threshold,
        "per_bin": per_bin,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute calibration report for usefulness scorer.")
    parser.add_argument("--qrels", type=Path, default=Path("artifacts/qrels.jsonl"))
    parser.add_argument("--run", type=Path, default=Path("reports/eval/runs/usefulness.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("reports/eval/calibration_report.json"))
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--relevant-threshold", type=int, default=2)
    args = parser.parse_args(argv)

    qrels = load_qrels(args.qrels)
    run_rows = load_scored_run(args.run)

    if not qrels or not run_rows:
        report = {
            "status": "Pending benchmark artifact",
            "note": (
                "Calibration requires adjudicated qrels and scored run files. "
                "Run retrieval baselines and annotate candidates first."
            ),
            "qrels_path": str(args.qrels),
            "run_path": str(args.run),
            "qrels_found": bool(qrels),
            "run_found": bool(run_rows),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"]}, indent=2))
        return 0

    scores: list[float] = []
    labels: list[int] = []

    for row in run_rows:
        qid = str(row.get("query_id", ""))
        rid = str(row.get("record_id", ""))
        score = float(row.get("score", 0.0))
        label = qrels.get(qid, {}).get(rid)
        if label is not None:
            scores.append(min(1.0, max(0.0, score)))
            labels.append(label)

    if not scores:
        report = {
            "status": "no_overlap",
            "note": "No overlap between run records and qrels. Verify query_id and record_id alignment.",
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"status": report["status"]}, indent=2))
        return 0

    calibration = compute_ece(scores, labels, n_bins=args.n_bins, relevant_threshold=args.relevant_threshold)
    report = {
        "status": "computed",
        "qrels_path": str(args.qrels),
        "run_path": str(args.run),
        **calibration,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ece": calibration.get("ece"), "total_pairs": calibration.get("total_pairs")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
