#!/usr/bin/env python3
"""Compare silver qrels to human/adjudicated gold qrels once gold labels exist.

If gold qrels are absent, writes a placeholder explaining how to run after annotation.

Usage:
    python scripts/eval/calibrate_silver_qrels.py \\
        --silver artifacts/qrels_silver.jsonl \\
        --gold artifacts/qrels.jsonl \\
        --out reports/eval/silver_gold_calibration.md
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.eval.silver_qrels_schema import SILVER_EVAL_WATERMARK, SilverQrelsEntry

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_silver(path: Path) -> dict[tuple[str, str], SilverQrelsEntry]:
    entries: dict[tuple[str, str], SilverQrelsEntry] = {}
    if not path.exists():
        return entries
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = SilverQrelsEntry.model_validate(json.loads(line))
            entries[(e.query_id, e.dataset_id)] = e
    return entries


def _load_gold(path: Path) -> dict[tuple[str, str], int]:
    gold: dict[tuple[str, str], int] = {}
    if not path.exists():
        return gold
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            # Handle both QrelsEntryV1 (relevance) and annotate_candidates output (label)
            rel = row.get("relevance") if row.get("relevance") is not None else row.get("label")
            if rel is None:
                continue
            qid = str(row.get("query_id", ""))
            did = str(row.get("dataset_id", "") or row.get("record_id", ""))
            if qid and did:
                gold[(qid, did)] = int(rel)
    return gold


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------


def _confusion_matrix(
    silver: dict[tuple[str, str], SilverQrelsEntry],
    gold: dict[tuple[str, str], int],
) -> dict[str, Any]:
    """Build confusion matrix for overlapping (query, dataset) pairs."""
    pairs = set(silver.keys()) & set(gold.keys())
    if not pairs:
        return {"overlap": 0}

    cm: dict[tuple[int, int], int] = defaultdict(int)
    for key in pairs:
        s_rel = silver[key].relevance
        g_rel = gold[key]
        cm[(g_rel, s_rel)] += 1

    # Accuracy
    correct = sum(v for (g, s), v in cm.items() if g == s)
    total = sum(cm.values())
    accuracy = correct / total if total else 0.0

    # Weighted kappa
    labels = list(range(4))
    n = total
    po = accuracy  # observed agreement
    # Expected agreement by chance
    gold_counts: dict[int, int] = defaultdict(int)
    silver_counts: dict[int, int] = defaultdict(int)
    for (g, s), v in cm.items():
        gold_counts[g] += v
        silver_counts[s] += v
    pe = sum(
        (gold_counts[lbl] / n) * (silver_counts[lbl] / n)
        for lbl in labels
        if n > 0
    )
    kappa = (po - pe) / (1.0 - pe) if (1.0 - pe) != 0 else 0.0

    return {
        "overlap": len(pairs),
        "accuracy": round(accuracy, 3),
        "weighted_kappa": round(kappa, 3),
        "confusion_matrix": {f"{g}_gold_{s}_silver": v for (g, s), v in cm.items()},
    }


def _high_conf_accuracy(
    silver: dict[tuple[str, str], SilverQrelsEntry],
    gold: dict[tuple[str, str], int],
    threshold: float = 0.70,
) -> dict[str, Any]:
    hc = {k: v for k, v in silver.items() if v.confidence >= threshold}
    overlap = set(hc.keys()) & set(gold.keys())
    if not overlap:
        return {"count": 0}
    correct = sum(1 for k in overlap if hc[k].relevance == gold[k])
    return {"count": len(overlap), "accuracy": round(correct / len(overlap), 3)}


def _hn_precision(
    silver: dict[tuple[str, str], SilverQrelsEntry],
    gold: dict[tuple[str, str], int],
) -> dict[str, Any]:
    hn_silver = {k: v for k, v in silver.items() if v.hard_negative_violation}
    overlap = set(hn_silver.keys()) & set(gold.keys())
    if not overlap:
        return {"count": 0}
    correct_zero = sum(1 for k in overlap if gold[k] == 0)
    return {"count": len(overlap), "precision": round(correct_zero / len(overlap), 3)}


def _calibration_by_bin(
    silver: dict[tuple[str, str], SilverQrelsEntry],
    gold: dict[tuple[str, str], int],
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    pairs = list(set(silver.keys()) & set(gold.keys()))
    if not pairs:
        return []

    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for key in pairs:
        conf = silver[key].confidence
        correct = silver[key].relevance == gold[key]
        bin_idx = min(int(conf * n_bins), n_bins - 1)
        bins[bin_idx].append((conf, correct))

    result = []
    for i, b in enumerate(bins):
        if not b:
            continue
        lo = i / n_bins
        hi = (i + 1) / n_bins
        avg_conf = mean(c for c, _ in b)
        acc = mean(int(ok) for _, ok in b)
        result.append({
            "bin": f"{lo:.1f}–{hi:.1f}",
            "count": len(b),
            "mean_confidence": round(avg_conf, 3),
            "accuracy": round(acc, 3),
        })
    return result


def _per_intent_agreement(
    silver: dict[tuple[str, str], SilverQrelsEntry],
    gold: dict[tuple[str, str], int],
    queries: dict[str, Any],
) -> dict[str, float]:
    by_intent: dict[str, list[bool]] = defaultdict(list)
    for key in set(silver.keys()) & set(gold.keys()):
        qid = key[0]
        q = queries.get(qid)
        intent = q.canonical_intent() if q else "unknown"
        by_intent[intent].append(silver[key].relevance == gold[key])
    return {intent: round(mean(v), 3) for intent, v in by_intent.items() if v}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _write_placeholder(out: Path, silver_path: Path, gold_path: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# Silver–Gold Calibration\n\n"
        f"> **{SILVER_EVAL_WATERMARK}**\n\n"
        f"Gold qrels are not yet available at `{gold_path}`.\n\n"
        f"Run this command after completing human annotation:\n\n"
        f"```bash\n"
        f"python scripts/eval/calibrate_silver_qrels.py \\\\\n"
        f"    --silver {silver_path} \\\\\n"
        f"    --gold {gold_path} \\\\\n"
        f"    --out {out}\n"
        f"```\n\n"
        f"## What will be computed\n\n"
        f"- Agreement accuracy (silver vs gold)\n"
        f"- Weighted kappa\n"
        f"- Confusion matrix\n"
        f"- Per-intent agreement\n"
        f"- Per-source agreement\n"
        f"- High-confidence silver accuracy\n"
        f"- Hard-negative violation precision\n"
        f"- Calibration by confidence bin\n",
        encoding="utf-8",
    )


def _write_report(
    out: Path,
    cm: dict[str, Any],
    hc: dict[str, Any],
    hn: dict[str, Any],
    bins: list[dict[str, Any]],
    intent_agr: dict[str, float],
    silver_count: int,
    gold_count: int,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Silver–Gold Calibration Report\n",
        f"> **{SILVER_EVAL_WATERMARK}**\n",
        f"Silver labels: {silver_count} | Gold labels: {gold_count} "
        f"| Overlap: {cm.get('overlap', 0)}\n",
        "\n## Overall agreement\n",
        f"- Accuracy: {cm.get('accuracy', 'N/A')}",
        f"- Weighted kappa: {cm.get('weighted_kappa', 'N/A')}",
        "\n## High-confidence silver (≥0.70) accuracy\n",
        f"- Count: {hc.get('count', 0)} | Accuracy: {hc.get('accuracy', 'N/A')}",
        "\n## Hard-negative violation precision\n",
        f"- Count: {hn.get('count', 0)} | Precision (gold=0): {hn.get('precision', 'N/A')}",
        "\n## Calibration by confidence bin\n",
    ]
    if bins:
        lines.append("| Bin | Count | Mean Conf | Accuracy |")
        lines.append("|-----|-------|-----------|----------|")
        for b in bins:
            lines.append(
                f"| {b['bin']} | {b['count']} | {b['mean_confidence']} | {b['accuracy']} |"
            )
    else:
        lines.append("No overlapping pairs available.")

    if intent_agr:
        lines.append("\n## Per-intent agreement\n")
        for intent, agr in sorted(intent_agr.items()):
            lines.append(f"- {intent}: {agr:.3f}")

    if cm.get("confusion_matrix"):
        lines.append("\n## Confusion matrix (gold_row × silver_col)\n")
        for k, v in sorted(cm["confusion_matrix"].items()):
            lines.append(f"- {k}: {v}")

    out.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate silver qrels against gold qrels.")
    p.add_argument("--silver", required=True, type=Path)
    p.add_argument("--gold", required=True, type=Path)
    p.add_argument("--out", default=Path("reports/eval/silver_gold_calibration.md"), type=Path)
    p.add_argument("--queries", default=Path("artifacts/benchmark_queries.jsonl"), type=Path)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    silver = _load_silver(args.silver)
    gold = _load_gold(args.gold)

    if not gold:
        print(f"No gold qrels found at {args.gold}. Writing placeholder.")
        _write_placeholder(args.out, args.silver, args.gold)
        print(f"Placeholder written to {args.out}")
        return

    # Load queries for intent breakdown
    queries: dict[str, Any] = {}
    if args.queries.exists():
        from scripts.eval.build_silver_qrels import _load_queries
        queries = _load_queries(args.queries)

    cm = _confusion_matrix(silver, gold)
    hc = _high_conf_accuracy(silver, gold)
    hn = _hn_precision(silver, gold)
    bins = _calibration_by_bin(silver, gold)
    intent_agr = _per_intent_agreement(silver, gold, queries)

    _write_report(args.out, cm, hc, hn, bins, intent_agr, len(silver), len(gold))
    print(f"Calibration report written to {args.out}")
    print(f"  Overlap: {cm.get('overlap', 0)} pairs")
    print(f"  Accuracy: {cm.get('accuracy', 'N/A')}")
    print(f"  Weighted kappa: {cm.get('weighted_kappa', 'N/A')}")


if __name__ == "__main__":
    main()
