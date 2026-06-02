#!/usr/bin/env python3
"""Compare v0.9 baseline with v2.0 results side-by-side.

Usage:
    python scripts/compare_versions.py
"""
from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    baseline_path = Path("reports/baseline_v09.json")
    eval_path = Path("reports/evaluation_suite_v2.json")
    corr_path = Path("reports/usefulness_correlation_v09.json")
    recall_path = Path("reports/turbovec_recall.json")

    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    eval_v2 = json.loads(eval_path.read_text()) if eval_path.exists() else {}
    corr = json.loads(corr_path.read_text()) if corr_path.exists() else {}
    recall = json.loads(recall_path.read_text()) if recall_path.exists() else {}

    layers = eval_v2.get("layers", {})
    spearman_v2 = layers.get("2_usefulness", {}).get("spearman_r")
    corpus_v2_ok = layers.get("3_corpus", {}).get("returncode") == 0
    recall_v2 = layers.get("4_index", {}).get("recall")
    graph_v2_pct = layers.get("5_graph", {}).get("pct_pairs_changed")

    # Fall back to standalone report files when eval suite hasn't been run
    if spearman_v2 is None:
        spearman_v2 = corr.get("spearman_r")
    if recall_v2 is None:
        recall_v2 = recall.get("mean_recall")

    print("=" * 60)
    print("Neural Search: v0.9 Baseline vs v2.0")
    print("=" * 60)
    print(f"{'Metric':<35} {'v0.9':>10} {'v2.0':>10}")
    print("-" * 60)
    print(f"{'Corpus size':<35} {baseline.get('total_corpus_records', '738'):>10} {'≥4000' if corpus_v2_ok else '?':>10}")
    print(f"{'Embedding model':<35} {'64d hashing':>10} {'BGE-1024d':>10}")
    print(f"{'Spearman r (usefulness corr.)':<35} {str(baseline.get('spearman_r', '0.5044')):>10} {str(spearman_v2 or '?'):>10}")
    print(f"{'TurboVec recall@50':<35} {'N/A':>10} {str(recall_v2 or '?'):>10}")
    print(f"{'Graph s9 pairs changed (%)':<35} {'0%':>10} {f'{graph_v2_pct}%' if graph_v2_pct else '?':>10}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
