#!/usr/bin/env python3
"""Compute inter-judge reliability for duplicated neuro-judge labels."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

DEFAULT_JUDGMENTS = Path("data/qrels/llm_judgments.jsonl")
DEFAULT_OUT = Path("reports/eval/dual_judge_reliability.json")
DEFAULT_MD = Path("reports/eval/dual_judge_reliability.md")


def _qwk(labels_a: list[int], labels_b: list[int], num_classes: int = 4) -> float:
    n = len(labels_a)
    if n == 0:
        return 0.0
    conf = [[0] * num_classes for _ in range(num_classes)]
    for a, b in zip(labels_a, labels_b, strict=False):
        conf[a][b] += 1
    weights = [
        [(i - j) ** 2 / (num_classes - 1) ** 2 for j in range(num_classes)]
        for i in range(num_classes)
    ]
    hist_a = [sum(row) for row in conf]
    hist_b = [sum(conf[i][j] for i in range(num_classes)) for j in range(num_classes)]
    expected = [
        [hist_a[i] * hist_b[j] / n for j in range(num_classes)]
        for i in range(num_classes)
    ]
    observed = sum(weights[i][j] * conf[i][j] for i in range(num_classes) for j in range(num_classes))
    expected_weighted = sum(
        weights[i][j] * expected[i][j]
        for i in range(num_classes)
        for j in range(num_classes)
    )
    return 1.0 if expected_weighted == 0.0 else 1.0 - observed / expected_weighted


def _confusion(labels_a: list[int], labels_b: list[int], num_classes: int = 4) -> list[list[int]]:
    conf = [[0] * num_classes for _ in range(num_classes)]
    for a, b in zip(labels_a, labels_b, strict=False):
        conf[a][b] += 1
    return conf


def _is_usable(rec: dict[str, Any]) -> bool:
    if "judge_error" in str(rec.get("rationale_short", "")):
        return False
    try:
        label = int(rec.get("label", -1))
    except (TypeError, ValueError):
        return False
    return 0 <= label <= 3


def load_judgments(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_reliability(records: list[dict[str, Any]]) -> dict[str, Any]:
    total_records = len(records)
    model_counts = Counter(str(r.get("judge_model") or "unknown") for r in records)
    usable = [r for r in records if _is_usable(r)]
    grouped: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    for rec in usable:
        key = (str(rec.get("query_id", "")), str(rec.get("dataset_id", "")))
        model = str(rec.get("judge_model") or "unknown")
        grouped[key].setdefault(model, int(rec["label"]))

    pair_labels: dict[tuple[str, str], tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for by_model in grouped.values():
        if len(by_model) < 2:
            continue
        for model_a, model_b in itertools.combinations(sorted(by_model), 2):
            a_labels, b_labels = pair_labels[(model_a, model_b)]
            a_labels.append(by_model[model_a])
            b_labels.append(by_model[model_b])

    pairwise: dict[str, dict[str, Any]] = {}
    for (model_a, model_b), (labels_a, labels_b) in sorted(pair_labels.items()):
        n = len(labels_a)
        pairwise[f"{model_a} :: {model_b}"] = {
            "model_a": model_a,
            "model_b": model_b,
            "n_overlap": n,
            "exact_agreement": sum(a == b for a, b in zip(labels_a, labels_b, strict=False)) / n,
            "agreement_within_1": sum(abs(a - b) <= 1 for a, b in zip(labels_a, labels_b, strict=False)) / n,
            "quadratic_weighted_kappa": _qwk(labels_a, labels_b),
            "confusion_matrix": _confusion(labels_a, labels_b),
        }

    overlap_pairs = sum(1 for by_model in grouped.values() if len(by_model) >= 2)
    return {
        "total_records": total_records,
        "usable_records": len(usable),
        "unique_labeled_pairs": len(grouped),
        "pairs_with_two_or_more_judges": overlap_pairs,
        "model_counts": dict(model_counts),
        "pairwise": pairwise,
        "estimable": bool(pairwise),
    }


def build_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Dual-Judge Reliability",
        "",
        f"- Total judgment rows: {report['total_records']}",
        f"- Usable non-error rows: {report['usable_records']}",
        f"- Unique labeled pairs: {report['unique_labeled_pairs']}",
        f"- Pairs with >=2 judges: {report['pairs_with_two_or_more_judges']}",
        "",
        "## Judge Rows",
        "",
    ]
    for model, count in sorted(report["model_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- `{model}`: {count}")
    lines.append("")
    lines.append("## Pairwise Agreement")
    lines.append("")
    if not report["pairwise"]:
        lines.append("QWK is not estimable from this artifact because no non-error `(query_id, dataset_id)` pair has labels from two judges.")
        lines.append("")
        return "\n".join(lines)
    lines.extend([
        "| Judge A | Judge B | Overlap | Exact | Within 1 | QWK |",
        "|---------|---------|---------|-------|----------|-----|",
    ])
    for stats in report["pairwise"].values():
        lines.append(
            f"| {stats['model_a']} | {stats['model_b']} | {stats['n_overlap']} "
            f"| {stats['exact_agreement']:.3f} "
            f"| {stats['agreement_within_1']:.3f} "
            f"| {stats['quadratic_weighted_kappa']:.3f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)

    if not args.judgments.exists():
        print(f"Judgments not found: {args.judgments}")
        return 1
    report = compute_reliability(load_judgments(args.judgments))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.md.write_text(build_markdown(report), encoding="utf-8")
    print(f"Dual-judge reliability -> {args.out}")
    print(f"Markdown -> {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
