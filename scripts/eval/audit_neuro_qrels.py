"""Audit neuro_judge labels — with or without human gold calibration.

Two modes:
  1. Self-analysis (no --human):   label/confidence/failure-mode distributions
  2. Calibration (--human labels): exact agreement, within-1, QWK, confusion matrix

Usage::

    # Self-analysis only
    python scripts/eval/audit_neuro_qrels.py \
        --judgments artifacts/field_state/neuro_qrels_consensus.jsonl \
        --out reports/eval/neuro_judge_audit.md

    # With human calibration
    python scripts/eval/audit_neuro_qrels.py \
        --judgments artifacts/field_state/neuro_qrels_consensus.jsonl \
        --human artifacts/field_state/adjudicated_qrels.jsonl \
        --out reports/eval/neuro_judge_human_calibration.md

    # With conflict file
    python scripts/eval/audit_neuro_qrels.py \
        --judgments artifacts/field_state/neuro_qrels_consensus.jsonl \
        --conflicts artifacts/field_state/neuro_qrels_conflicts.jsonl \
        --out reports/eval/neuro_judge_consensus_report.md
"""
# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from neural_search.eval.neuro_judge.calibration import calibrate  # noqa: E402
from neural_search.eval.neuro_judge.evidence_packet import NEURO_JUDGE_WATERMARK  # noqa: E402


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Self-analysis helpers
# ---------------------------------------------------------------------------


def _label_distribution(records: list[dict]) -> dict[int, int]:
    cnt: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in records:
        lbl = r.get("label", r.get("relevance"))
        if lbl is not None:
            cnt[int(lbl)] = cnt.get(int(lbl), 0) + 1
    return cnt


def _confidence_buckets(records: list[dict]) -> dict[str, int]:
    buckets: dict[str, int] = {"<0.5": 0, "0.5–0.7": 0, "0.7–0.9": 0, ">=0.9": 0}
    for r in records:
        c = float(r.get("confidence", 0.5))
        if c < 0.5:
            buckets["<0.5"] += 1
        elif c < 0.7:
            buckets["0.5–0.7"] += 1
        elif c < 0.9:
            buckets["0.7–0.9"] += 1
        else:
            buckets[">=0.9"] += 1
    return buckets


def _top_failure_modes(records: list[dict], n: int = 10) -> list[tuple[str, int]]:
    cnt: Counter = Counter()
    for r in records:
        for fm in r.get("failure_modes", []):
            cnt[str(fm)] += 1
    return cnt.most_common(n)


def _evidence_completeness_buckets(records: list[dict]) -> dict[str, int]:
    buckets = {"0": 0, "(0,0.5)": 0, "[0.5,0.8)": 0, "[0.8,1.0)": 0, "1.0": 0}
    for r in records:
        value = float(r.get("evidence_completeness", 0.0))
        if value == 0:
            buckets["0"] += 1
        elif value < 0.5:
            buckets["(0,0.5)"] += 1
        elif value < 0.8:
            buckets["[0.5,0.8)"] += 1
        elif value < 1.0:
            buckets["[0.8,1.0)"] += 1
        else:
            buckets["1.0"] += 1
    return buckets


def _missing_dimension_counts(records: list[dict]) -> list[tuple[str, int]]:
    cnt: Counter = Counter()
    for r in records:
        for dim in r.get("required_dimensions_missing", []):
            cnt[str(dim)] += 1
    return cnt.most_common()


def _examples_by_label(records: list[dict], label: int, max_ex: int = 3) -> list[dict]:
    return [r for r in records if r.get("label", r.get("relevance")) == label][:max_ex]


def _high_label_missing_info(records: list[dict], max_ex: int = 5) -> list[dict]:
    return [
        r for r in records
        if r.get("label", 0) >= 2 and len(r.get("missing_information", [])) >= 2
    ][:max_ex]


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def _fmt_label_dist(dist: dict[int, int], total: int) -> str:
    lines = ["| Label | Count | % |", "|-------|-------|---|"]
    for lbl in [0, 1, 2, 3]:
        n = dist.get(lbl, 0)
        lines.append(f"| {lbl} | {n} | {100*n//max(total,1)}% |")
    return "\n".join(lines)


def _fmt_conf_dist(buckets: dict[str, int], total: int) -> str:
    lines = ["| Range | Count | % |", "|-------|-------|---|"]
    for k, v in buckets.items():
        lines.append(f"| {k} | {v} | {100*v//max(total,1)}% |")
    return "\n".join(lines)


def _fmt_failure_modes(modes: list[tuple[str, int]]) -> str:
    if not modes:
        return "None detected."
    lines = ["| Mode | Count |", "|------|-------|"]
    for mode, cnt in modes:
        lines.append(f"| {mode} | {cnt} |")
    return "\n".join(lines)


def _fmt_examples(examples: list[dict], title: str) -> str:
    if not examples:
        return f"### {title}\n\nNone.\n"
    lines = [f"### {title}\n"]
    for e in examples:
        qid = e.get("query_id", "?")
        did = e.get("dataset_id", "?")
        lbl = e.get("label", e.get("relevance", "?"))
        conf = e.get("confidence", "?")
        rationale = str(e.get("rationale_short", ""))[:120]
        missing = e.get("missing_information", [])
        lines.append(f"**{qid} / {did}**")
        lines.append(f"- Label: {lbl} | Confidence: {conf}")
        lines.append(f"- Rationale: {rationale}")
        if missing:
            lines.append(f"- Missing: {', '.join(str(m) for m in missing[:4])}")
        lines.append("")
    return "\n".join(lines)


def _fmt_confusion(matrix: list[list[int]]) -> str:
    header = "         | pred 0 | pred 1 | pred 2 | pred 3 |"
    sep = "---------|--------|--------|--------|--------|"
    rows = [header, sep]
    for i, row in enumerate(matrix):
        cells = " | ".join(f"{v:6d}" for v in row)
        rows.append(f"  true {i} | {cells} |")
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------


def build_audit_report(
    records: list[dict],
    conflict_records: list[dict],
    human_records: list[dict] | None,
    judge_path: str,
    human_path: str | None,
    conflict_path: str | None,
) -> str:
    total = len(records)
    label_dist = _label_distribution(records)
    conf_buckets = _confidence_buckets(records)
    failure_modes = _top_failure_modes(records)
    hn_count = sum(1 for r in records if r.get("hard_negative_detected"))
    abstain_count = sum(1 for r in records if r.get("abstain_recommended"))
    completeness_buckets = _evidence_completeness_buckets(records)
    missing_dimensions = _missing_dimension_counts(records)

    # Check for suspicious patterns
    red_flags: list[str] = []
    n0 = label_dist.get(0, 0)
    n3 = label_dist.get(3, 0)
    n2 = label_dist.get(2, 0)
    if n0 == 0 and total > 10:
        red_flags.append("No label-0 records — hard negatives may be under-detected")
    if n3 > n2 and total > 10:
        red_flags.append("More label-3 than label-2 — possible over-crediting")
    if total > 0 and n2 / max(total, 1) > 0.7:
        red_flags.append("Over 70% label-2 — possible label collapse")
    conf_pct_high = conf_buckets.get(">=0.9", 0) / max(total, 1)
    if conf_pct_high > 0.6 and total > 10:
        red_flags.append("Over 60% records have confidence ≥0.9 — possible overconfidence")

    sections = [
        "# Neuro-Judge Audit Report",
        "",
        f"> {NEURO_JUDGE_WATERMARK}",
        "",
        f"**Judgment file**: `{judge_path}`  ",
    ]
    if human_path:
        sections.append(f"**Human labels**: `{human_path}`  ")
    if conflict_path:
        sections.append(f"**Conflicts file**: `{conflict_path}`  ")
    sections += [
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total pairs | {total} |",
        f"| Conflicts | {len(conflict_records)} |",
        f"| Hard-negative detected | {hn_count} |",
        f"| Abstain recommended | {abstain_count} |",
    ]

    if red_flags:
        sections += [
            "",
            "## ⚠ Red Flags",
            "",
        ]
        for flag in red_flags:
            sections.append(f"- {flag}")

    sections += [
        "",
        "## Label Distribution",
        "",
        _fmt_label_dist(label_dist, total),
        "",
        "## Confidence Distribution",
        "",
        _fmt_conf_dist(conf_buckets, total),
        "",
        "## Evidence Completeness",
        "",
        _fmt_conf_dist(completeness_buckets, total),
        "",
        "## Missing Required Dimensions",
        "",
        _fmt_failure_modes(missing_dimensions),
        "",
        "## Top Failure Modes",
        "",
        _fmt_failure_modes(failure_modes),
        "",
        _fmt_examples(_examples_by_label(records, 0), "Label 0 Examples"),
        _fmt_examples(_examples_by_label(records, 1), "Label 1 Examples"),
        _fmt_examples(_examples_by_label(records, 2), "Label 2 Examples"),
        _fmt_examples(_examples_by_label(records, 3), "Label 3 Examples"),
        _fmt_examples(
            [r for r in records if float(r.get("confidence", 0)) >= 0.9],
            "High-Confidence Examples (≥0.9)",
        ),
        _fmt_examples(
            [r for r in records if float(r.get("confidence", 0)) < 0.5],
            "Low-Confidence Examples (<0.5)",
        ),
        _fmt_examples(_high_label_missing_info(records), "High Label but Severe Missing Information"),
    ]

    # Conflict section
    if conflict_records:
        sections += [
            "## Conflict Analysis",
            "",
            f"Total conflicts: {len(conflict_records)}",
            "",
            "### Conflict Reasons",
            "",
        ]
        reasons = Counter(r.get("conflict_reason", "unknown") for r in conflict_records)
        for reason, cnt in reasons.most_common():
            sections.append(f"- {reason}: {cnt}")
        sections.append("")

    # Human calibration section
    if human_records is not None:
        sections += ["## Human Calibration", ""]
        if not human_records:
            sections.append("No human labels found. Calibration skipped.\n")
        else:
            # Normalise: prefer numeric 'relevance' field over string 'label' field
            for h in human_records:
                if "relevance" in h:
                    h["label"] = int(h["relevance"])
                elif isinstance(h.get("label"), str):
                    _str_map = {
                        "not_relevant": 0, "weakly_relevant": 1,
                        "partially_relevant": 2, "highly_relevant": 3,
                    }
                    h["label"] = _str_map.get(h["label"], 0)
            report = calibrate(records, human_records)
            sections += [
                "| Metric | Value |",
                "|--------|-------|",
                f"| Pairs evaluated | {report.n_pairs} |",
                f"| Exact agreement | {report.exact_agreement:.3f} |",
                f"| Agreement within 1 | {report.agreement_within_1:.3f} |",
                f"| QWK | {report.quadratic_weighted_kappa:.3f} |",
                "",
                "### Confusion Matrix",
                "",
                "```",
                _fmt_confusion(report.confusion_matrix),
                "```",
                "",
            ]
            if report.false_high_examples:
                sections.append("### False-High Examples (judge > human)\n")
                for m in report.false_high_examples[:5]:
                    sections.append(
                        f"- {m.query_id}/{m.dataset_id}: judge={m.judge_label} human={m.human_label}"
                    )
                sections.append("")
            if report.false_low_examples:
                sections.append("### False-Low Examples (judge < human)\n")
                for m in report.false_low_examples[:5]:
                    sections.append(
                        f"- {m.query_id}/{m.dataset_id}: judge={m.judge_label} human={m.human_label}"
                    )
                sections.append("")
    else:
        sections += [
            "## Human Calibration",
            "",
            "No human labels provided. Run with `--human <path>` to enable calibration.",
            "",
        ]

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit neuro_judge labels")
    parser.add_argument(
        "--judgments", "--judge-labels",
        dest="judgments",
        default="artifacts/field_state/neuro_qrels_consensus.jsonl",
    )
    parser.add_argument(
        "--human", "--human-labels",
        dest="human",
        default=None,
        help="Optional human gold labels for calibration",
    )
    parser.add_argument(
        "--conflicts",
        default=None,
        help="Optional conflicts JSONL",
    )
    parser.add_argument(
        "--out", "--output",
        dest="output",
        default="reports/eval/neuro_judge_audit.md",
    )
    args = parser.parse_args(argv)

    j_path = _REPO / args.judgments
    if not j_path.exists():
        sys.exit(f"[ERROR] Judgments file not found: {j_path}")

    print(f"Loading judgments from {j_path}...")
    judge_records = _load_jsonl(j_path)

    conflict_records: list[dict] = []
    if args.conflicts:
        conf_path = _REPO / args.conflicts
        if conf_path.exists():
            conflict_records = _load_jsonl(conf_path)

    human_records: list[dict] | None = None
    if args.human:
        h_path = _REPO / args.human
        if h_path.exists():
            print(f"Loading human labels from {h_path}...")
            human_records = _load_jsonl(h_path)
        else:
            print(f"[WARN] Human labels not found: {h_path}", file=sys.stderr)
            human_records = []

    out_path = _REPO / args.output
    md = build_audit_report(
        judge_records,
        conflict_records,
        human_records,
        args.judgments,
        args.human,
        args.conflicts,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"Audit report written to {out_path}")

    # Print summary stats
    total = len(judge_records)
    lbl_dist = _label_distribution(judge_records)
    print(f"Total: {total} | Labels: 0={lbl_dist[0]} 1={lbl_dist[1]} 2={lbl_dist[2]} 3={lbl_dist[3]}")


if __name__ == "__main__":
    main()
