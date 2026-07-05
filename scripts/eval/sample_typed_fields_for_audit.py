"""Typed-field extraction audit harness.

Mirrors scripts/eval/sample_findings_for_audit.py but reviews the 27 typed
fields from neural_search.literature.typed_finding_extractor (frequency
band, temporal pattern, spatial frame, negation, condition, effect scale,
etc.) instead of the core region/task/modality/species/direction fields.

This is the audit scaffold for the "30-50 gold-reviewed finding examples"
step called for in reports/strategy/brainknow_comparison_publishable_plan.md
(Milestone 2) and reports/strategy/next_gen_kg_development_plan.md (Phase 2)
— it produces the sample and review template; a human reviewer (not this
script) fills in correctness judgments.

Samples findings stratified by which typed fields are populated, biased
toward records with at least one non-empty typed field so the audit reviews
actual extractor output rather than mostly-empty rows.

Usage:
    python scripts/eval/sample_typed_fields_for_audit.py
    python scripts/eval/sample_typed_fields_for_audit.py --n 50
    python scripts/eval/sample_typed_fields_for_audit.py --seed 7
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neural_search.literature.typed_finding_extractor import (
    enrich_finding,  # noqa: E402
)

FINDINGS_PATH = ROOT / "artifacts/literature/findings_tier1_ollama.jsonl"
AUDIT_JSONL = ROOT / "reports/eval/typed_field_audit_sample.jsonl"
AUDIT_CSV = ROOT / "reports/eval/typed_field_audit_template.csv"
AUDIT_INSTRUCTIONS = ROOT / "reports/eval/typed_field_audit_instructions.md"

# The fields actually promoted to graph edges (kg_builder.add_findings_to_graph)
# plus negation (consulted by relationship_builder's supports/contradicts logic)
# are reviewed first; the remaining typed fields ride along for context.
PRIMARY_AUDIT_FIELDS = ["negation", "frequency_band", "temporal_pattern", "spatial_frame"]
CONTEXT_AUDIT_FIELDS = ["condition", "effect_scale", "behavioral_measure", "population_type"]

AUDIT_CSV_COLUMNS = [
    "finding_id",
    "paper_id",
    "finding_text",
    *PRIMARY_AUDIT_FIELDS,
    *CONTEXT_AUDIT_FIELDS,
    "human_correct",  # TRUE / FALSE / PARTIAL
    "wrong_fields",  # space-separated subset of the field columns above
    "notes",
]

INSTRUCTIONS_TEXT = """# Typed Field Extraction Audit Instructions

## Purpose
Evaluate the precision of the 27 rule-based typed fields in
`neural_search/literature/typed_finding_extractor.py`, focusing first on the
four fields wired into the knowledge graph and relationship builder:
`negation`, `frequency_band`, `temporal_pattern`, `spatial_frame`.

## File to Fill
`reports/eval/typed_field_audit_template.csv`

## Columns

| Column | Description |
|--------|-------------|
| `finding_id` / `paper_id` | Identifiers — do not change |
| `finding_text` | The extracted finding sentence |
| `negation` | True if the extractor flagged this as a negated finding |
| `frequency_band` | Extracted frequency band(s), e.g. theta, gamma |
| `temporal_pattern` | Extracted temporal pattern(s), e.g. transient, oscillatory |
| `spatial_frame` | Extracted spatial frame, e.g. local, inter_regional |
| `condition` / `effect_scale` / `behavioral_measure` / `population_type` | Context fields — review only if time permits |
| `human_correct` | Your judgment: `TRUE` / `FALSE` / `PARTIAL` |
| `wrong_fields` | If not TRUE, which column(s) were wrong (space-separated column names) |
| `notes` | Free text |

## Guidelines

1. Read `finding_text` and judge whether each populated field is actually
   correct for that sentence — not whether it is plausible in general.
2. An empty field is not necessarily wrong; only mark `wrong_fields` for
   fields that are populated but incorrect, or that should have matched but
   didn't (note the latter in `notes`, since `wrong_fields` is for false
   positives the column scheme can track).
3. Pay special attention to `negation` — this field gates whether a finding
   counts as supporting evidence in `relationship_builder.py`. A missed
   negation (should be True but is False) is the most consequential error
   type to flag in `notes`.

## Target
- 80%+ precision on the four primary fields -> safe to keep wiring them into
  the graph as-is.
- 60-80% -> usable but flag the dominant failure pattern for a targeted
  lexicon fix before expanding further (see
  docs/superpowers/plans/2026-06-22-typed-field-coverage-relationship-expansion.md).
- <60% on a field -> stop promoting that field into graph edges until fixed.

## Contact
Sid: sid.soccer.21@gmail.com
"""


def load_findings(path: Path, max_records: int = 200_000) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            if line.strip():
                records.append(json.loads(line))
    return records


def _populated_primary_fields(record: dict) -> tuple[str, ...]:
    populated = []
    for field in PRIMARY_AUDIT_FIELDS:
        value = record.get(field)
        if field == "negation":
            if value:
                populated.append(field)
        elif value:
            populated.append(field)
    return tuple(sorted(populated))


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    """Sample, biased toward records with at least one populated primary field."""
    rng = random.Random(seed)

    with_signal = [r for r in records if _populated_primary_fields(r)]
    without_signal = [r for r in records if not _populated_primary_fields(r)]

    buckets: dict[tuple[str, ...], list[dict]] = {}
    for r in with_signal:
        buckets.setdefault(_populated_primary_fields(r), []).append(r)

    bucket_names = list(buckets.keys())
    rng.shuffle(bucket_names)
    selected: list[dict] = []
    target_with_signal = min(len(with_signal), int(n * 0.8))
    per_bucket = max(1, target_with_signal // max(len(bucket_names), 1))
    for bucket in bucket_names:
        pool = buckets[bucket]
        rng.shuffle(pool)
        selected.extend(pool[:per_bucket])
        if len(selected) >= target_with_signal:
            break

    rng.shuffle(without_signal)
    selected.extend(without_signal[: max(0, n - len(selected))])

    rng.shuffle(selected)
    return selected[:n]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fmt(value) -> str:
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    if isinstance(value, bool):
        return "TRUE" if value else ""
    return str(value) if value else ""


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_CSV_COLUMNS)
        writer.writeheader()
        for r in records:
            row = {
                "finding_id": r.get("finding_id", ""),
                "paper_id": r.get("paper_id", ""),
                "finding_text": (r.get("finding_text") or "")[:300],
                "human_correct": "",
                "wrong_fields": "",
                "notes": "",
            }
            for field in (*PRIMARY_AUDIT_FIELDS, *CONTEXT_AUDIT_FIELDS):
                row[field] = _fmt(r.get(field))
            writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40, help="Number of findings to sample (default 40)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args(argv)

    if not FINDINGS_PATH.exists():
        print(f"x Findings file not found: {FINDINGS_PATH}")
        print("  Wait for extraction job to complete before running this script.")
        return

    print(f"Loading findings from {FINDINGS_PATH.relative_to(ROOT)} ...")
    raw_records = load_findings(FINDINGS_PATH)
    records = [enrich_finding(r) for r in raw_records]
    print(f"  {len(records):,} total findings (enriched with typed fields)")

    with_signal = sum(1 for r in records if _populated_primary_fields(r))
    print(f"  {with_signal:,} findings have at least one primary typed field populated")

    sample = stratified_sample(records, n=args.n, seed=args.seed)
    print(f"  Sampled {len(sample)} findings (stratified, seed={args.seed})")

    write_jsonl(sample, AUDIT_JSONL)
    write_csv(sample, AUDIT_CSV)
    AUDIT_INSTRUCTIONS.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_INSTRUCTIONS.write_text(INSTRUCTIONS_TEXT, encoding="utf-8")

    print(f"JSONL        -> {AUDIT_JSONL.relative_to(ROOT)}")
    print(f"CSV          -> {AUDIT_CSV.relative_to(ROOT)}")
    print(f"Instructions -> {AUDIT_INSTRUCTIONS.relative_to(ROOT)}")
    print("\nFill in human_correct, wrong_fields, notes columns in the CSV.")


if __name__ == "__main__":
    main()
