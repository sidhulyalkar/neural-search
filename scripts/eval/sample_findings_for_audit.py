"""Finding extraction audit harness.

Samples 100 findings from artifacts/literature/findings_tier1_ollama.jsonl,
stratified by region/task/modality/species/result_direction/confidence,
and produces:
  - reports/eval/finding_audit_sample.jsonl   (full records for reference)
  - reports/eval/finding_audit_template.csv   (blank audit columns for manual review)

Usage:
    python scripts/eval/sample_findings_for_audit.py
    python scripts/eval/sample_findings_for_audit.py --n 50
    python scripts/eval/sample_findings_for_audit.py --seed 99
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FINDINGS_PATH = ROOT / "artifacts/literature/findings_tier1_ollama.jsonl"
AUDIT_JSONL = ROOT / "reports/eval/finding_audit_sample.jsonl"
AUDIT_CSV = ROOT / "reports/eval/finding_audit_template.csv"
AUDIT_INSTRUCTIONS = ROOT / "reports/eval/finding_audit_instructions.md"

AUDIT_CSV_COLUMNS = [
    "finding_id",
    "paper_id",
    "finding_text",
    "regions",
    "tasks",
    "modalities",
    "species",
    "result_direction",
    "confidence",
    "human_correct",       # TRUE / FALSE / PARTIAL
    "error_type",          # region_wrong / task_wrong / direction_wrong / hallucinated / species_wrong / none
    "notes",
]

INSTRUCTIONS_TEXT = """# Finding Extraction Audit Instructions

## Purpose
Evaluate the precision of LLM-extracted findings from `artifacts/literature/findings_tier1_ollama.jsonl`.
This audit determines whether extraction quality is sufficient to cite in the Neural Search whitepaper.

## File to Fill
`reports/eval/finding_audit_template.csv`

## Columns

| Column | Description | Values |
|--------|-------------|--------|
| `finding_id` | Unique ID — do not change | — |
| `paper_id` | OpenAlex paper ID — do not change | — |
| `finding_text` | Extracted finding sentence | — |
| `regions` | Extracted brain region(s) | — |
| `tasks` | Extracted task(s) | — |
| `modalities` | Extracted modality/modalities | — |
| `species` | Extracted species | — |
| `result_direction` | increase / decrease / no_change / correlation / mixed | — |
| `confidence` | Extractor confidence [0–1] | — |
| `human_correct` | Your judgment | `TRUE` / `FALSE` / `PARTIAL` |
| `error_type` | If not TRUE, what is wrong | `region_wrong`, `task_wrong`, `direction_wrong`, `hallucinated`, `species_wrong`, `none` |
| `notes` | Free text | any |

## Guidelines

1. Open the paper (use `paper_id` to look up on OpenAlex: https://openalex.org/works/<paper_id>).
2. Find the relevant passage in the abstract or methods.
3. Assess whether:
   - The finding text faithfully represents what the paper reports.
   - The brain region(s) are correct and specific.
   - The task is correctly identified.
   - The result direction (increase/decrease/no_change) is correct.
   - The species is correct.
4. Mark `human_correct`:
   - `TRUE` — all fields accurate.
   - `PARTIAL` — some fields correct, one minor error.
   - `FALSE` — major error or hallucination.
5. If `FALSE` or `PARTIAL`, fill `error_type` with the primary error category.
6. Record any notes in the `notes` column.

## Target
- 80%+ precision → findings acceptable for whitepaper claims with caveats.
- 60–80% → report precision with specific failure modes; use as audit queue only.
- <60% → extraction quality insufficient; re-run with improved prompt.

## Contact
Sid: sid.soccer.21@gmail.com
"""


def load_findings(path: Path, max_records: int = 200_000) -> list[dict]:
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_records:
                break
            if line.strip():
                records.append(json.loads(line))
    return records


def stratified_sample(records: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)

    def bucket(r: dict) -> str:
        regions = (r.get("regions") or [])
        tasks = (r.get("tasks") or [])
        modalities = (r.get("modalities") or [])
        species = (r.get("species") or [])
        direction = r.get("result_direction") or "unknown"
        conf = r.get("confidence", 0.5)
        conf_bucket = "high" if conf >= 0.8 else "medium" if conf >= 0.5 else "low"
        region = regions[0] if regions else "no_region"
        task = tasks[0] if tasks else "no_task"
        return f"{region}|{task}|{direction}|{conf_bucket}"

    buckets: dict[str, list[dict]] = {}
    for r in records:
        b = bucket(r)
        buckets.setdefault(b, []).append(r)

    # Take roughly equal from each bucket until we have n
    bucket_names = list(buckets.keys())
    rng.shuffle(bucket_names)
    selected: list[dict] = []
    per_bucket = max(1, n // len(bucket_names))
    for b in bucket_names:
        pool = buckets[b]
        rng.shuffle(pool)
        selected.extend(pool[:per_bucket])
        if len(selected) >= n:
            break
    # Fill remainder from any bucket
    if len(selected) < n:
        remaining = [r for r in records if r not in set(map(id, selected))]
        rng.shuffle(remaining)
        selected.extend(remaining[: n - len(selected)])
    rng.shuffle(selected)
    return selected[:n]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_csv(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_CSV_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "finding_id": r.get("finding_id", ""),
                "paper_id": r.get("paper_id", ""),
                "finding_text": (r.get("finding_text") or "")[:300],
                "regions": "; ".join(r.get("regions") or []),
                "tasks": "; ".join(r.get("tasks") or []),
                "modalities": "; ".join(r.get("modalities") or []),
                "species": "; ".join(r.get("species") or []),
                "result_direction": r.get("result_direction") or "",
                "confidence": r.get("confidence") or "",
                "human_correct": "",
                "error_type": "",
                "notes": "",
            })


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="Number of findings to sample (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    args = parser.parse_args(argv)

    if not FINDINGS_PATH.exists():
        print(f"✗ Findings file not found: {FINDINGS_PATH}")
        print("  Wait for extraction job to complete before running this script.")
        return

    print(f"Loading findings from {FINDINGS_PATH.relative_to(ROOT)} ...")
    records = load_findings(FINDINGS_PATH)
    print(f"  {len(records):,} total findings")

    # Quick stats
    directions = Counter(r.get("result_direction") or "unknown" for r in records)
    has_region = sum(1 for r in records if r.get("regions"))
    has_task = sum(1 for r in records if r.get("tasks"))
    print(f"  has_region={has_region:,}  has_task={has_task:,}  directions={dict(directions)}")

    sample = stratified_sample(records, n=args.n, seed=args.seed)
    print(f"  Sampled {len(sample)} findings (stratified, seed={args.seed})")

    write_jsonl(sample, AUDIT_JSONL)
    write_csv(sample, AUDIT_CSV)
    with open(AUDIT_INSTRUCTIONS, "w") as f:
        f.write(INSTRUCTIONS_TEXT)

    print(f"✓ JSONL   → {AUDIT_JSONL.relative_to(ROOT)}")
    print(f"✓ CSV     → {AUDIT_CSV.relative_to(ROOT)}")
    print(f"✓ Guide   → {AUDIT_INSTRUCTIONS.relative_to(ROOT)}")
    print("\nFill in human_correct, error_type, notes columns in the CSV.")


if __name__ == "__main__":
    main()
