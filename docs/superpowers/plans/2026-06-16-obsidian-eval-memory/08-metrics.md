# Task 08 — Metrics Scripts Updates

**Files:**
- Modify: `scripts/eval/compute_ir_metrics.py`
- Modify: `scripts/eval/compute_calibration.py`
- Create: `scripts/eval/hard_negative_analysis.py`

---

## Part A — compute_ir_metrics.py

The existing script already supports `--qrels` and bootstrap CI. We add:
1. `--qrels-tier` flag that tags the report with the tier
2. A stderr warning when tier is not `gold`

- [ ] **Step 1: Add --qrels-tier flag**

In `scripts/eval/compute_ir_metrics.py`, locate the `argparse` block and add:

```python
# Find this block:
parser.add_argument("--qrels", required=True, type=Path, ...)

# Add after it:
parser.add_argument(
    "--qrels-tier",
    choices=["gold", "silver", "bronze"],
    default=None,
    help="Tier label for this qrels file (used for warnings and report tagging).",
)
```

- [ ] **Step 2: Add tier warning to main()**

After parsing args and before running metrics, insert:

```python
if args.qrels_tier and args.qrels_tier != "gold":
    import sys as _sys
    _sys.stderr.write(
        f"WARNING: Using {args.qrels_tier.upper()} qrels. "
        f"Results from {args.qrels_tier} labels should NOT be cited as "
        f"scientific validation. Use gold qrels for whitepaper claims.\n"
    )
```

- [ ] **Step 3: Tag the output report with the tier**

In the section that builds the output dict (look for where `results` or `report` dict is assembled), add:

```python
report["qrels_tier"] = args.qrels_tier or "unknown"
report["qrels_path"] = str(args.qrels)
```

- [ ] **Step 4: Verify all three tier files work**

```bash
python scripts/eval/compute_ir_metrics.py \
    --qrels artifacts/qrels_silver.jsonl \
    --qrels-tier silver \
    --run reports/eval/runs/usefulness.jsonl \
    --out reports/eval/eval_report_silver.json 2>&1 | head -5
```

Expected stderr: `WARNING: Using SILVER qrels ...`

```bash
python scripts/eval/compute_ir_metrics.py \
    --qrels artifacts/qrels_bronze.jsonl \
    --qrels-tier bronze \
    --run reports/eval/runs/usefulness.jsonl \
    --out reports/eval/eval_report_bronze.json 2>&1 | head -5
```

Expected stderr: `WARNING: Using BRONZE qrels ...`

---

## Part B — compute_calibration.py

Same pattern — add `--qrels-tier` and a stderr warning.

- [ ] **Step 5: Add tier flag to compute_calibration.py**

```python
# In the argparse block, add:
parser.add_argument(
    "--qrels-tier",
    choices=["gold", "silver", "bronze"],
    default=None,
)
```

- [ ] **Step 6: Add warning**

```python
if args.qrels_tier and args.qrels_tier != "gold":
    import sys as _sys
    _sys.stderr.write(
        f"WARNING: Calibration computed on {args.qrels_tier.upper()} qrels. "
        f"Do not report these calibration figures as final.\n"
    )
```

---

## Part C — hard_negative_analysis.py (new script)

- [ ] **Step 7: Create `scripts/eval/hard_negative_analysis.py`**

```python
#!/usr/bin/env python3
"""Dedicated hard-negative violation analysis across all qrel tiers.

Produces a report showing:
- Overall HN violation rate per tier
- Violations broken down by query
- Violations broken down by source adapter

Usage:
    python scripts/eval/hard_negative_analysis.py \
        --qrels-gold artifacts/qrels_gold.jsonl \
        --qrels-silver artifacts/qrels_silver.jsonl \
        --qrels-bronze artifacts/qrels_bronze.jsonl \
        --run reports/eval/runs/usefulness.jsonl \
        --out reports/eval/hard_negative_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _load_qrels(path: Path | None) -> dict[str, dict[str, int]]:
    """Return {query_id: {record_id: label}}."""
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qrels[row["query_id"]][row["record_id"]] = int(row["label"])
    return dict(qrels)


def _load_run(path: Path) -> dict[str, list[str]]:
    """Return {query_id: [record_id ordered by rank]}."""
    if not path.exists():
        return {}
    rows_by_q: dict[str, list[tuple[int, str]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows_by_q[row["query_id"]].append(
                (int(row.get("rank", 10**9)), row["record_id"])
            )
    return {
        qid: [rid for _, rid in sorted(rows)]
        for qid, rows in rows_by_q.items()
    }


def _violations(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[str]],
    cutoff: int = 10,
) -> dict:
    """Return per-query violation counts and overall rate."""
    total_queries = 0
    total_hn = 0
    violated_queries = 0
    by_query: dict[str, int] = {}
    by_source: dict[str, int] = defaultdict(int)

    for qid, ranked in run.items():
        q_qrels = qrels.get(qid, {})
        hard_neg_ids = {rid for rid, lbl in q_qrels.items() if lbl == 0}
        if not hard_neg_ids:
            continue
        total_queries += 1
        total_hn += len(hard_neg_ids)
        top_k = ranked[:cutoff]
        violations = [rid for rid in top_k if rid in hard_neg_ids]
        if violations:
            violated_queries += 1
            by_query[qid] = len(violations)
            for rid in violations:
                source = rid.split(":")[0]
                by_source[source] += 1

    rate = violated_queries / total_queries if total_queries else 0.0
    return {
        "total_queries_with_hn": total_queries,
        "total_hard_negatives": total_hn,
        "violated_queries": violated_queries,
        "violation_rate": round(rate, 4),
        "by_query": by_query,
        "by_source": dict(by_source),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels-gold", type=Path, default=None)
    parser.add_argument("--qrels-silver", type=Path, default=None)
    parser.add_argument("--qrels-bronze", type=Path, default=None)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--cutoff", type=int, default=10)
    args = parser.parse_args()

    run = _load_run(args.run)
    report = {"cutoff": args.cutoff}

    for tier_name, tier_path in [
        ("gold", args.qrels_gold),
        ("silver", args.qrels_silver),
        ("bronze", args.qrels_bronze),
    ]:
        qrels = _load_qrels(tier_path)
        if qrels:
            report[tier_name] = _violations(qrels, run, args.cutoff)
        else:
            report[tier_name] = {"note": f"{tier_name} qrels not available"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Hard-negative report → {args.out}")

    # Print summary
    for tier in ("gold", "silver", "bronze"):
        if "violation_rate" in report.get(tier, {}):
            vr = report[tier]["violation_rate"]
            vq = report[tier]["violated_queries"]
            tq = report[tier]["total_queries_with_hn"]
            print(f"  {tier.upper():8s}: violation_rate={vr:.1%}  ({vq}/{tq} queries)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Smoke-test (will warn if run files are missing)**

```bash
python scripts/eval/hard_negative_analysis.py \
    --qrels-gold artifacts/qrels_gold.jsonl \
    --qrels-silver artifacts/qrels_silver.jsonl \
    --qrels-bronze artifacts/qrels_bronze.jsonl \
    --run reports/eval/runs/usefulness.jsonl \
    --out reports/eval/hard_negative_report.json 2>&1
```

Expected: writes JSON report, prints per-tier summary rows.

- [ ] **Step 9: Re-enable test_search_quality.py**

Now that `compute_hard_negative_violations` lives in `neural_search.eval.label_ensemble`,
update the import stub added in Task 01:

```python
# Replace the pytest.mark.skip block added in Task 01 with:
from neural_search.eval.label_ensemble import compute_hard_negative_violations

# Then remove the pytestmark = pytest.mark.skip line.
# Also update any remaining references to the old RelevanceJudgment / RelevanceLabelSet types.
# Those tests can be removed or replaced with equivalent assertions using the new qrel dicts.
```

Run the test to verify it passes (or remove tests that can't be ported):

```bash
pytest tests/test_search_quality.py -v
```

- [ ] **Step 10: Commit**

```bash
git add scripts/eval/compute_ir_metrics.py \
    scripts/eval/compute_calibration.py \
    scripts/eval/hard_negative_analysis.py \
    tests/test_search_quality.py
git commit -m "feat(eval): metrics tier support + hard_negative_analysis.py"
```
