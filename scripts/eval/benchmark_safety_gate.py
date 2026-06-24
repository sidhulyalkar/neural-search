"""Benchmark safety gate.

Scans reports/eval/ for documents that claim gold/NDCG/MRR metrics
when gold qrels are empty or when the source corpus was a stale snapshot.
Exits 0 if safe; exits 1 if blockers found.

Usage:
    python scripts/eval/benchmark_safety_gate.py           # fail on blockers
    python scripts/eval/benchmark_safety_gate.py --warn-only  # print but do not fail
    python scripts/eval/benchmark_safety_gate.py --annotate   # add STALE banners to reports

Also callable from pytest via test_benchmark_safety_gate.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports/eval"
GOLD_PATH = ROOT / "artifacts/qrels_gold.jsonl"
ADJ_PATH = ROOT / "artifacts/field_state/adjudicated_qrels.jsonl"
MANIFEST_PATH = ROOT / "reports/eval/current_artifact_manifest.json"

# Patterns that indicate claimed-gold or peer-review-grade metrics
GOLD_CLAIM_PATTERNS = [
    re.compile(r"\bNDCG@\d+\s*[=>]\s*[\d.]+", re.I),
    re.compile(r"\bMRR\s*[=>]\s*[\d.]+", re.I),
    re.compile(r"\bRecall@\d+\s*[=>]\s*[\d.]+", re.I),
    re.compile(r"\b675\s+annotated", re.I),
    re.compile(r"\bgold\s+benchmark", re.I),
    re.compile(r"\bpeer[- ]reviewed?\s+results?", re.I),
]

# Stale corpus markers
STALE_CORPUS_PATTERNS = [
    re.compile(r"\b10[,.]?404\b"),
    re.compile(r"corpus_size.*10404", re.I),
    re.compile(r"indexed_ids.*10404", re.I),
]

STALE_BANNER = (
    "\n---\n"
    "⚠️  **STALE ARTIFACT BANNER** — This report was generated from a historical snapshot "
    "(10,404-record corpus or 675-pair qrels estimate) and should NOT be cited as current "
    "or peer-reviewed evidence. Current source of truth: "
    "`reports/eval/current_artifact_manifest.json`.\n"
    "---\n"
)

EVIDENCE_STATUS_LABELS = {
    "gold": "human gold — peer-review grade",
    "human_adjudicated_smoke": "human-adjudicated smoke test (3 pairs) — workflow validated only",
    "silver_diagnostic": "LLM-generated silver diagnostic — not peer-reviewed",
    "historical_stale": "historical stale snapshot — do not cite",
    "engineering_validation": "engineering validation — not retrieval evidence",
}


def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def classify_report(path: Path, text: str) -> str:
    """Assign an evidence status label to a report file."""
    name = path.name.lower()
    if any(p.search(text) for p in STALE_CORPUS_PATTERNS):
        return "historical_stale"
    if "ndcg" in name or "mrr" in name or "retrieval_metrics" in name:
        if count_lines(GOLD_PATH) == 0:
            return "silver_diagnostic"
        return "gold"
    if "recall" in name and "turbovec" in text.lower():
        return "engineering_validation"
    if "silver" in name or "bronze" in name:
        return "silver_diagnostic"
    return "engineering_validation"


def scan_report(path: Path) -> dict:
    """Scan a single report file. Returns a finding dict."""
    try:
        text = path.read_text(errors="replace")
    except Exception as e:
        return {"path": str(path.relative_to(ROOT)), "error": str(e), "issues": []}

    issues = []
    gold_rows = count_lines(GOLD_PATH)

    for pat in GOLD_CLAIM_PATTERNS:
        m = pat.search(text)
        if m:
            issues.append({
                "type": "gold_claim_while_empty",
                "match": m.group(0),
                "gold_rows": gold_rows,
                "severity": "BLOCKER" if gold_rows == 0 else "WARN",
            })

    for pat in STALE_CORPUS_PATTERNS:
        m = pat.search(text)
        if m:
            issues.append({
                "type": "stale_corpus_reference",
                "match": m.group(0),
                "severity": "WARN",
            })

    status = classify_report(path, text)
    return {
        "path": str(path.relative_to(ROOT)),
        "evidence_status": status,
        "evidence_label": EVIDENCE_STATUS_LABELS.get(status, status),
        "issues": issues,
    }


def annotate_report(path: Path) -> None:
    """Prepend a STALE banner to a Markdown report if it contains stale markers."""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return
    if STALE_BANNER.strip() in text:
        return
    if any(p.search(text) for p in STALE_CORPUS_PATTERNS):
        with open(path, "w") as f:
            f.write(STALE_BANNER + text)
        print(f"  Annotated: {path.relative_to(ROOT)}")


def run_gate(warn_only: bool = False, annotate: bool = False) -> dict:
    md_reports = list(REPORTS_DIR.glob("*.md"))
    json_reports = list(REPORTS_DIR.glob("*.json"))

    findings = []
    blockers = []
    warnings_list = []

    for p in md_reports + json_reports:
        if p.name in ("qrels_progress_report.md", "qrels_progress_report.json",
                      "current_artifact_manifest.json"):
            continue
        finding = scan_report(p)
        findings.append(finding)
        for issue in finding.get("issues", []):
            if issue.get("severity") == "BLOCKER":
                blockers.append({"file": finding["path"], **issue})
            else:
                warnings_list.append({"file": finding["path"], **issue})
        if annotate and p.suffix == ".md":
            annotate_report(p)

    gold_rows = count_lines(GOLD_PATH)
    adj_rows = count_lines(ADJ_PATH)

    result = {
        "generated_at": datetime.now(UTC).isoformat(),
        "gold_qrels_rows": gold_rows,
        "adj_qrels_rows": adj_rows,
        "safe": len(blockers) == 0,
        "blockers": blockers,
        "warnings": warnings_list,
        "report_findings": findings,
        "evidence_status_labels": EVIDENCE_STATUS_LABELS,
    }

    out_path = REPORTS_DIR / "benchmark_safety_gate_report.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warn-only", action="store_true", help="Print findings but exit 0")
    parser.add_argument("--annotate", action="store_true", help="Add STALE banners to stale .md reports")
    args = parser.parse_args(argv)

    result = run_gate(warn_only=args.warn_only, annotate=args.annotate)

    print(f"Benchmark Safety Gate — {result['generated_at']}")
    print(f"  Gold qrels: {result['gold_qrels_rows']} rows")
    print(f"  Adjudicated: {result['adj_qrels_rows']} rows")
    print()

    if result["blockers"]:
        print(f"❌ BLOCKERS ({len(result['blockers'])}):")
        for b in result["blockers"]:
            print(f"   {b['file']}: {b['type']} — '{b.get('match', '')}' (gold rows={b.get('gold_rows')})")
    else:
        print("✅ No BLOCKER-level issues found.")

    if result["warnings"]:
        print(f"\n⚠  Warnings ({len(result['warnings'])}):")
        for w in result["warnings"]:
            print(f"   {w['file']}: {w['type']} — '{w.get('match', '')}'")

    print(f"\nReport → reports/eval/benchmark_safety_gate_report.json")

    if result["blockers"] and not args.warn_only:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
