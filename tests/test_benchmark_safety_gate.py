"""Tests for the benchmark safety gate."""
# Adjust import path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval.benchmark_safety_gate import (
    GOLD_PATH,
    ROOT,
    count_lines,
    run_gate,
    scan_report,
)


def test_gate_runs_and_returns_dict():
    result = run_gate(warn_only=True)
    assert isinstance(result, dict)


def test_gate_has_required_keys():
    result = run_gate(warn_only=True)
    for key in ("generated_at", "gold_qrels_rows", "safe", "blockers", "warnings", "report_findings"):
        assert key in result, f"Missing key: {key}"


def test_gate_gold_rows_matches_file():
    result = run_gate(warn_only=True)
    expected = count_lines(GOLD_PATH)
    assert result["gold_qrels_rows"] == expected


def test_gate_report_findings_is_list():
    result = run_gate(warn_only=True)
    assert isinstance(result["report_findings"], list)


def test_gate_blockers_list_when_gold_empty():
    """When gold qrels are empty, any NDCG/MRR claim should be flagged as BLOCKER."""
    result = run_gate(warn_only=True)
    gold_rows = result["gold_qrels_rows"]
    if gold_rows == 0:
        # Every BLOCKER must reference gold_claim_while_empty
        for b in result["blockers"]:
            assert b.get("type") == "gold_claim_while_empty", (
                f"Unexpected blocker type: {b}"
            )


def test_scan_report_stale_corpus():
    """A report referencing the old 10,404-row corpus snapshot should be flagged as stale.

    Uses a synthetic fixture (written inside ROOT, since scan_report() calls
    path.relative_to(ROOT) internally) rather than the real corpus_manifest.json:
    that file used to carry a stale 10,404-row corpus_size (fixed 2026-07-05 by
    refreezing it against the current 7,171-row full_corpus_v09.jsonl), and
    depending on production data staying wrong made this test fragile to a
    legitimate data fix.
    """
    stale_path = ROOT / "reports" / "eval" / ".tmp_test_stale_scan.json"
    stale_path.write_text('{"corpus_size": 10404, "note": "legacy snapshot"}', encoding="utf-8")
    try:
        finding = scan_report(stale_path)
    finally:
        stale_path.unlink(missing_ok=True)
    stale_issues = [i for i in finding["issues"] if i["type"] == "stale_corpus_reference"]
    assert len(stale_issues) > 0, "corpus_size: 10404 should trigger stale_corpus_reference"


def test_evidence_status_labels_present():
    result = run_gate(warn_only=True)
    labels = result.get("evidence_status_labels", {})
    assert "gold" in labels
    assert "silver_diagnostic" in labels
    assert "historical_stale" in labels
