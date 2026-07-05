"""Tests for whitespace-safe doc-id handling in TREC qrels export.

Regression guard for the 4 malformed rows where neuromorpho dataset ids
(e.g. "neuromorpho:Physio Lab - Medical Faculty - UoI") contained spaces and
broke the positional TREC format.
"""

from scripts.eval.build_canonical_qrels import canonicalize
from scripts.eval.docid import normalize_docid


def test_normalize_docid_collapses_whitespace():
    assert normalize_docid("neuromorpho:Physio Lab - Medical Faculty - UoI") == (
        "neuromorpho:Physio_Lab_-_Medical_Faculty_-_UoI"
    )
    assert normalize_docid("neuromorpho:Munoz et al.") == "neuromorpho:Munoz_et_al."


def test_normalize_docid_noop_and_idempotent():
    assert normalize_docid("dandi:001051") == "dandi:001051"
    once = normalize_docid("neuromorpho:Allen Cell Types")
    assert normalize_docid(once) == once  # idempotent


def test_canonicalize_trec_rows_are_four_columns():
    judgments = [
        {"query_id": "can_0201", "dataset_id": "neuromorpho:Physio Lab - Medical Faculty - UoI",
         "label": 0, "rationale_short": "ok"},
        {"query_id": "can_0001", "dataset_id": "dandi:001051", "label": 1, "rationale_short": "ok"},
    ]
    trec_lines, jsonl_rows = canonicalize(judgments)
    # Every TREC line must split into exactly 4 whitespace-delimited tokens.
    for line in trec_lines:
        assert len(line.split()) == 4, line
    # TREC export uses the normalized id ...
    assert trec_lines[0] == "can_0201 0 neuromorpho:Physio_Lab_-_Medical_Faculty_-_UoI 0"
    # ... while canonical JSONL preserves the original id for provenance.
    assert jsonl_rows[0]["dataset_id"] == "neuromorpho:Physio Lab - Medical Faculty - UoI"
