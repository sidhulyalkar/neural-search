from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from neural_search.eval.neuro_judge.evidence_packet import EvidencePacket
from neural_search.eval.neuro_judge.judge import MockNeuroJudge

FIXTURE_PATH = Path("tests/fixtures/mock_neuro_judge_adversarial.jsonl")


def _fixture_cases() -> list[dict]:
    return [
        json.loads(line)
        for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _judgments():
    judge = MockNeuroJudge()
    rows = []
    for case in _fixture_cases():
        packet = EvidencePacket.model_validate(case["packet"])
        rows.append((case, judge.judge(packet)))
    return rows


def test_canonical_fixture_suite_has_label_spread() -> None:
    labels = [judgment.label for _case, judgment in _judgments()]
    counts = Counter(labels)
    assert len(counts) >= 3
    assert max(counts.values()) / len(labels) <= 0.8


def test_canonical_expected_labels() -> None:
    for case, judgment in _judgments():
        assert judgment.label == case["expected_label"], case["case_id"]


def test_hard_negative_fixtures_cannot_receive_high_labels() -> None:
    for case, judgment in _judgments():
        if case["hard_negative"]:
            assert judgment.label < 2, case["case_id"]


def test_direct_match_fixture_receives_label_three() -> None:
    direct = [
        judgment
        for case, judgment in _judgments()
        if case["case_id"] == "direct_match_raw_ephys"
    ][0]
    assert direct.label == 3
    assert direct.evidence_completeness == 1.0
    assert direct.abstain_recommended is False


def test_raw_processed_only_fixture_abstains_with_missing_raw_data() -> None:
    processed_only = [
        judgment
        for case, judgment in _judgments()
        if case["case_id"] == "raw_ap_required_alf_processed_only"
    ][0]
    assert processed_only.label <= 2
    assert "raw_data" in processed_only.required_dimensions_missing
    assert processed_only.abstain_recommended is True
