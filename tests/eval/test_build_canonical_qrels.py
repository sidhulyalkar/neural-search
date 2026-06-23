from scripts.eval.build_canonical_qrels import canonicalize


def test_drops_errors_dedups_and_emits_both_forms():
    judgments = [
        {"query_id": "q_1", "dataset_id": "dandi:1", "label": 3, "rationale_short": "great"},
        {"query_id": "q_1", "dataset_id": "dandi:2", "label": 0,
         "rationale_short": "judge_error: all_retries_failed"},   # dropped
        {"query_id": "q_1", "dataset_id": "dandi:1", "label": 2,  # dup -> first kept
         "rationale_short": "ok"},
    ]
    trec_lines, jsonl_rows = canonicalize(judgments)
    assert trec_lines == ["q_1 0 dandi:1 3"]
    assert jsonl_rows == [{"query_id": "q_1", "dataset_id": "dandi:1", "label": 3}]


def test_drops_negative_and_missing_labels():
    from scripts.eval.build_canonical_qrels import canonicalize
    judgments = [
        {"query_id": "q_1", "dataset_id": "d:1", "label": -1, "rationale_short": "x"},
        {"query_id": "q_1", "dataset_id": "d:2", "rationale_short": "no label key"},
        {"query_id": "q_1", "dataset_id": "d:3", "label": 1, "rationale_short": "ok"},
    ]
    trec, rows = canonicalize(judgments)
    assert trec == ["q_1 0 d:3 1"]
    assert rows == [{"query_id": "q_1", "dataset_id": "d:3", "label": 1}]
