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
