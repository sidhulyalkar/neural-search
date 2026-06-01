from neural_search.schemas import SearchResult


def test_search_result_has_usefulness_score_field():
    """SearchResult must have usefulness_score as optional dict."""
    result = SearchResult(dataset_id="ds_test", score=0.5)
    assert hasattr(result, "usefulness_score")
    assert result.usefulness_score is None


def test_search_result_usefulness_score_accepts_dict():
    result = SearchResult(
        dataset_id="ds_test",
        score=0.5,
        usefulness_score={"total_score": 0.75, "intent": "replication"},
    )
    assert result.usefulness_score["total_score"] == 0.75
