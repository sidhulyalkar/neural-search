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


from neural_search.search import search_datasets


def test_search_result_carries_usefulness_score():
    """search_datasets results must include usefulness_score dict."""
    response = search_datasets("mouse decision-making neuropixels")
    assert len(response.results) > 0
    first = response.results[0]
    assert first.usefulness_score is not None
    assert "total_score" in first.usefulness_score
    assert 0.0 <= first.usefulness_score["total_score"] <= 1.0


def test_usefulness_score_contains_intent():
    """usefulness_score must include the classified intent name."""
    response = search_datasets("replicate Steinmetz 2019 experiment")
    assert len(response.results) > 0
    first = response.results[0]
    assert first.usefulness_score is not None
    assert "intent" in first.usefulness_score


def test_usefulness_score_contains_dimension_scores():
    """usefulness_score must have a dimension_scores dict."""
    response = search_datasets("mouse hippocampus calcium imaging")
    assert len(response.results) > 0
    first = response.results[0]
    assert first.usefulness_score is not None
    assert "dimension_scores" in first.usefulness_score
    dims = first.usefulness_score["dimension_scores"]
    assert isinstance(dims, dict)
    assert len(dims) > 0
