from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.schemas import SearchRequest
from neural_search.search import search_datasets
from neural_search.search.query_builder import (
    combine_query_and_structured_text,
    structured_query_to_filters,
)


def test_search_request_accepts_structured_experiment_query():
    request = SearchRequest.model_validate(
        {
            "query": "reward omission",
            "structured_query": {
                "task": ["reversal_learning"],
                "modality": ["extracellular_ephys"],
                "min_analysis_readiness_score": 90,
                "reviewed_trusted_only": True,
            },
        }
    )

    assert request.structured_query is not None
    assert request.structured_query.task == ["reversal_learning"]
    assert request.structured_query.reviewed_trusted_only is True


def test_structured_query_compiles_to_filters_and_query_text():
    structured = {
        "task": ["go_nogo"],
        "behavior": ["lick"],
        "analysis_goal": ["peri_event_time_histogram"],
        "min_analysis_readiness_score": 80,
        "reviewed_trusted_only": True,
    }

    assert structured_query_to_filters(structured) == {
        "tasks": ["go_nogo"],
        "behaviors": ["lick"],
        "min_analysis_readiness_score": 80,
        "qa_status": ["reviewed", "trusted"],
    }
    combined = combine_query_and_structured_text("response inhibition", structured)
    assert "response inhibition" in combined
    assert "task: go_nogo" in combined
    assert "analysis goal: peri_event_time_histogram" in combined


def test_structured_filters_search_same_retrieval_api():
    response = search_datasets(
        "",
        structured_query={
            "task": ["visual_decision_making"],
            "modality": ["neuropixels"],
            "species": ["mouse"],
        },
    )

    assert response.results
    assert response.results[0].dataset_id == "DEMO_VISUAL_DECISION_NEUROPIXELS"


def test_min_readiness_and_reviewed_trusted_filters():
    records = build_demo_seed()
    for record in records:
        record["dataset"]["qa_status"] = "auto_generated"
    records[0]["dataset"]["qa_status"] = "trusted"

    response = search_datasets(
        "go nogo",
        structured_query={"reviewed_trusted_only": True},
        datasets=records,
    )

    assert [result.dataset_id for result in response.results] == ["DEMO_GONOGO_CALCIUM"]

    too_high = search_datasets(
        "go nogo",
        structured_query={"min_analysis_readiness_score": 96},
        datasets=records,
    )
    assert too_high.results == []
