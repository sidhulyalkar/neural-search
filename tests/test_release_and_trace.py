from __future__ import annotations

from neural_search.contracts import SearchResponseV1, SearchResultV1
from neural_search.release.check import build_release_summary, write_release_summary
from neural_search.schemas import SearchResponse, SearchResult
from neural_search.search.trace import capture_search_trace


def test_versioned_contracts_and_legacy_search_schemas_are_compatible() -> None:
    legacy_result = SearchResult(dataset_id="dataset:test:1", score=1.0)
    legacy_response = SearchResponse(query="demo", results=[legacy_result])
    contract = SearchResponseV1(
        query=legacy_response.query,
        results=[
            SearchResultV1(
                dataset_id=str(legacy_result.dataset_id),
                score=legacy_result.score,
                score_breakdown=legacy_result.score_breakdown,
                graph_context=None,
                filtered_constraints=[],
                missing_metadata=[],
            )
        ],
    )

    assert legacy_response.filtered_constraints == []
    assert legacy_result.linked_papers == []
    assert contract.schema_version == "v1"
    assert contract.results[0].score_breakdown == {}


def test_search_trace_captures_scores_constraints_and_timings() -> None:
    datasets = [
        {
            "dataset": {
                "id": "GOOD",
                "source_id": "GOOD",
                "title": "Mouse Neuropixels visual task",
                "description": "Trials with spikes and visual choices",
                "species": ["mouse"],
                "modalities": ["neuropixels"],
                "brain_regions": ["visual_cortex"],
                "tasks": ["visual_decision_making"],
                "behaviors": ["choice"],
                "data_standards": ["NWB"],
                "has_trials": True,
                "has_behavior": True,
            },
            "card": {
                "summary": "Ready for visual decision analysis.",
                "analysis_readiness": {"score": 90},
                "suggested_analyses": ["event_aligned_analysis"],
                "missing_fields": [],
            },
        },
        {
            "dataset": {
                "id": "BAD_EEG",
                "source_id": "BAD_EEG",
                "title": "Human EEG task",
                "species": ["human"],
                "modalities": ["eeg"],
                "tasks": ["motor_imagery"],
            }
        },
    ]

    trace = capture_search_trace(
        "mouse visual decision making without EEG",
        datasets=datasets,
        retrieval_config={"hard_negative_filters": {"enabled": True}},
    )

    assert trace.results
    assert trace.results[0].dataset_id == "GOOD"
    assert trace.filtered_constraints
    assert trace.timings_ms["search"] >= 0
    assert "negative_constraints" in trace.parsed_query


def test_release_summary_records_artifacts_and_can_be_written(tmp_path) -> None:
    paths = write_release_summary(tmp_path)
    summary = build_release_summary()

    assert paths["json"].endswith("release_summary.json")
    assert paths["markdown"].endswith("release_summary.md")
    assert summary["artifact_versions"]["embedding_provider"] == "hashing"
    assert "real_datasets" in summary["artifacts"]
    assert "demo_v02" in summary["benchmarks"]
