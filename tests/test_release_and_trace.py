from __future__ import annotations

import json

import neural_search.release.check as release_check
from neural_search.contracts import SearchResponseV1, SearchResultV1
from neural_search.release.check import (
    build_release_summary,
    summarize_graph_quality_artifact,
    write_release_summary,
)
from neural_search.schemas import SearchResponse, SearchResult
from neural_search.search.trace import capture_search_trace, write_search_trace


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


def test_search_trace_can_be_exported(tmp_path) -> None:
    trace = capture_search_trace(
        "mouse visual decision making",
        datasets=[
            {
                "dataset": {
                    "id": "TRACE_DEMO",
                    "source_id": "TRACE_DEMO",
                    "title": "Mouse visual decision task",
                    "species": ["mouse"],
                    "modalities": ["neuropixels"],
                    "tasks": ["visual_decision_making"],
                }
            }
        ],
        limit=1,
    )

    output = write_search_trace(trace, tmp_path / "trace.json")

    assert output.exists()
    assert '"TRACE_DEMO"' in output.read_text(encoding="utf-8")


def test_release_summary_records_artifacts_and_can_be_written(tmp_path) -> None:
    paths = write_release_summary(tmp_path)
    summary = build_release_summary()

    assert paths["json"].endswith("release_summary.json")
    assert paths["markdown"].endswith("release_summary.md")
    assert summary["artifact_versions"]["embedding_provider"] == "hashing"
    assert "real_datasets" in summary["artifacts"]
    assert "staleness" in summary["artifacts"]["real_datasets"]
    assert "demo_v02" in summary["benchmarks"]
    assert "graph_quality" in summary
    assert "source_quality" in summary
    assert "release_warnings" in summary


def test_release_summary_includes_non_failing_source_quality_warnings(tmp_path) -> None:
    readiness_path = tmp_path / "scientific_readiness_report.json"
    readiness_path.write_text(
        json.dumps(
            {
                "source_quality": {
                    "mean_quality_score": 0.62,
                    "trust_level_counts": {"unknown": 1, "low": 1},
                    "warning_count": 2,
                }
            }
        ),
        encoding="utf-8",
    )

    summary = build_release_summary(readiness_report=readiness_path)
    markdown = write_release_summary(
        tmp_path / "release",
        readiness_report=readiness_path,
    )

    assert summary["source_quality"]["available"] is True
    assert "mean source quality is below 0.70" in summary["release_warnings"]
    assert not any("source quality" in failure for failure in summary["known_failures"])
    assert (tmp_path / "release" / "release_summary.md").exists()
    assert markdown["markdown"].endswith("release_summary.md")


def test_release_summary_includes_non_failing_graph_quality_warnings(
    tmp_path,
    monkeypatch,
) -> None:
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(
        json.dumps(
            {
                "nodes": {
                    "node:dataset:D1": {
                        "node_id": "node:dataset:D1",
                        "node_type": "dataset",
                        "label": "D1",
                        "confidence": 1.2,
                    }
                },
                "edges": {
                    "edge:dataset:D1:has_task:T1": {
                        "edge_id": "edge:dataset:D1:has_task:T1",
                        "source_node_id": "node:dataset:D1",
                        "target_node_id": "node:task:T1",
                        "edge_type": "dataset_has_task",
                        "confidence": 0.2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    quality = summarize_graph_quality_artifact(graph_path)
    monkeypatch.setattr(release_check, "ARTIFACTS", {"demo_graph": graph_path})
    monkeypatch.setattr(release_check, "ARTIFACT_INPUTS", {"demo_graph": []})
    monkeypatch.setattr(release_check, "BENCHMARK_REPORTS", {})
    summary = release_check.build_release_summary(readiness_report=None)

    assert quality["available"] is True
    assert quality["passed"] is False
    assert quality["error_count"] == 2
    assert quality["issue_counts"] == {
        "dangling_edge_reference": 1,
        "invalid_node_confidence": 1,
        "weak_edge": 1,
    }
    assert summary["release_ready"] is True
    assert summary["graph_quality"]["demo_graph"]["error_count"] == 2
    assert "demo_graph graph QA has 2 error(s)" in summary["release_warnings"]
