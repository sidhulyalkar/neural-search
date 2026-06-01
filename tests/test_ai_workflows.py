import json

from neural_search.workflows import (
    audit_benchmark_report,
    load_and_audit_benchmark_report,
    run_dataset_discovery,
)


def _dataset(source_id: str, **overrides):
    dataset = {
        "id": source_id,
        "source": "demo",
        "source_id": source_id,
        "title": source_id.replace("_", " "),
        "description": "Mouse visual cortex Neuropixels recordings with choice events.",
        "species": ["mouse"],
        "modalities": ["neuropixels"],
        "brain_regions": ["visual_cortex"],
        "tasks": ["visual_decision_making"],
        "behaviors": ["choice"],
        "data_standards": ["NWB"],
        "has_behavior": True,
        "has_trials": True,
        "license": "CC-BY-4.0",
        "metadata_json": {},
    }
    dataset.update(overrides)
    return dataset


def _card(dataset_id: str):
    return {
        "dataset_id": dataset_id,
        "summary": "Reusable neural dataset.",
        "scientific_labels": {},
        "analysis_readiness": {"score": 90},
        "missing_fields": [],
        "suggested_analyses": ["event_aligned_activity"],
        "provenance": {},
    }


def test_dataset_discovery_workflow_exposes_auditable_search_fields(tmp_path):
    response = run_dataset_discovery(
        "Find mouse visual cortex recordings NOT EEG",
        datasets=[
            {
                "dataset": _dataset("GOOD"),
                "card": _card("GOOD"),
                "papers": [{"id": "P1", "title": "Visual cortex paper"}],
            },
            {
                "dataset": _dataset("BAD_EEG", modalities=["eeg"]),
                "card": _card("BAD_EEG"),
                "papers": [],
            },
        ],
        retrieval_config={
            "graph": {"enabled": True, "path": str(tmp_path / "missing_graph.json")},
            "field_embeddings": {
                "enabled": True,
                "path": str(tmp_path / "missing_embeddings.jsonl"),
            },
        },
        limit=5,
    )

    assert response.workflow == "dataset_discovery"
    assert response.total_count == 1
    assert response.filtered_constraints == [
        {"dataset_id": "BAD_EEG", "violations": ["eeg"]}
    ]

    result = response.results[0]
    assert result.dataset_id == "GOOD"
    assert result.linked_papers == [{"id": "P1", "title": "Visual cortex paper"}]
    assert "graph_score" in result.score_breakdown
    assert "field_semantic_score" in result.score_breakdown
    assert result.matched_terms


def test_benchmark_audit_workflow_classifies_actionable_failures(tmp_path):
    report = {
        "suite": "demo_v02",
        "total_queries": 2,
        "mean_precision_at_5": 0.75,
        "mean_label_recall_at_10": 0.5,
        "recommendations": ["Add synonym coverage."],
        "queries": [
            {
                "query_id": "q_good",
                "query": "good query",
                "precision_at_5": 1.0,
                "label_recall_at_10": 1.0,
                "why_failed": [],
                "hard_negative_violations": [],
                "missed_expected_datasets": [],
                "top_false_positives": [],
            },
            {
                "query_id": "q_bad",
                "query": "bad query",
                "precision_at_5": 0.2,
                "label_recall_at_10": 0.4,
                "why_failed": ["Precision below threshold"],
                "hard_negative_violations": ["BAD: hard-negative modality"],
                "missed_expected_datasets": ["EXPECTED"],
                "top_false_positives": ["BAD"],
                "missing_expected_tasks": ["go_nogo"],
                "parsed_query": {
                    "filtered_negative_constraints": [
                        {"dataset_id": "FILTERED", "violations": ["eeg"]}
                    ]
                },
            },
        ],
    }

    audit = audit_benchmark_report(report)

    assert audit.workflow == "benchmark_audit"
    assert audit.failed_query_count == 1
    assert audit.hard_negative_violation_count == 1
    assert audit.filtered_constraint_count == 1
    assert audit.aggregate_metrics["mean_precision_at_5"] == 0.75
    assert audit.recommendations == ["Add synonym coverage."]
    assert audit.issues[0].failure_types == [
        "constraints",
        "dataset_recall",
        "label_recall",
        "precision",
    ]

    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert load_and_audit_benchmark_report(path) == audit
