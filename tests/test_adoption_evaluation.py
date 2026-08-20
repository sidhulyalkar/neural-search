from pathlib import Path

from neural_search.evaluation.adoption import evaluate_adoption_file


FIXTURE = Path("data/eval/adoption_events_demo.jsonl")


def test_adoption_fixture_reports_workflow_metrics():
    report = evaluate_adoption_file(FIXTURE)

    assert report["sessions"] == 3
    assert report["metrics"]["search_success_rate"] == 0.75
    assert report["metrics"]["useful_dataset_discovery_rate"] == 0.666667
    assert report["metrics"]["notebook_execution_success_rate"] == 0.5
    assert report["metrics"]["metadata_failure_recovery_rate"] == 1.0
    assert report["metrics"]["positive_reuse_decision_rate"] == 0.666667
    assert report["metrics"]["novel_useful_discovery_rate"] == 0.5
    assert report["metrics"]["workflow_completion_rate"] == 0.666667
    assert report["metrics"]["time_to_first_successful_search_seconds"]["median"] == 20.0
    assert report["interpretation"]["gold_relevance_claim"] is False
