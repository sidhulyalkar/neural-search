from neural_search.evaluation.run_benchmark import BenchmarkQuery, evaluate_query


def _record(source_id, **overrides):
    dataset = {
        "id": source_id,
        "source": "demo",
        "source_id": source_id,
        "title": source_id,
        "description": "Mouse OFC reversal learning electrophysiology with reward omission.",
        "species": ["mouse"],
        "modalities": ["extracellular_ephys"],
        "brain_regions": ["OFC"],
        "tasks": ["reversal_learning"],
        "behaviors": ["choice", "reward", "omission"],
        "data_standards": ["NWB"],
        "license": "CC-BY-4.0",
        "has_behavior": True,
        "has_trials": True,
        "linked_paper_ids": [],
        "metadata_json": {},
    }
    dataset.update(overrides)
    card = {
        "dataset_id": source_id,
        "summary": dataset["description"],
        "scientific_labels": {
            "tasks": [{"id": value, "label": value} for value in dataset["tasks"]],
            "behaviors": [{"id": value, "label": value} for value in dataset["behaviors"]],
            "modalities": [{"id": value, "label": value} for value in dataset["modalities"]],
            "brain_regions": [{"id": value, "label": value} for value in dataset["brain_regions"]],
            "species": [{"id": value, "label": value} for value in dataset["species"]],
        },
        "analysis_readiness": {"score": 90},
        "missing_fields": [],
        "suggested_analyses": ["event_aligned_activity"],
        "provenance": {"linked_paper_count": 0},
        "why_relevant": [],
    }
    return {"dataset": dataset, "card": card}


def test_evaluate_query_reports_expected_id_metrics():
    query = BenchmarkQuery(
        id="expected_id",
        query="reversal learning OFC electrophysiology reward omission",
        expected_dataset_ids=["REVERSAL"],
        expected_tasks=["reversal_learning"],
        expected_modalities_any=["extracellular_ephys"],
        minimum_precision_at_5=0.5,
        minimum_label_recall_at_10=0.5,
    )

    evaluation = evaluate_query(
        query,
        datasets=[
            _record("VISUAL", tasks=["visual_decision_making"], brain_regions=["visual_cortex"]),
            _record("REVERSAL"),
        ],
    )

    assert evaluation.precision_at_1 == 1.0
    assert evaluation.recall_at_10 == 1.0
    assert evaluation.missed_expected_datasets == []
    assert evaluation.why_failed == []


def test_evaluate_query_flags_hard_negative_modalities():
    query = BenchmarkQuery(
        id="hard_negative",
        query="visual cortex mouse recordings not EEG",
        expected_species=["mouse"],
        expected_regions_any=["visual_cortex"],
        hard_negative_modalities=["eeg"],
    )

    evaluation = evaluate_query(
        query,
        datasets=[
            _record(
                "EEG_NEGATIVE",
                modalities=["eeg"],
                brain_regions=["visual_cortex"],
                tasks=["visual_decision_making"],
            )
        ],
    )

    assert evaluation.hard_negative_violations
    assert evaluation.top_false_positives == ["EEG_NEGATIVE"]
    assert any("Hard-negative violations" in reason for reason in evaluation.why_failed)
