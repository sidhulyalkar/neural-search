from neural_search.ontology import (
    get_task_by_id,
    load_ontology,
    match_behavior_labels,
    match_tasks,
)


def test_load_ontology_has_expected_tasks():
    ontology = load_ontology("data/ontology/behavioral_task_ontology.yaml")

    assert len(ontology.tasks) >= 50
    assert get_task_by_id("reversal_learning").label == "Reversal Learning"


def test_synonym_matching_returns_evidence_and_confidence():
    matches = match_tasks("probabilistic reversal learning with reward omission")

    reversal = next(match for match in matches if match.id == "reversal_learning")
    assert reversal.confidence >= 0.9
    assert reversal.evidence == "probabilistic reversal learning"


def test_behavior_label_matching():
    matches = match_behavior_labels("lickometer traces and reward omission events")
    ids = {match.id for match in matches}

    assert {"lick", "reward", "omission"} <= ids

