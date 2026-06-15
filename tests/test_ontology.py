from neural_search.ontology import (
    get_task_by_id,
    load_ontology,
    load_recording_scales,
    match_all,
    match_behavior_labels,
    match_brain_regions,
    match_modalities,
    match_recording_scales,
    match_tasks,
    validate_ontology,
)


def test_load_ontology_has_expected_tasks():
    ontology = load_ontology("data/ontology/behavioral_task_ontology.yaml")

    assert len(ontology.tasks) >= 50
    assert get_task_by_id("reversal_learning").label == "Reversal Learning"


def test_validate_ontology_schema():
    ontology = validate_ontology("data/ontology/behavioral_task_ontology.yaml")

    for task in ontology.tasks:
        assert task.id
        assert task.label
        assert task.category
        assert task.definition
        assert isinstance(task.synonyms, list)
        assert isinstance(task.common_events, list)
        assert isinstance(task.relevant_modalities, list)
        assert isinstance(task.relevant_regions, list)
        assert isinstance(task.suggested_analyses, list)


def test_load_recording_scales_schema():
    scales = load_recording_scales("data/ontology/recording_scales.yaml")

    assert len(scales) >= 10
    for scale in scales:
        assert scale.id
        assert scale.label
        assert scale.compatible_modalities
        assert scale.aliases


def test_synonym_matching_returns_evidence_and_confidence():
    matches = match_tasks("probabilistic reversal learning with reward omission")

    reversal = next(match for match in matches if match.id == "reversal_learning")
    assert reversal.confidence >= 0.9
    assert reversal.evidence == "probabilistic reversal learning"


def test_behavior_label_matching():
    matches = match_behavior_labels("lickometer traces and reward omission events")
    ids = {match.id for match in matches}

    assert {"lick", "reward", "omission"} <= ids


def test_fuzzy_task_synonym_matching():
    matches = match_tasks("probabilistic reversl learning with reward omission")

    assert matches[0].id == "reversal_learning"
    assert matches[0].match_type == "fuzzy"
    assert matches[0].confidence >= 0.65


def test_fuzzy_region_and_modality_matching():
    region_ids = {match.id for match in match_brain_regions("OFC recordings in mouse")}
    modality_ids = {
        match.id
        for match in match_modalities("neuropixel recording for BCI cursor control")
    }

    assert "OFC" in region_ids
    assert "neuropixels" in modality_ids
    assert "bci" in modality_ids


def test_recording_scale_matching_distinguishes_lfp_and_single_unit():
    lfp_ids = {match.id for match in match_recording_scales("hippocampus LFP theta oscillations")}
    unit_ids = {
        match.id for match in match_recording_scales("PFC single-unit spike times")
    }

    assert "local_field_potential" in lfp_ids
    assert "single_unit_spikes" in unit_ids
    assert "single_unit_spikes" not in lfp_ids


def test_match_all_returns_requested_groups():
    matches = match_all(
        "Find reversal learning datasets with OFC recordings and reward omission"
    )

    assert "reversal_learning" in {match.id for match in matches["tasks"]}
    assert {"reward", "omission"} <= {match.id for match in matches["behaviors"]}
    assert "OFC" in {match.id for match in matches["regions"]}
    assert set(matches) == {
        "tasks",
        "behaviors",
        "regions",
        "modalities",
        "recording_scales",
        "affordances",
    }
