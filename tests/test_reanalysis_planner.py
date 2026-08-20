from neural_search.reanalysis import build_reanalysis_plan

EPHYS_RECORD = {
    "dataset_id": "dandi:demo001",
    "source": "dandi",
    "source_id": "demo001",
    "title": "Neuropixels reversal learning with trial events",
    "modalities": ["neuropixels", "extracellular_ephys", "spikes"],
    "species": ["mouse"],
    "tasks": ["reversal_learning"],
    "brain_regions": ["striatum"],
    "description": "Sorted units with spike times and trial events aligned to reward outcomes.",
}


def test_reanalysis_planner_separates_feasibility_from_novelty():
    plan = build_reanalysis_plan(EPHYS_RECORD, limit=20)

    assert "extracellular_ephys" in plan.matched_data_forms
    candidate = next(item for item in plan.candidates if item.method_id == "information_theory")
    assert candidate.feasibility_status == "supported_by_metadata"
    assert candidate.missing_required_signals == []
    assert candidate.novelty_status == "possible_new_use_unverified"
    assert candidate.requires_human_review is True
    assert any("not proof of novelty" in warning for warning in plan.warnings)


def test_existing_paper_method_evidence_removes_novelty_claim():
    plan = build_reanalysis_plan(
        EPHYS_RECORD,
        existing_method_evidence={
            "information_theory": {
                "paper_openalex_id": "W123",
                "method_confidence": 0.91,
            }
        },
        limit=20,
    )

    candidate = next(item for item in plan.candidates if item.method_id == "information_theory")
    assert candidate.novelty_status == "existing_use_evidence"
    assert any(
        evidence.kind == "existing_target_paper_method"
        for evidence in candidate.evidence
    )


def test_missing_required_signals_create_explicit_blockers():
    sparse = {
        "dataset_id": "openneuro:demo",
        "source": "openneuro",
        "source_id": "demo",
        "title": "EEG recording",
        "modalities": ["eeg"],
    }
    plan = build_reanalysis_plan(sparse, limit=20)

    eeg_candidates = [item for item in plan.candidates if item.data_form == "eeg_meg"]
    assert eeg_candidates
    assert any(item.missing_required_signals for item in eeg_candidates)
    assert any(
        item.feasibility_status in {"conditional_missing_signals", "blocked_by_missing_signals"}
        for item in eeg_candidates
    )
