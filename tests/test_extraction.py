from neural_search.extraction import extract_dataset_labels
from neural_search.ingestion.demo_seed import build_demo_seed


def test_behavior_label_extraction_from_fixture_text():
    record = next(
        item
        for item in build_demo_seed()
        if item["dataset"]["source_id"] == "DEMO_GONOGO_CALCIUM"
    )

    behavior_ids = {label.id for label in record["extraction"].behaviors}

    assert {"lick", "reward", "omission"} <= behavior_ids


def test_modality_extraction_from_fixture_text():
    record = next(
        item
        for item in build_demo_seed()
        if item["dataset"]["source_id"] == "DEMO_REACHING_ECOG_IEEG"
    )

    modality_ids = {label.id for label in record["extraction"].modalities}

    assert {"ecog", "ieeg", "pose_tracking"} <= modality_ids


def test_visual_decision_fixture_extracts_task_and_modality():
    extraction = extract_dataset_labels(
        title="Visual decision-making Neuropixels dataset",
        description="Mouse visual discrimination task with choice and reward.",
        file_paths=["sub-01/session.nwb"],
        source_metadata={"license": "CC-BY-4.0"},
        linked_paper_abstracts=[],
    )

    task_ids = {label.id for label in extraction.tasks}
    modality_ids = {label.id for label in extraction.modalities}

    assert "visual_decision_making" in task_ids
    assert "neuropixels" in modality_ids
