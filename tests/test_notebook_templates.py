import nbformat

from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.notebooks import generate_nwb_starter_notebook
from neural_search.notebooks.templates import (
    available_templates_for_dataset,
    evaluate_template_for_dataset,
    get_notebook_template,
    load_notebook_templates,
)
from neural_search.recipes import get_recipe


def _record(dataset_id: str) -> dict:
    return next(
        item
        for item in build_demo_seed()
        if item["dataset"]["source_id"] == dataset_id
    )


def test_loads_required_notebook_templates():
    template_ids = {template["id"] for template in load_notebook_templates()}

    assert {
        "generic_nwb_inspection",
        "ephys_trial_alignment",
        "calcium_trial_alignment",
        "behavior_only_bids_inspection",
        "eeg_bids_inspection",
        "reversal_learning_basic",
        "go_nogo_basic",
        "reaching_basic",
        "visual_decision_basic",
    }.issubset(template_ids)


def test_template_availability_explains_missing_requirements():
    record = _record("DEMO_REVERSAL_EPHYS")
    eeg_template = get_notebook_template("eeg_bids_inspection")
    assert eeg_template is not None

    status = evaluate_template_for_dataset(eeg_template, record["dataset"])

    assert status["available"] is False
    assert any("modality" in reason for reason in status["missing_requirements"])


def test_dataset_available_templates_include_matching_task_template():
    record = _record("DEMO_GONOGO_CALCIUM")
    statuses = available_templates_for_dataset(record["dataset"])
    by_id = {status["id"]: status for status in statuses}

    assert by_id["go_nogo_basic"]["available"] is True
    assert by_id["calcium_trial_alignment"]["available"] is True
    assert by_id["reversal_learning_basic"]["available"] is False


def test_template_generation_records_metadata_and_summary_cell(tmp_path):
    record = _record("DEMO_REVERSAL_EPHYS")
    template = get_notebook_template("reversal_learning_basic")
    recipe = get_recipe("reversal_learning_basic")
    assert template is not None
    assert recipe is not None
    status = evaluate_template_for_dataset(template, record["dataset"])
    output_path = tmp_path / "template.ipynb"

    response = generate_nwb_starter_notebook(
        record["dataset"],
        record["assets"][0],
        output_path,
        notebook_template=template,
        template_warnings=status["missing_requirements"],
        recipes=[recipe],
    )

    notebook = nbformat.read(response.output_path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert response.valid is True
    assert notebook.metadata["neural_search"]["template_id"] == "reversal_learning_basic"
    assert "inspection_summary" in source
    assert "q_learning_update" in source
