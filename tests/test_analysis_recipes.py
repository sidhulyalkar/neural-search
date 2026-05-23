import nbformat

from neural_search.cards import generate_dataset_card_json
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.notebooks import generate_nwb_starter_notebook
from neural_search.recipes import get_recipe, match_recipes_for_tasks


def test_recipe_matching_for_reversal_learning():
    matches = match_recipes_for_tasks(["reversal_learning"])

    assert matches
    assert matches[0]["id"] == "reversal_learning_basic"
    assert "q_learning_modeling" in matches[0]["analyses"]


def test_dataset_card_selects_starter_recipes():
    record = next(
        item
        for item in build_demo_seed()
        if item["dataset"]["source_id"] == "DEMO_REACHING_ECOG_IEEG"
    )

    card = generate_dataset_card_json(
        record["dataset"],
        record["extraction"],
        record["papers"],
        record["assets"],
    )

    recipe_ids = [recipe["id"] for recipe in card.analysis_plan["starter_recipes"]]
    assert "reaching_basic" in recipe_ids


def test_notebook_includes_selected_recipe_cells(tmp_path):
    record = next(
        item
        for item in build_demo_seed()
        if item["dataset"]["source_id"] == "DEMO_REVERSAL_EPHYS"
    )
    recipe = get_recipe("reversal_learning_basic")
    assert recipe is not None
    output_path = tmp_path / "recipe.ipynb"

    response = generate_nwb_starter_notebook(
        record["dataset"],
        record["assets"][0],
        output_path,
        recipes=[recipe],
    )

    notebook = nbformat.read(response.output_path, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    assert response.valid is True
    assert "Analysis Recipe: Reversal learning starter analyses" in source
    assert "q_learning_update" in source
    assert notebook.metadata["neural_search"]["recipe_ids"] == ["reversal_learning_basic"]
