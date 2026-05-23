from neural_search.cards import generate_dataset_card_json
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.qa.state import (
    attach_qa_to_card,
    attach_qa_to_dataset,
    load_qa_state,
    reviewed_dataset_cards,
    update_dataset_status,
)


def test_qa_status_updates_round_trip(tmp_path):
    state_path = tmp_path / "dataset_qa.json"

    record = update_dataset_status(
        "DEMO_GONOGO_CALCIUM",
        "trusted",
        notes="Ready for demo.",
        path=state_path,
    )

    assert record["qa_status"] == "trusted"
    assert record["reviewer_notes"] == "Ready for demo."
    assert load_qa_state(state_path)["DEMO_GONOGO_CALCIUM"]["qa_status"] == "trusted"


def test_qa_fields_attach_to_dataset_and_card():
    record = build_demo_seed()[0]
    dataset = attach_qa_to_dataset(
        record["dataset"],
        {
            record["dataset"]["source_id"]: {
                "qa_status": "reviewed",
                "task_labels_verified": True,
            }
        },
    )
    card = generate_dataset_card_json(dataset, record["extraction"], record["papers"])
    attach_qa_to_card(card, dataset, {dataset["source_id"]: dataset})

    assert dataset["qa_status"] == "reviewed"
    assert card.qa_status == "reviewed"
    assert card.task_labels_verified is True


def test_reviewed_card_export_uses_persisted_state(tmp_path, monkeypatch):
    state_path = tmp_path / "dataset_qa.json"
    update_dataset_status("DEMO_GONOGO_CALCIUM", "reviewed", path=state_path)
    monkeypatch.setattr(
        "neural_search.qa.state.load_qa_state",
        lambda: load_qa_state(state_path),
    )

    cards = reviewed_dataset_cards()

    assert [item["dataset"]["source_id"] for item in cards] == ["DEMO_GONOGO_CALCIUM"]
    assert cards[0]["card"]["qa_status"] == "reviewed"
