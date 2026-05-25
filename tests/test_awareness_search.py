from __future__ import annotations

from neural_search.awareness.search import search_datasets_with_awareness


def _record(
    source_id: str,
    title: str,
    description: str,
    *,
    species: list[str],
    modalities: list[str],
    behaviors: list[str] | None = None,
    tasks: list[str] | None = None,
) -> dict:
    return {
        "dataset": {
            "id": source_id,
            "source": "fixture",
            "source_id": source_id,
            "title": title,
            "description": description,
            "url": f"https://example.org/{source_id}",
            "species": species,
            "modalities": modalities,
            "brain_regions": [],
            "tasks": tasks or [],
            "behaviors": behaviors or [],
            "data_standards": ["BIDS"],
            "has_behavior": bool(behaviors),
            "has_trials": True,
            "license": "CC0",
            "linked_paper_ids": [],
            "metadata_json": {},
        }
    }


DATASETS = [
    _record(
        "GOOD_EEG",
        "Human BCI motor imagery EEG",
        "Human EEG channels, sampling rate, events, labels, and behavior trials for BCI decoding.",
        species=["human"],
        modalities=["eeg"],
        behaviors=["motor imagery", "button press"],
        tasks=["bci_decoding"],
    ),
    _record(
        "BAD_FMRI",
        "Human fMRI behavior task",
        "Human BOLD fMRI images, participants, events, and behavior labels.",
        species=["human"],
        modalities=["fmri"],
        behaviors=["button press"],
        tasks=["decision making"],
    ),
    _record(
        "MOUSE_EPHYS",
        "Mouse Neuropixels decision behavior",
        "Mouse Neuropixels units, spike times, events, and behavior tracking.",
        species=["mouse"],
        modalities=["neuropixels"],
        behaviors=["choice"],
        tasks=["decision making"],
    ),
]


def test_awareness_search_adds_query_and_result_annotations() -> None:
    response = search_datasets_with_awareness(
        "human EEG BCI decoding with behavior without fMRI",
        datasets=DATASETS,
        limit=3,
    )

    assert response.parsed_query["query_awareness"]["requested_data_forms"]
    assert "mri" in response.parsed_query["query_awareness"]["excluded_data_forms"]
    assert response.results
    result = response.results[0]
    assert "awareness_score" in result.score_breakdown
    assert "data_form_awareness" in result.dataset_card_preview


def test_awareness_rerank_prefers_data_form_fit() -> None:
    response = search_datasets_with_awareness(
        "human EEG BCI decoding with behavior without fMRI",
        datasets=DATASETS,
        limit=3,
        awareness_weight=0.6,
        rerank=True,
    )

    assert response.results[0].dataset_id == "GOOD_EEG"
    assert response.results[0].score_breakdown["awareness_rerank_weight"] == 0.6
    assert response.results[0].score_breakdown["awareness_score"] > 0.6


def test_awareness_warnings_surface_when_hard_filtering_is_disabled() -> None:
    response = search_datasets_with_awareness(
        "human behavior without fMRI",
        datasets=DATASETS,
        limit=3,
        retrieval_config={"hard_negative_filters": {"enabled": False}},
    )
    fmri_result = next(result for result in response.results if result.dataset_id == "BAD_FMRI")

    assert any("excluded data forms" in warning for warning in fmri_result.warnings)
    assert fmri_result.score_breakdown["awareness_score"] < 0.5
