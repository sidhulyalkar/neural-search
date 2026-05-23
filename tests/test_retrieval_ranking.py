from neural_search.search import parse_query, score_dataset_against_query, search_datasets


def _dataset(source_id: str, **overrides):
    dataset = {
        "id": source_id,
        "source": "test",
        "source_id": source_id,
        "title": source_id.replace("_", " "),
        "description": "Mouse OFC trial dataset with choice and reward events.",
        "url": "https://example.org",
        "species": ["mouse"],
        "modalities": ["neuropixels"],
        "brain_regions": ["OFC"],
        "tasks": ["reversal_learning"],
        "behaviors": ["choice", "reward"],
        "data_standards": ["NWB"],
        "has_behavior": True,
        "has_trials": True,
        "license": "CC-BY-4.0",
        "linked_paper_ids": [],
        "metadata_json": {},
    }
    dataset.update(overrides)
    return dataset


def _card(dataset_id: str, **overrides):
    card = {
        "dataset_id": dataset_id,
        "summary": "Trial-aligned neural data with behavior labels.",
        "scientific_labels": {
            "tasks": [{"id": "reversal_learning", "label": "Reversal Learning"}],
            "behaviors": [{"id": "choice", "label": "Choice"}],
            "modalities": [{"id": "neuropixels", "label": "Neuropixels"}],
            "brain_regions": [{"id": "OFC", "label": "OFC"}],
            "species": [{"id": "mouse", "label": "Mouse"}],
        },
        "analysis_readiness": {"score": 95},
        "missing_fields": [],
        "suggested_analyses": ["choice_decoding", "event_aligned_activity"],
        "provenance": {"linked_paper_count": 0},
        "why_relevant": ["Choice matched from evidence 'choice'."],
    }
    card.update(overrides)
    return card


def test_ranking_prefers_exact_behavior_modality_and_analysis_ready_dataset():
    relevant = _dataset("RELEVANT")
    mismatch = _dataset(
        "MISMATCH",
        modalities=["calcium_imaging"],
        brain_regions=["visual_cortex"],
        description="Mouse visual calcium imaging without OFC Neuropixels.",
    )
    mismatch_card = _card(
        "MISMATCH",
        scientific_labels={
            "tasks": [{"id": "reversal_learning", "label": "Reversal Learning"}],
            "behaviors": [{"id": "choice", "label": "Choice"}],
            "modalities": [{"id": "calcium_imaging", "label": "Calcium imaging"}],
            "brain_regions": [{"id": "visual_cortex", "label": "Visual cortex"}],
            "species": [{"id": "mouse", "label": "Mouse"}],
        },
        analysis_readiness={"score": 75},
    )

    response = search_datasets(
        "mouse OFC Neuropixels data for choice decoding",
        datasets=[
            {"dataset": mismatch, "card": mismatch_card},
            {"dataset": relevant, "card": _card("RELEVANT")},
        ],
    )

    assert response.results[0].dataset_id == "RELEVANT"
    assert "choice_decoding" in response.parsed_query["analysis"]
    assert response.results[0].reusable_reason

    mismatch_result = next(result for result in response.results if result.dataset_id == "MISMATCH")
    assert any("Modality mismatch" in warning for warning in mismatch_result.warnings)


def test_linked_papers_boost_confidence_without_dominating_relevance():
    relevant = _dataset("RELEVANT", linked_paper_ids=[])
    unrelated_with_papers = _dataset(
        "PAPER_RICH",
        title="Paper rich unrelated dataset",
        description="Human ECoG recordings during auditory monitoring.",
        species=["human"],
        modalities=["ecog"],
        brain_regions=["auditory_cortex"],
        tasks=["auditory_oddball"],
        behaviors=["reaction_time"],
        linked_paper_ids=["P1", "P2", "P3"],
    )
    unrelated_card = _card(
        "PAPER_RICH",
        scientific_labels={
            "tasks": [{"id": "auditory_oddball", "label": "Auditory oddball"}],
            "behaviors": [{"id": "reaction_time", "label": "Reaction time"}],
            "modalities": [{"id": "ecog", "label": "ECoG"}],
            "brain_regions": [{"id": "auditory_cortex", "label": "Auditory cortex"}],
            "species": [{"id": "human", "label": "Human"}],
        },
        suggested_analyses=[],
        provenance={"linked_paper_count": 3},
    )

    parsed = parse_query("Find mouse OFC Neuropixels datasets to decode choice")
    relevant_score = score_dataset_against_query(
        relevant,
        _card("RELEVANT"),
        parsed,
    )
    unrelated_score = score_dataset_against_query(
        unrelated_with_papers,
        unrelated_card,
        parsed,
    )

    assert relevant_score.score > unrelated_score.score
    assert any("Linked papers" in reason for reason in unrelated_score.why_matched)
