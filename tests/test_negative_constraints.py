from neural_search.search import parse_query, search_datasets


def _dataset(source_id: str, **overrides):
    dataset = {
        "id": source_id,
        "source": "demo",
        "source_id": source_id,
        "title": source_id,
        "description": "Mouse neural recordings with behavior.",
        "species": ["mouse"],
        "modalities": ["neuropixels", "extracellular_ephys"],
        "brain_regions": ["visual_cortex"],
        "tasks": ["visual_decision_making"],
        "behaviors": ["choice", "reward"],
        "data_standards": ["NWB"],
        "has_behavior": True,
        "has_trials": True,
        "license": "CC-BY-4.0",
        "metadata_json": {},
    }
    dataset.update(overrides)
    return dataset


def _card(dataset_id: str, **overrides):
    card = {
        "dataset_id": dataset_id,
        "summary": "Reusable neural dataset.",
        "scientific_labels": {},
        "analysis_readiness": {"score": 90},
        "missing_fields": [],
        "suggested_analyses": ["event_aligned_activity"],
        "provenance": {},
    }
    card.update(overrides)
    return card


def test_negative_constraint_parser_handles_required_phrases():
    cases = {
        "exclude fMRI": ("hard_excluded_modalities", "fmri"),
        "excluding EEG": ("hard_excluded_modalities", "eeg"),
        "without Utah array": ("hard_excluded_recording_devices", "utah_array"),
        "not pure behavior-only": ("hard_excluded_dataset_types", "behavior_only"),
        "no seizure datasets": ("hard_excluded_tasks", "seizure_monitoring"),
        "mouse but not rat": ("hard_excluded_species", "rat"),
        "human iEEG NOT fMRI": ("hard_excluded_modalities", "fmri"),
        "decision-making but not auditory": ("hard_excluded_tasks", "auditory_processing"),
        "NWB but not OpenNeuro": ("hard_excluded_sources", "openneuro"),
    }

    for query, (field, expected) in cases.items():
        parsed = parse_query(query)
        assert expected in parsed["negative_constraints"][field]


def test_search_filters_hard_negative_modality_violations():
    response = search_datasets(
        "Find visual cortex recordings from mouse NOT using EEG or fMRI.",
        datasets=[
            {"dataset": _dataset("GOOD"), "card": _card("GOOD")},
            {
                "dataset": _dataset("BAD_EEG", modalities=["eeg"]),
                "card": _card("BAD_EEG"),
            },
            {
                "dataset": _dataset("BAD_FMRI", modalities=["fmri"]),
                "card": _card("BAD_FMRI"),
            },
        ],
    )

    assert [result.dataset_id for result in response.results] == ["GOOD"]
    assert {
        item["dataset_id"] for item in response.parsed_query["filtered_negative_constraints"]
    } == {"BAD_EEG", "BAD_FMRI"}


def test_search_filters_behavior_only_source_species_and_analysis_violations():
    response = search_datasets(
        "Find NWB mouse event aligned datasets but not OpenNeuro, rat, pure behavior-only, or seizure detection.",
        datasets=[
            {"dataset": _dataset("GOOD"), "card": _card("GOOD")},
            {
                "dataset": _dataset("BAD_SOURCE", source="openneuro", data_standards=["BIDS", "OpenNeuro"]),
                "card": _card("BAD_SOURCE"),
            },
            {
                "dataset": _dataset("BAD_SPECIES", species=["rat"]),
                "card": _card("BAD_SPECIES"),
            },
            {
                "dataset": _dataset(
                    "BAD_BEHAVIOR_ONLY",
                    modalities=["behavior_video", "pose_tracking"],
                    has_behavior=True,
                ),
                "card": _card("BAD_BEHAVIOR_ONLY"),
            },
            {
                "dataset": _dataset("BAD_ANALYSIS"),
                "card": _card("BAD_ANALYSIS", suggested_analyses=["seizure_detection"]),
            },
        ],
    )

    assert [result.dataset_id for result in response.results] == ["GOOD"]
