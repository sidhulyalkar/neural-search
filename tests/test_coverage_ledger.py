from neural_search.coverage import (
    CoverageLedger,
    build_completion_worklist,
    build_coverage_entries,
    build_coverage_state_entries,
    build_gap_report,
    modality_compatibility,
    propagate_confidence,
)


def _dataset(
    dataset_id: str,
    *,
    source: str = "dandi",
    species: list[str] | None = None,
    brain_regions: list[str] | None = None,
    modalities: list[str] | None = None,
    recording_scales: list[str] | None = None,
    tasks: list[str] | None = None,
    behaviors: list[str] | None = None,
    provenance: str | None = None,
) -> dict:
    return {
        "dataset_id": dataset_id,
        "source": source,
        "source_id": dataset_id.rsplit(":", maxsplit=1)[-1],
        "title": dataset_id,
        "species": species or [],
        "brain_regions": brain_regions or [],
        "modalities": modalities or [],
        "recording_scales": recording_scales or [],
        "tasks": tasks or [],
        "behavioral_events": behaviors or [],
        "brain_regions_provenance": provenance,
        "usability_flags": {
            "has_raw_data": True,
            "has_behavior": bool(tasks or behaviors),
            "has_standard_format": True,
        },
    }


def test_coverage_entries_include_scale_access_and_analysis_level():
    records = [
        _dataset(
            "dataset:dandi:000640",
            species=["rat"],
            brain_regions=["mPFC"],
            modalities=["lfp"],
            recording_scales=["local_field_potential"],
            tasks=["reversal_learning"],
        )
    ]

    entries = build_coverage_entries(records, snapshot_id="test")
    dimensions = {entry.dimension for entry in entries}

    assert "recording_scales" in dimensions
    assert "access_tiers" in dimensions
    assert "analysis_levels" in dimensions
    assert any(entry.value_id == "mesoscale_field_potential" for entry in entries)
    assert any(entry.access_tier == "open_access" for entry in entries)


def test_coverage_entries_accept_dataset_like_rows_without_dataset_id():
    records = [
        {
            "source": "neuromorpho",
            "source_id": "Renner",
            "title": "NeuroMorpho: Renner",
            "species": ["human"],
            "modalities": ["neuron_morphology"],
        }
    ]

    entries = build_coverage_entries(records, snapshot_id="test")

    assert entries
    assert {entry.dataset_id for entry in entries} == {"dataset:neuromorpho:Renner"}


def test_coverage_state_entries_make_missing_dimensions_explicit():
    records = [
        _dataset(
            "dataset:neuromorpho:Renner",
            source="neuromorpho",
            species=["human"],
            modalities=["neuron_morphology"],
        )
    ]

    states = build_coverage_state_entries(records, snapshot_id="test")
    by_dimension = {state.dimension: state for state in states}

    assert len(states) == 8
    assert by_dimension["species"].coverage_state == "observed"
    assert by_dimension["tasks"].coverage_state == "not_applicable"
    assert by_dimension["behavioral_events"].coverage_state == "not_applicable"
    assert by_dimension["tasks"].reason


def test_gap_report_tracks_missing_and_low_confidence_silver_labels():
    records = [
        _dataset(
            "dataset:dandi:000640",
            species=["rat"],
            brain_regions=["mPFC"],
            modalities=["lfp"],
            provenance="gemini_flash_inferred_silver_not_human_gold",
        ),
        _dataset("dataset:neurovault:001", source="neurovault", species=["human"]),
    ]

    report = build_gap_report(records, snapshot_id="test")

    assert report.dataset_count == 2
    assert report.missing_dimension_counts["brain_regions"] == 1
    assert report.low_confidence_counts["brain_regions"] == 1
    assert report.access_tier_counts["open_access"] == 2
    assert any("brain_regions" in item for item in report.recommendations)


def test_gap_report_tracks_state_coverage_separately_from_value_coverage():
    records = [
        _dataset(
            "dataset:neuromorpho:Renner",
            source="neuromorpho",
            species=["human"],
            modalities=["neuron_morphology"],
        ),
        {
            "source": "unknown_source",
            "source_id": "001",
            "title": "Sparse unknown record",
        },
    ]

    report = build_gap_report(records, snapshot_id="test")

    assert report.coverage_rates["tasks"] == 0.0
    assert report.field_state_coverage_rates["tasks"] == 1.0
    assert report.not_applicable_counts["tasks"] == 1
    assert report.unknown_state_counts["species"] == 1
    assert report.actionable_state_rates["species"] == 0.5
    assert report.source_state_coverage
    assert report.completion_worklist_summary


def test_completion_worklist_uses_source_specific_actions():
    records = [
        {
            "source": "openneuro",
            "source_id": "ds000001",
            "title": "OpenNeuro sparse BIDS record",
            "species": ["human"],
        },
        {
            "source": "dandi",
            "source_id": "000001",
            "title": "DANDI sparse NWB record",
            "species": ["mouse"],
            "modalities": ["neuropixels"],
        },
    ]

    worklist = build_completion_worklist(records, snapshot_id="test")
    actions = {
        (item.source, item.dimension): item.recommended_action
        for item in worklist
    }

    assert any(item.current_state == "unknown_needs_review" for item in worklist)
    assert "BIDS" in actions[("openneuro", "brain_regions")]
    assert "NWB" in actions[("dandi", "tasks")]
    assert worklist[0].priority >= worklist[-1].priority


def test_gap_report_surfaces_dark_cross_dimension_pairs():
    records = [
        _dataset(
            "dataset:dandi:mouse_ca1",
            species=["mouse"],
            brain_regions=["ca1"],
            modalities=["neuropixels"],
        ),
        _dataset(
            "dataset:dandi:rat_pfc",
            species=["rat"],
            brain_regions=["prefrontal_cortex"],
            modalities=["lfp"],
        ),
    ]

    report = build_gap_report(records, snapshot_id="test")

    assert {
        ("mouse", "prefrontal_cortex"),
        ("rat", "ca1"),
    } & {(item["left"], item["right"]) for item in report.dark_species_region_pairs}


def test_coverage_ledger_writes_artifacts(tmp_path):
    ledger = CoverageLedger(
        [
            _dataset(
                "dataset:dandi:000004",
                species=["human"],
                brain_regions=["temporal_cortex"],
                modalities=["extracellular_ephys"],
                recording_scales=["single_unit_spikes"],
            )
        ],
        snapshot_id="test",
    )

    paths = ledger.write(tmp_path)

    assert paths["entries"].exists()
    assert paths["states"].exists()
    assert paths["completion_worklist"].exists()
    assert paths["gap_report_json"].exists()
    assert paths["gap_report_md"].read_text(encoding="utf-8").startswith(
        "# Coverage Ledger Gap Report"
    )


def test_modality_compatibility_and_confidence_propagation():
    compat = modality_compatibility("lfp", "eeg")
    spatial = modality_compatibility("calcium_widefield", "fmri")

    assert compat.compatibility_class == "mesoscale_oscillation_comparable"
    assert compat.comparable
    assert spatial.compatibility_class == "mesoscale_spatial_comparable"
    assert propagate_confidence(0.7, 0.8) == 0.5599999999999999
