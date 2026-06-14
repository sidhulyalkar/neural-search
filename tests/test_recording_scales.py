from neural_search.search.core import parse_query, score_dataset_against_query


def _dataset(dataset_id: str, title: str, recording_scales: list[str]) -> dict:
    return {
        "id": dataset_id,
        "source_id": dataset_id,
        "source": "demo",
        "title": title,
        "description": title,
        "species": ["mouse"],
        "modalities": ["extracellular_ephys"],
        "recording_scales": recording_scales,
        "brain_regions": ["hippocampus"],
        "tasks": [],
        "behaviors": [],
        "data_standards": ["NWB"],
        "license": "CC-BY",
        "has_raw_data": True,
        "has_processed_data": False,
    }


def test_parse_query_extracts_recording_scale_constraints():
    parsed = parse_query("mouse hippocampus LFP theta oscillations")

    assert "local_field_potential" in parsed["recording_scales"]
    assert "hippocampus" in parsed["brain_regions"]


def test_recording_scale_affects_dataset_scoring():
    parsed = parse_query("mouse hippocampus LFP theta oscillations")
    retrieval_config = {
        "weights": {"metadata": 1.0},
        "required_metadata_fields": [],
        "penalties": {"missing_required_field": 0.0},
    }
    lfp_result = score_dataset_against_query(
        _dataset("LFP_MATCH", "Mouse hippocampus local field potential", ["local_field_potential"]),
        None,
        parsed,
        retrieval_config,
    )
    unit_result = score_dataset_against_query(
        _dataset("UNIT_MATCH", "Mouse hippocampus single-unit spikes", ["single_unit_spikes"]),
        None,
        parsed,
        retrieval_config,
    )

    assert lfp_result.score > unit_result.score
    assert "local field potential" in lfp_result.matched_terms
