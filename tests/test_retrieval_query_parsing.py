from neural_search.ontology import normalize_text
from neural_search.search import parse_query


def test_parse_query_detects_scientific_intents():
    parsed = parse_query(
        "Find human OFC Neuropixels data to decode choice with event alignment "
        "and latent state modeling."
    )

    assert "choice" in parsed["behaviors"]
    assert "neuropixels" in parsed["modalities"]
    assert "human" in parsed["species"]
    assert {normalize_text(region) for region in parsed["brain_regions"]} == {"ofc"}
    assert {
        "choice_decoding",
        "event_aligned_activity",
        "latent_state_modeling",
    }.issubset(set(parsed["analysis"]))


def test_parse_query_expands_generic_neural_recording_modalities():
    parsed = parse_query("Go/NoGo datasets with neural recordings and lick events")

    assert "go_nogo" in parsed["tasks"]
    assert "lick" in parsed["behaviors"]
    assert {"calcium_imaging", "extracellular_ephys", "neuropixels"}.issubset(
        set(parsed["modalities"])
    )
    assert parsed["task_intent"][0]["match_type"] in {"label", "synonym", "id"}


def test_parse_query_does_not_treat_ieeg_as_eeg():
    parsed = parse_query("Find iEEG datasets with seizure onset annotations")

    assert "ieeg" in parsed["modalities"]
    assert "eeg" not in parsed["modalities"]
