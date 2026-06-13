from neural_search.extraction import extract_dataset_labels
from neural_search.ontology import match_brain_regions
from neural_search.search.core import parse_query


def _ids(labels):
    return {label.id for label in labels}


def test_macaque_area_mt_query_extracts_precise_visual_motion_region():
    extraction = extract_dataset_labels(
        title="Macaque area MT visual motion single-unit recordings",
        description="Single-unit extracellular ephys from middle temporal visual area during motion stimuli.",
    )

    assert "macaque" in _ids(extraction.species)
    assert "extracellular_ephys" in _ids(extraction.modalities)
    assert {"area_mt", "visual_cortex"} <= _ids(extraction.brain_regions)
    assert "somatosensory_area_2" not in _ids(extraction.brain_regions)


def test_dorsomedial_frontal_cortex_expands_to_prefrontal_parent():
    matches = match_brain_regions("rat LFP theta in dorsomedial frontal cortex")
    ids = {match.id for match in matches}

    assert {"dmFC", "prefrontal_cortex"} <= ids
    assert "mPFC" not in ids
    assert "cortex" not in ids


def test_dorsal_and_ventral_striatum_do_not_cross_match():
    dorsal_ids = {
        match.id for match in match_brain_regions("mouse dorsal striatum spike trains")
    }
    ventral_ids = {
        match.id for match in match_brain_regions("rat ventral striatum LFP")
    }

    assert {"dorsal_striatum", "striatum"} <= dorsal_ids
    assert "ventral_striatum" not in dorsal_ids
    assert {"ventral_striatum", "striatum"} <= ventral_ids
    assert "dorsal_striatum" not in ventral_ids


def test_query_parser_surfaces_precise_region_constraints():
    parsed = parse_query("ephys spike trains dorsal striatum mouse")

    assert "extracellular_ephys" in parsed["modalities"]
    assert "mouse" in parsed["species"]
    assert {"dorsal_striatum", "striatum"} <= set(parsed["brain_regions"])

