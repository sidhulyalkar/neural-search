from neural_search.extraction import extract_dataset_labels
from neural_search.ontology import (
    expand_brain_region_ids,
    expand_brain_region_query,
    get_brain_regions,
    match_brain_regions,
)
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


def test_species_aware_m1_alias_maps_to_primary_motor_child():
    mouse_ids = {match.id for match in match_brain_regions("mouse M1 neuropixels")}
    context_free_ids = {match.id for match in match_brain_regions("M1 finance data")}

    assert {"primary_motor_cortex", "motor_cortex"} <= mouse_ids
    assert "primary_motor_cortex" not in context_free_ids
    assert "motor_cortex" not in context_free_ids


def test_barrel_cortex_maps_to_somatosensory_hierarchy():
    ids = {match.id for match in match_brain_regions("mouse barrel cortex whisker ephys")}

    assert {"barrel_cortex", "primary_somatosensory_cortex", "somatosensory_cortex"} <= ids
    assert "somatosensory_area_2" not in ids


def test_hippocampus_descendant_expansion_is_explicit():
    broad = set(expand_brain_region_ids(["hippocampus"]))
    exact = set(expand_brain_region_ids(["hippocampus"], include_descendants=False))
    query = expand_brain_region_query("hippocampus calcium imaging")

    assert {"hippocampus", "ca1", "ca2", "ca3", "dentate_gyrus", "subiculum"} <= broad
    assert exact == {"hippocampus"}
    assert "ca1" in query["expanded_region_ids"]


def test_spinal_horn_siblings_do_not_cross_match():
    dorsal_ids = {match.id for match in match_brain_regions("dorsal horn spinal cord calcium imaging")}
    ventral_ids = {match.id for match in match_brain_regions("ventral horn motor neuron recordings")}

    assert {"dorsal_horn", "spinal_cord"} <= dorsal_ids
    assert "ventral_horn" not in dorsal_ids
    assert {"ventral_horn", "spinal_cord"} <= ventral_ids
    assert "dorsal_horn" not in ventral_ids


def test_brain_region_metadata_supports_system_and_species_scope():
    regions = {region.id: region for region in get_brain_regions()}

    assert regions["barrel_cortex"].system == "cortical"
    assert regions["barrel_cortex"].species_scope == ["mouse", "rat"]
    assert regions["primary_motor_cortex"].species_aliases["mouse"] == ["m1"]
