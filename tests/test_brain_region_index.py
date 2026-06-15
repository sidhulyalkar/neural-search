from neural_search.corpus.brain_region_index import build_brain_region_index


def test_brain_region_index_expands_hierarchy_and_categories():
    index = build_brain_region_index()

    ca1 = index["ca1"]
    hippocampus = index["hippocampus"]
    barrel = index["barrel_cortex"]

    assert ca1.system == "hippocampal_formation"
    assert "hippocampus" in ca1.parents
    assert "ca1" in hippocampus.children
    assert "brain_system:hippocampal_formation" in ca1.index_categories
    assert "parent_region:hippocampus" in ca1.index_categories
    assert "atlas:neural_search_region:ca1" in ca1.index_categories
    assert "species_scope:mouse" in barrel.index_categories
