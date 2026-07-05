"""Tests for the Scholarpedia concept authority KG builder."""

from __future__ import annotations

from neural_search.graph.schema import validate_graph
from neural_search.ingestion.scholarpedia_builder import (
    SCHOLARPEDIA_CONCEPTS,
    build_scholarpedia_kg,
)
from neural_search.search.concept_authority import (
    expand_query_with_concepts,
    get_concept_aliases,
    load_scholarpedia_index,
)


def test_build_scholarpedia_kg_returns_nodes_and_edges():
    kg = build_scholarpedia_kg()
    assert len(kg.nodes) > 0
    assert len(kg.edges) > 0


def test_build_scholarpedia_kg_has_expected_concept_count():
    kg = build_scholarpedia_kg()
    concept_nodes = [n for n in kg.nodes.values() if not n.properties.get("is_domain_node") and not n.properties.get("is_alias_node")]
    assert len(concept_nodes) >= len(SCHOLARPEDIA_CONCEPTS)


def test_build_scholarpedia_kg_validates():
    kg = build_scholarpedia_kg()
    validate_graph(kg)


def test_build_scholarpedia_kg_all_records_carry_license():
    kg = build_scholarpedia_kg()
    for node in kg.nodes.values():
        if node.properties.get("source") == "scholarpedia":
            assert node.properties.get("license") == "CC BY-NC-SA 3.0"
    for edge in kg.edges.values():
        if "scholarpedia" in edge.properties.get("source", ""):
            assert edge.properties.get("license") == "CC BY-NC-SA 3.0"


def test_build_scholarpedia_kg_has_domain_nodes():
    kg = build_scholarpedia_kg()
    domains = {e["domain"] for e in SCHOLARPEDIA_CONCEPTS.values()}
    domain_nodes = [n for n in kg.nodes.values() if n.properties.get("is_domain_node")]
    domain_labels = {n.node_id for n in domain_nodes}
    for domain in domains:
        assert any(domain in nid for nid in domain_labels)


def test_build_scholarpedia_kg_has_related_edges():
    kg = build_scholarpedia_kg()
    related_edges = [e for e in kg.edges.values() if e.edge_type == "concept_related_to_concept"]
    assert len(related_edges) > 0


def test_build_scholarpedia_kg_related_edges_require_human_review():
    kg = build_scholarpedia_kg()
    for edge in kg.edges.values():
        if edge.edge_type == "concept_related_to_concept":
            assert edge.properties.get("requires_human_review") is True


def test_build_scholarpedia_kg_has_alias_edges():
    kg = build_scholarpedia_kg()
    alias_edges = [e for e in kg.edges.values() if e.edge_type == "concept_has_alias"]
    assert len(alias_edges) > 0


def test_build_scholarpedia_kg_has_domain_edges():
    kg = build_scholarpedia_kg()
    domain_edges = [e for e in kg.edges.values() if e.edge_type == "concept_in_domain"]
    assert len(domain_edges) == len(SCHOLARPEDIA_CONCEPTS)


def test_build_scholarpedia_kg_metadata_populated():
    kg = build_scholarpedia_kg()
    assert kg.metadata.get("concept_count") == len(SCHOLARPEDIA_CONCEPTS)
    assert kg.metadata.get("source") == "scholarpedia"
    assert kg.metadata.get("license") == "CC BY-NC-SA 3.0"


def test_load_scholarpedia_index_returns_non_empty_dict():
    index = load_scholarpedia_index()
    assert isinstance(index, dict)
    assert len(index) > 0


def test_load_scholarpedia_index_contains_canonical_slugs():
    index = load_scholarpedia_index()
    for slug in SCHOLARPEDIA_CONCEPTS:
        readable = slug.replace("_", " ")
        assert readable in index or slug in index


def test_load_scholarpedia_index_maps_stdp():
    index = load_scholarpedia_index()
    assert index.get("stdp") == "spike_timing_dependent_plasticity"


def test_load_scholarpedia_index_maps_ltp():
    index = load_scholarpedia_index()
    assert index.get("ltp") == "long_term_potentiation"


def test_load_scholarpedia_index_maps_pca():
    index = load_scholarpedia_index()
    assert index.get("pca") == "dimensionality_reduction"


def test_get_concept_aliases_known_concept():
    aliases = get_concept_aliases("spike_timing_dependent_plasticity")
    assert len(aliases) > 0
    assert "STDP" in aliases


def test_get_concept_aliases_unknown_returns_empty():
    assert get_concept_aliases("this_does_not_exist") == []


def test_get_concept_aliases_hippocampus():
    aliases = get_concept_aliases("hippocampus")
    assert "CA1" in aliases or "hippocampus" in aliases


def test_expand_query_finds_predictive_coding():
    result = expand_query_with_concepts("predictive coding hippocampus")
    assert "predictive_coding" in result
    assert "hippocampus" in result


def test_expand_query_stdp_alias():
    result = expand_query_with_concepts("datasets with STDP protocols")
    assert "spike_timing_dependent_plasticity" in result


def test_expand_query_ltp_alias():
    result = expand_query_with_concepts("long-term potentiation in CA1")
    assert "long_term_potentiation" in result
    assert "hippocampus" in result


def test_expand_query_no_matches_returns_empty():
    result = expand_query_with_concepts("foobar baz xyz irrelevant")
    assert result == []


def test_expand_query_empty_string():
    assert expand_query_with_concepts("") == []


def test_expand_query_deduplicates():
    result = expand_query_with_concepts("hippocampus hippocampus theta oscillations")
    assert result.count("hippocampus") == 1


def test_expand_query_gamma_oscillations():
    result = expand_query_with_concepts("gamma band activity during visual processing")
    assert "gamma_oscillations" in result


def test_expand_query_reinforcement_learning():
    result = expand_query_with_concepts("dopamine and temporal difference learning in basal ganglia")
    assert "reinforcement_learning" in result


def test_expand_query_returns_list():
    result = expand_query_with_concepts("predictive coding")
    assert isinstance(result, list)
