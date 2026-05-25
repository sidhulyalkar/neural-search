from neural_search.graph import (
    build_graph_from_records,
    compute_graph_features_for_result,
    dataset_node_id,
    make_edge_id,
    make_node_id,
)
from neural_search.normalized import make_dataset_id, make_evidence_label_id
from neural_search.schemas import EvidenceLabel, NormalizedDatasetRecord
from neural_search.search import search_datasets


def _label(label_type: str, label: str) -> EvidenceLabel:
    return EvidenceLabel(
        id=make_evidence_label_id(label_type, label),
        label=label,
        label_type=label_type,
        confidence=0.9,
        evidence_text=label,
        extractor_name="test",
        extractor_version="v0.8.0",
    )


def _record(source_id: str, species: str, title: str | None = None) -> dict:
    return {
        "dataset": {
            "id": source_id,
            "source": "demo",
            "source_id": source_id,
            "title": title or f"{species} visual cortex Neuropixels",
            "description": f"{species} neural recordings with behavior events.",
            "species": [species],
            "modalities": ["neuropixels"],
            "brain_regions": ["visual_cortex"],
            "tasks": ["visual_decision_making"],
            "behaviors": ["choice"],
            "data_standards": ["NWB"],
            "has_behavior": True,
            "has_trials": True,
            "license": "CC-BY-4.0",
            "metadata_json": {},
        },
        "card": {
            "dataset_id": source_id,
            "summary": "Reusable neural recording dataset.",
            "scientific_labels": {},
            "analysis_readiness": {"score": 90},
            "missing_fields": [],
            "suggested_analyses": ["event_aligned_activity"],
            "provenance": {},
        },
    }


def test_species_query_expands_non_human_primate_to_macaque_matches():
    response = search_datasets(
        "non-human primate visual cortex neuropixels",
        datasets=[_record("MAC", "macaque"), _record("MOUSE", "mouse")],
        retrieval_config={"graph": {"enabled": False}, "field_embeddings": {"enabled": False}},
    )

    assert response.results[0].dataset_id == "MAC"
    assert "non_human_primate" in response.parsed_query["species"]


def test_species_hard_negative_filters_broader_animal_type():
    response = search_datasets(
        "visual cortex neural recordings without rodents",
        datasets=[_record("MOUSE", "mouse"), _record("HUMAN", "human")],
        retrieval_config={"graph": {"enabled": False}, "field_embeddings": {"enabled": False}},
    )

    assert [result.dataset_id for result in response.results] == ["HUMAN"]
    assert response.filtered_constraints == [
        {"dataset_id": "MOUSE", "violations": ["rodent"]}
    ]


def test_human_only_query_filters_non_human_species():
    response = search_datasets(
        "human only visual cortex recordings",
        datasets=[_record("MOUSE", "mouse"), _record("HUMAN", "human")],
        retrieval_config={"graph": {"enabled": False}, "field_embeddings": {"enabled": False}},
    )

    assert [result.dataset_id for result in response.results] == ["HUMAN"]
    assert response.filtered_constraints == [
        {"dataset_id": "MOUSE", "violations": ["mouse"]}
    ]


def test_graph_builder_adds_species_taxon_context_edges():
    dataset = NormalizedDatasetRecord(
        dataset_id=make_dataset_id("demo", "MAC"),
        source="demo",
        source_id="MAC",
        title="Macaque visual cortex Neuropixels",
        species=[_label("species", "macaque")],
        modalities=[_label("modality", "neuropixels")],
    )

    graph = build_graph_from_records([dataset], [])
    dataset_id = dataset_node_id(dataset)
    species_id = make_node_id("species", "macaque")
    taxon_id = make_node_id("taxon_group", "non_human_primate")

    assert make_edge_id(dataset_id, "dataset_has_species", species_id) in graph.edges
    assert make_edge_id(species_id, "species_in_taxon_group", taxon_id) in graph.edges

    features = compute_graph_features_for_result(
        graph,
        dataset.dataset_id,
        {"species": ["non_human_primate"]},
    )
    assert features["species_context"]["taxon_groups"]
    assert features["matched_query_context"]["taxon_groups"] == ["non human primate"]
