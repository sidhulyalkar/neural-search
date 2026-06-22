from neural_search.graph.schema import SUPPORTED_NODE_TYPES, SUPPORTED_EDGE_TYPES


def test_claim_node_type_registered():
    assert "claim" in SUPPORTED_NODE_TYPES


def test_claim_edge_types_registered():
    for edge_type in (
        "claim_supports_finding",
        "claim_contradicts_claim",
        "claim_supported_by_dataset",
        "claim_supported_by_paper",
        "claim_derived_from_finding",
    ):
        assert edge_type in SUPPORTED_EDGE_TYPES, f"Missing edge type: {edge_type}"
