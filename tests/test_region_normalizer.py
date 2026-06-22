import json
import pytest
from pathlib import Path
from neural_search.literature.region_normalizer import (
    build_name_index,
    get_parent_chain,
    normalize_region,
    normalize_finding,
)

MOCK_STRUCTURES = {
    "997": {"name": "root", "acronym": "root", "parent_id": None},
    "1080": {"name": "hippocampal formation", "acronym": "hpf", "parent_id": 997},
    "407": {"name": "hippocampus", "acronym": "hc", "parent_id": 1080},
    "382": {"name": "cornu ammonis 1", "acronym": "ca1", "parent_id": 407},
}


def test_build_name_index_includes_names_and_acronyms():
    idx = build_name_index(MOCK_STRUCTURES)
    assert idx["hippocampus"] == 407
    assert idx["hc"] == 407
    assert idx["ca1"] == 382
    assert idx["cornu ammonis 1"] == 382


def test_normalize_region_known():
    idx = build_name_index(MOCK_STRUCTURES)
    assert normalize_region("CA1", idx, MOCK_STRUCTURES) == "cornu ammonis 1"
    assert normalize_region("hippocampus", idx, MOCK_STRUCTURES) == "hippocampus"


def test_normalize_region_unknown_passthrough():
    idx = build_name_index(MOCK_STRUCTURES)
    assert normalize_region("mystery_region", idx, MOCK_STRUCTURES) == "mystery_region"


def test_get_parent_chain():
    chain = get_parent_chain(382, MOCK_STRUCTURES)
    assert "cornu ammonis 1" in chain
    assert "hippocampus" in chain
    assert "hippocampal formation" in chain


def test_normalize_finding_adds_regions_normalized():
    idx = build_name_index(MOCK_STRUCTURES)
    finding = {"finding_id": "f1", "regions": ["CA1", "mystery_region"]}
    result = normalize_finding(finding, idx, MOCK_STRUCTURES)
    assert result["regions_normalized"] == ["cornu ammonis 1", "mystery_region"]
    assert result["finding_id"] == "f1"  # original fields preserved


def test_normalize_finding_empty_regions():
    idx = build_name_index(MOCK_STRUCTURES)
    finding = {"finding_id": "f2", "regions": []}
    result = normalize_finding(finding, idx, MOCK_STRUCTURES)
    assert result["regions_normalized"] == []
