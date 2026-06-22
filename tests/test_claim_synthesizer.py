from __future__ import annotations

import pytest
from neural_search.literature.claim_synthesizer import (
    cluster_findings,
    detect_contradictions,
    _claim_id_from_cluster,
    _opposite_directions,
)

FINDINGS = [
    {
        "finding_id": f"f{i}",
        "paper_id": f"paper:openalex:W{i}",
        "finding_text": "Theta oscillations increase during spatial navigation",
        "result_direction": "increase",
        "regions_normalized": ["hippocampus"],
        "regions": ["hippocampus"],
        "species": ["mouse"],
        "tasks": ["spatial navigation"],
        "confidence": 0.9,
    }
    for i in range(5)
]

FINDINGS_MIXED = [
    *FINDINGS,
    {
        "finding_id": "f_contra",
        "paper_id": "paper:openalex:W99",
        "finding_text": "Theta oscillations decrease after lesion",
        "result_direction": "decrease",
        "regions_normalized": ["hippocampus"],
        "regions": ["hippocampus"],
        "species": ["mouse"],
        "tasks": ["spatial navigation"],
        "confidence": 0.8,
    },
]


def test_cluster_findings_groups_by_region_direction_species():
    clusters = cluster_findings(FINDINGS, min_size=2)
    assert len(clusters) == 1
    c = clusters[0]
    assert c["direction"] == "increase"
    assert "hippocampus" in c["regions"]
    assert c["n_findings"] == 5


def test_cluster_findings_min_size_filters_small_clusters():
    clusters = cluster_findings(FINDINGS[:2], min_size=3)
    assert len(clusters) == 0


def test_cluster_findings_separate_directions():
    clusters = cluster_findings(FINDINGS_MIXED, min_size=1)
    directions = {c["direction"] for c in clusters}
    assert "increase" in directions
    assert "decrease" in directions


def test_claim_id_from_cluster_is_stable():
    clusters = cluster_findings(FINDINGS, min_size=1)
    id1 = _claim_id_from_cluster(clusters[0])
    id2 = _claim_id_from_cluster(clusters[0])
    assert id1 == id2
    assert id1.startswith("node:claim:")


def test_opposite_directions():
    assert _opposite_directions("increase", "decrease")
    assert _opposite_directions("decrease", "increase")
    assert not _opposite_directions("increase", "correlation")
    assert not _opposite_directions("increase", "increase")


def test_detect_contradictions_marks_contested_claims():
    claims = [
        {
            "claim_id": "node:claim:theta_increase_001",
            "direction": "increase",
            "regions": ["hippocampus"],
            "species": ["mouse"],
            "contradicted_by": [],
            "status": "active",
        },
        {
            "claim_id": "node:claim:theta_decrease_001",
            "direction": "decrease",
            "regions": ["hippocampus"],
            "species": ["mouse"],
            "contradicted_by": [],
            "status": "active",
        },
    ]
    result = detect_contradictions(claims)
    assert "node:claim:theta_decrease_001" in result[0]["contradicted_by"]
    assert result[0]["status"] == "contested"
    assert result[1]["status"] == "contested"


def test_detect_contradictions_no_false_positives():
    claims = [
        {
            "claim_id": "node:claim:theta_increase_001",
            "direction": "increase",
            "regions": ["hippocampus"],
            "species": ["mouse"],
            "contradicted_by": [],
            "status": "active",
        },
        {
            "claim_id": "node:claim:pfc_increase_001",
            "direction": "increase",
            "regions": ["prefrontal cortex"],
            "species": ["mouse"],
            "contradicted_by": [],
            "status": "active",
        },
    ]
    result = detect_contradictions(claims)
    assert result[0]["status"] == "active"
    assert result[1]["status"] == "active"
