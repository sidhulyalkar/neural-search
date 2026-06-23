from __future__ import annotations

import json
from unittest.mock import MagicMock

from neural_search.literature.claim_synthesizer import (
    _claim_id_from_cluster,
    _opposite_directions,
    cluster_findings,
    detect_contradictions,
    synthesize_claim,
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


def test_cluster_findings_aggregates_frequency_bands_and_injury_models():
    findings = [
        {**FINDINGS[0], "frequency_band": ["theta"], "injury_model": ["alzheimer_app"]},
        {**FINDINGS[1], "frequency_band": ["theta"], "injury_model": []},
        {**FINDINGS[2], "frequency_band": [], "injury_model": []},
    ]
    clusters = cluster_findings(findings, min_size=1)
    assert clusters[0]["frequency_bands"] == ["theta"]
    assert clusters[0]["injury_models"] == ["alzheimer_app"]


def test_cluster_findings_without_typed_fields_has_empty_aggregates():
    clusters = cluster_findings(FINDINGS, min_size=1)
    assert clusters[0]["frequency_bands"] == []
    assert clusters[0]["injury_models"] == []


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


def _claim(claim_id, direction, regions, *, frequency_bands=None, injury_models=None):
    return {
        "claim_id": claim_id,
        "direction": direction,
        "regions": regions,
        "species": ["mouse"],
        "frequency_bands": frequency_bands or [],
        "injury_models": injury_models or [],
        "contradicted_by": [],
        "status": "active",
    }


def test_detect_contradictions_blocked_by_mismatched_frequency_bands():
    claims = [
        _claim("c1", "increase", ["hippocampus"], frequency_bands=["theta"]),
        _claim("c2", "decrease", ["hippocampus"], frequency_bands=["gamma"]),
    ]
    result = detect_contradictions(claims)
    assert result[0]["status"] == "active"
    assert result[1]["status"] == "active"


def test_detect_contradictions_still_fires_for_matching_frequency_bands():
    claims = [
        _claim("c1", "increase", ["hippocampus"], frequency_bands=["theta"]),
        _claim("c2", "decrease", ["hippocampus"], frequency_bands=["theta"]),
    ]
    result = detect_contradictions(claims)
    assert result[0]["status"] == "contested"
    assert result[1]["status"] == "contested"


def test_detect_contradictions_one_sided_frequency_band_does_not_block():
    claims = [
        _claim("c1", "increase", ["hippocampus"], frequency_bands=["theta"]),
        _claim("c2", "decrease", ["hippocampus"]),  # no frequency_bands at all
    ]
    result = detect_contradictions(claims)
    assert result[0]["status"] == "contested"
    assert result[1]["status"] == "contested"


def test_detect_contradictions_blocked_by_mismatched_injury_models():
    claims = [
        _claim("c1", "increase", ["hippocampus"], injury_models=["alzheimer_app"]),
        _claim("c2", "decrease", ["hippocampus"], injury_models=["parkinson_6ohda"]),
    ]
    result = detect_contradictions(claims)
    assert result[0]["status"] == "active"
    assert result[1]["status"] == "active"


def test_detect_contradictions_missing_typed_fields_falls_back_to_region_only():
    # Old-style claim dicts with no frequency_bands/injury_models keys at all
    # must keep today's region-only behavior.
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
    assert result[0]["status"] == "contested"
    assert result[1]["status"] == "contested"


class TestSynthesizeClaimPropagatesTypedFields:
    def test_frequency_bands_and_injury_models_propagate_to_claim(self):
        cluster = {
            "cluster_id": "node:claim:hippocampus_increase_abc",
            "regions": ["hippocampus"],
            "direction": "increase",
            "species": ["mouse"],
            "n_findings": 1,
            "frequency_bands": ["theta"],
            "injury_models": ["alzheimer_app"],
            "findings": [
                {
                    "finding_text": "Theta power increases in hippocampus.",
                    "confidence": 0.9,
                    "paper_id": "paper:openalex:W1",
                }
            ],
        }
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text=json.dumps({
                "statement": "Theta power increases in hippocampus.",
                "magnitude_summary": "moderate",
                "timescale": "second",
                "evidence_strength": "direct",
            }))
        ]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        claim = synthesize_claim(
            cluster,
            mock_client,
            config={"model": "test-model", "user_template": "{regions} {direction} {n_findings} {findings_text} {species}"},
        )

        assert claim["frequency_bands"] == ["theta"]
        assert claim["injury_models"] == ["alzheimer_app"]

    def test_missing_aggregates_default_to_empty_list(self):
        cluster = {
            "cluster_id": "node:claim:hippocampus_increase_def",
            "regions": ["hippocampus"],
            "direction": "increase",
            "species": ["mouse"],
            "n_findings": 1,
            "findings": [
                {
                    "finding_text": "Theta power increases in hippocampus.",
                    "confidence": 0.9,
                    "paper_id": "paper:openalex:W1",
                }
            ],
        }
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text=json.dumps({"statement": "x", "magnitude_summary": "N/A", "timescale": "unknown", "evidence_strength": "indirect"}))
        ]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        claim = synthesize_claim(
            cluster,
            mock_client,
            config={"model": "test-model", "user_template": "{regions} {direction} {n_findings} {findings_text} {species}"},
        )

        assert claim["frequency_bands"] == []
        assert claim["injury_models"] == []
