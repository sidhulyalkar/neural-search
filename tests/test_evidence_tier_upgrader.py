"""Tests for applying live file-validation results as evidence_tier upgrades."""

from __future__ import annotations

import json

from neural_search.graph.evidence_tier_upgrader import (
    apply_file_validation_upgrades,
    load_confirmed_validations,
)
from neural_search.graph.schema import GraphEdge
from neural_search.kg.schemas.evidence_tier import EvidenceTier


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _candidate_edge(source="node:dataset:dandi:000458", family="time_frequency", tier="heuristic_candidate"):
    return GraphEdge(
        edge_id=f"edge:{source}:candidate:{family}",
        source_node_id=source,
        target_node_id="method:fft",
        edge_type="dataset_old_dataset_new_method_candidate",
        confidence=0.9,
        properties={"analysis_family": family, "evidence_tier": tier},
    )


def test_load_confirmed_validations_filters_to_confirmed_rows(tmp_path):
    path = tmp_path / "validation.jsonl"
    _write_jsonl(
        path,
        [
            {
                "dataset_node_id": "node:dataset:dandi:000458",
                "analysis_family": "time_frequency",
                "confirmed": True,
                "validator": "dandi",
                "source_id": "000458",
            },
            {
                "dataset_node_id": "node:dataset:dandi:000480",
                "analysis_family": "time_frequency",
                "confirmed": False,
                "validator": "dandi",
                "source_id": "000480",
            },
        ],
    )
    confirmed = load_confirmed_validations(path)
    assert list(confirmed.keys()) == [("node:dataset:dandi:000458", "time_frequency")]


def test_load_confirmed_validations_missing_file_returns_empty(tmp_path):
    assert load_confirmed_validations(tmp_path / "missing.jsonl") == {}


def test_apply_upgrades_promotes_matching_edge(tmp_path):
    path = tmp_path / "validation.jsonl"
    _write_jsonl(
        path,
        [
            {
                "dataset_node_id": "node:dataset:dandi:000458",
                "analysis_family": "time_frequency",
                "confirmed": True,
                "validator": "dandi",
                "source_id": "000458",
            }
        ],
    )
    edge = _candidate_edge()
    edges = {edge.edge_id: edge}

    upgraded, count = apply_file_validation_upgrades(edges, path)

    assert count == 1
    new_edge = upgraded[edge.edge_id]
    assert new_edge.properties["evidence_tier"] == EvidenceTier.FILE_VALIDATED.value
    assert new_edge.properties["file_validation"]["validator"] == "dandi"
    # original object must be untouched (immutability)
    assert edge.properties["evidence_tier"] == "heuristic_candidate"


def test_apply_upgrades_skips_non_matching_analysis_family(tmp_path):
    path = tmp_path / "validation.jsonl"
    _write_jsonl(
        path,
        [
            {
                "dataset_node_id": "node:dataset:dandi:000458",
                "analysis_family": "decoding",
                "confirmed": True,
                "validator": "dandi",
                "source_id": "000458",
            }
        ],
    )
    edge = _candidate_edge(family="time_frequency")
    edges = {edge.edge_id: edge}

    upgraded, count = apply_file_validation_upgrades(edges, path)

    assert count == 0
    assert upgraded[edge.edge_id].properties["evidence_tier"] == "heuristic_candidate"


def test_apply_upgrades_never_downgrades_stronger_tier(tmp_path):
    path = tmp_path / "validation.jsonl"
    _write_jsonl(
        path,
        [
            {
                "dataset_node_id": "node:dataset:dandi:000458",
                "analysis_family": "time_frequency",
                "confirmed": True,
                "validator": "dandi",
                "source_id": "000458",
            }
        ],
    )
    edge = _candidate_edge(tier=EvidenceTier.HUMAN_VALIDATED.value)
    edges = {edge.edge_id: edge}

    upgraded, count = apply_file_validation_upgrades(edges, path)

    assert count == 0
    assert upgraded[edge.edge_id].properties["evidence_tier"] == EvidenceTier.HUMAN_VALIDATED.value


def test_apply_upgrades_with_no_validation_file_is_noop(tmp_path):
    edge = _candidate_edge()
    edges = {edge.edge_id: edge}
    upgraded, count = apply_file_validation_upgrades(edges, tmp_path / "missing.jsonl")
    assert count == 0
    assert upgraded == edges
