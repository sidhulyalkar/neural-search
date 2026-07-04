"""Tests for neural_search.graph.reprocessing_candidate_builder."""

from __future__ import annotations

import json

from neural_search.graph.reprocessing_candidate_builder import (
    attach_reprocessing_candidate_status,
)
from neural_search.graph.schema import KnowledgeGraph, KnowledgeGraphNode


def _write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _dataset_graph(node_id: str) -> KnowledgeGraph:
    return KnowledgeGraph(
        nodes={node_id: KnowledgeGraphNode(node_id=node_id, node_type="dataset", label=node_id)},
        edges={},
    )


def test_flags_dataset_with_stale_nwb_version(tmp_path) -> None:
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        validation_path,
        [
            {
                "dataset_node_id": "node:dataset:dandi:000458",
                "validator": "dandi",
                "raw_result": {"nwb_version": "2.2.5", "asset_path": "sub-01/sub-01.nwb"},
            }
        ],
    )
    graph = _dataset_graph("node:dataset:dandi:000458")

    updated = attach_reprocessing_candidate_status(graph, validation_path=validation_path)

    status = updated.nodes["node:dataset:dandi:000458"].properties["reprocessing_candidate"]
    assert status["nwb_version"] == "2.2.5"
    assert status["threshold"] == "2.6.0"


def test_does_not_flag_current_nwb_version(tmp_path) -> None:
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        validation_path,
        [
            {
                "dataset_node_id": "node:dataset:dandi:000458",
                "validator": "dandi",
                "raw_result": {"nwb_version": "2.7.0"},
            }
        ],
    )
    graph = _dataset_graph("node:dataset:dandi:000458")

    updated = attach_reprocessing_candidate_status(graph, validation_path=validation_path)

    assert "reprocessing_candidate" not in updated.nodes["node:dataset:dandi:000458"].properties


def test_ignores_non_dandi_rows(tmp_path) -> None:
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        validation_path,
        [
            {
                "dataset_node_id": "node:dataset:openneuro:ds000117",
                "validator": "openneuro",
                "raw_result": {"nwb_version": "2.0.0"},
            }
        ],
    )
    graph = _dataset_graph("node:dataset:openneuro:ds000117")

    updated = attach_reprocessing_candidate_status(graph, validation_path=validation_path)

    assert "reprocessing_candidate" not in updated.nodes["node:dataset:openneuro:ds000117"].properties


def test_skips_dataset_not_in_graph(tmp_path) -> None:
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        validation_path,
        [
            {
                "dataset_node_id": "node:dataset:dandi:999999",
                "validator": "dandi",
                "raw_result": {"nwb_version": "2.1.0"},
            }
        ],
    )
    graph = _dataset_graph("node:dataset:dandi:000458")

    updated = attach_reprocessing_candidate_status(graph, validation_path=validation_path)

    assert updated.nodes["node:dataset:dandi:000458"].properties == {}


def test_is_noop_when_artifact_missing(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.jsonl"
    graph = _dataset_graph("node:dataset:dandi:000458")

    updated = attach_reprocessing_candidate_status(graph, validation_path=missing)

    assert updated.nodes["node:dataset:dandi:000458"].properties == {}


def test_does_not_mutate_input_graph(tmp_path) -> None:
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        validation_path,
        [{"dataset_node_id": "node:dataset:dandi:000458", "validator": "dandi", "raw_result": {"nwb_version": "2.0.0"}}],
    )
    graph = _dataset_graph("node:dataset:dandi:000458")

    attach_reprocessing_candidate_status(graph, validation_path=validation_path)

    assert "reprocessing_candidate" not in graph.nodes["node:dataset:dandi:000458"].properties


def test_malformed_version_string_is_not_flagged(tmp_path) -> None:
    validation_path = tmp_path / "validation.jsonl"
    _write_jsonl(
        validation_path,
        [{"dataset_node_id": "node:dataset:dandi:000458", "validator": "dandi", "raw_result": {"nwb_version": "unknown"}}],
    )
    graph = _dataset_graph("node:dataset:dandi:000458")

    updated = attach_reprocessing_candidate_status(graph, validation_path=validation_path)

    assert "reprocessing_candidate" not in updated.nodes["node:dataset:dandi:000458"].properties
