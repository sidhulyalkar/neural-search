import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from apps.api.main import app

client = TestClient(app)


def test_graph_overview_returns_nodes_and_links():
    r = client.get("/api/graph/overview")
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert "links" in data
    assert "meta" in data
    assert data["meta"]["node_count"] == len(data["nodes"])


def test_graph_overview_node_schema():
    r = client.get("/api/graph/overview")
    data = r.json()
    for node in data["nodes"][:5]:
        assert "id" in node
        assert "type" in node
        assert node["type"] in ("system", "region", "finding_cluster", "dataset", "paper")
        assert "label" in node
        assert "color" in node


def test_graph_subgraph_with_region_filter():
    r = client.get("/api/graph/subgraph", params={"regions": "hippocampus"})
    assert r.status_code == 200
    data = r.json()
    assert "nodes" in data
    assert len(data["nodes"]) > 0
    # Should include a hippocampus region or cluster node
    ids = [n["id"] for n in data["nodes"]]
    assert any("hippocampus" in nid for nid in ids)


def test_graph_subgraph_respects_limit():
    r = client.get("/api/graph/subgraph", params={"limit": 10})
    assert r.status_code == 200
    data = r.json()
    assert len(data["nodes"]) <= 10


def test_graph_topic_hippocampal():
    r = client.get("/api/graph/topic/hippocampal")
    assert r.status_code == 200
    data = r.json()
    assert "topic" in data
    assert data["topic"]["slug"] == "hippocampal"
    assert "nodes" in data


def test_graph_topic_unknown_slug():
    r = client.get("/api/graph/topic/nonexistent_slug_xyz")
    assert r.status_code == 404


def test_literature_consensus():
    r = client.get("/api/literature/consensus")
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    if rows:
        row = rows[0]
        assert "region" in row
        assert "direction" in row
        assert "n_findings" in row
        assert "consensus_strength" in row


def test_literature_findings_with_region():
    r = client.get("/api/literature/findings", params={"region": "hippocampus", "limit": 5})
    assert r.status_code == 200
    rows = r.json()
    assert isinstance(rows, list)
    assert len(rows) <= 5


def test_dataset_neighborhood():
    r = client.get("/api/datasets/dandi:000003/neighborhood")
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        data = r.json()
        assert "dataset_id" in data
        assert "linked_papers" in data
        assert "finding_clusters" in data
        assert "consensus_by_region" in data
