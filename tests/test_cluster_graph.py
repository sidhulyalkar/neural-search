import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.literature.build_cluster_graph import (
    DIRECTION_COLORS,
    assign_system,
    build_graph,
)


def test_direction_colors_complete():
    for direction in ("increase", "decrease", "correlation", "no_change"):
        assert direction in DIRECTION_COLORS


def test_assign_system_known_region():
    assert assign_system("hippocampus") == "hippocampal_formation"


def test_assign_system_unknown_region():
    assert assign_system("unknown_region_xyz") == "other"


def test_build_graph_returns_nodes_and_links():
    graph = build_graph(max_edges=100)
    assert "nodes" in graph
    assert "links" in graph
    assert len(graph["nodes"]) > 0
    assert len(graph["links"]) > 0


def test_build_graph_node_schema():
    graph = build_graph(max_edges=10)
    for node in graph["nodes"]:
        assert "id" in node
        assert "type" in node
        assert node["type"] in ("system", "region", "finding_cluster", "dataset", "paper")
        assert "label" in node
        assert "scale_level" in node
        assert "size" in node
        assert "color" in node
        assert "meta" in node


def test_build_graph_link_schema():
    graph = build_graph(max_edges=10)
    node_ids = {n["id"] for n in graph["nodes"]}
    for link in graph["links"]:
        assert "source" in link
        assert "target" in link
        assert "type" in link
        assert "weight" in link
        assert "color" in link
        assert link["source"] in node_ids, f"source {link['source']} not in nodes"
        assert link["target"] in node_ids, f"target {link['target']} not in nodes"


def test_build_graph_no_duplicate_node_ids():
    graph = build_graph(max_edges=10)
    ids = [n["id"] for n in graph["nodes"]]
    assert len(ids) == len(set(ids)), "Duplicate node IDs found"


def test_build_graph_system_nodes_present():
    graph = build_graph(max_edges=10)
    system_nodes = [n for n in graph["nodes"] if n["type"] == "system"]
    assert len(system_nodes) >= 5
