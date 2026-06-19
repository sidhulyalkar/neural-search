"""Compute 3D spring-layout positions for all cluster graph nodes.

Reads:  artifacts/graph/cluster_graph.json
Writes: artifacts/graph/galaxy_points.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).parent.parent.parent
INPUT_PATH = REPO_ROOT / "artifacts/graph/cluster_graph.json"
OUTPUT_PATH = REPO_ROOT / "artifacts/graph/galaxy_points.json"

# Scale factor — spread nodes across a 2000-unit cube for three.js
SCALE = 400


def compute_layout() -> dict:
    data = json.loads(INPUT_PATH.read_text())
    nodes = data["nodes"]
    links = data["links"]

    G = nx.Graph()
    node_meta = {n["id"]: n for n in nodes}
    G.add_nodes_from(n["id"] for n in nodes)
    for link in links:
        G.add_edge(link["source"], link["target"])

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("Computing 3D spring layout (may take 30–60 s for large graphs)…")

    # Use scale_level as a y-axis seed to enforce the biological-scale axis.
    # Nodes with higher scale_level (system/dataset) start higher in y.
    seed_pos: dict[str, tuple[float, float, float]] = {}
    rng = random.Random(42)
    for n in nodes:
        scale = node_meta[n["id"]]["scale_level"]
        seed_pos[n["id"]] = (
            rng.uniform(-1, 1),
            (scale - 2) * 0.3 + rng.uniform(-0.1, 0.1),
            rng.uniform(-1, 1),
        )

    pos = nx.spring_layout(G, dim=3, seed=42, pos=seed_pos, k=0.8, iterations=80)

    galaxy_nodes = []
    for n in nodes:
        nid = n["id"]
        if nid not in pos:
            continue
        x, y, z = pos[nid]
        galaxy_nodes.append({
            "id": nid,
            "x": round(float(x) * SCALE, 2),
            "y": round(float(y) * SCALE, 2),
            "z": round(float(z) * SCALE, 2),
            "type": n["type"],
            "color": n["color"],
            "size": n["size"],
            "label": n["label"],
        })

    return {"nodes": galaxy_nodes}


def main() -> None:
    layout = compute_layout()
    OUTPUT_PATH.write_text(json.dumps(layout))
    print(f"Done — {len(layout['nodes'])} positioned nodes → {OUTPUT_PATH}")
    sizes = [abs(n["x"]) for n in layout["nodes"]]
    print(f"X range: {min(sizes):.1f} – {max(sizes):.1f}")


if __name__ == "__main__":
    main()
