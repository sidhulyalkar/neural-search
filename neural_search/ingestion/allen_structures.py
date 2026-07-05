"""Allen Brain Atlas structure graph fetcher and flattener.

Fetches the CCF mouse (atlas_id=1) and Human Brain Atlas (atlas_id=10)
structure trees from the Allen Brain Atlas public API and flattens them
into a list of AllenStructure dataclass instances.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

ALLEN_STRUCTURE_GRAPH_URL = "https://api.brain-map.org/api/v2/structure_graph_download/{atlas_id}.json"
USER_AGENT = "neuralsearch/0.1 (neuralsearch@example.com)"

ATLAS_NAMES: dict[int, str] = {
    1: "allen_ccf_mouse",
    10: "allen_human",
}

ARTIFACTS_DIR = Path(__file__).parent.parent.parent / "artifacts" / "atlas"


@dataclass
class AllenStructure:
    allen_id: int
    acronym: str
    name: str
    parent_id: int | None
    color_hex: str
    graph_order: int
    st_level: int
    atlas_id: int
    children_ids: list[int] = field(default_factory=list)


def fetch_allen_structure_graph(atlas_id: int = 1) -> dict[str, Any]:
    """Fetch the Allen Brain Atlas structure graph for the given atlas.

    atlas_id=1 is the mouse CCF (Common Coordinate Framework).
    atlas_id=10 is the Human Brain Atlas (HBA).
    Returns the raw JSON dict, or an empty dict on failure.
    """
    url = ALLEN_STRUCTURE_GRAPH_URL.format(atlas_id=atlas_id)
    try:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(
                url,
                params={"mailto": "neuralsearch@example.com"},
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as exc:
        logger.warning("Allen structure graph fetch failed (atlas_id=%d): %s", atlas_id, exc)
        return {}
    except Exception as exc:
        logger.warning("Unexpected error fetching Allen structure graph (atlas_id=%d): %s", atlas_id, exc)
        return {}

    if not data.get("success"):
        logger.warning("Allen API returned success=false for atlas_id=%d", atlas_id)
        return {}

    return data


def _flatten_node(
    node: dict[str, Any],
    atlas_id: int,
    parent_id: int | None,
    results: list[AllenStructure],
) -> None:
    """Recursively visit one tree node and append a flat AllenStructure."""
    children = node.get("children") or []
    children_ids = [int(c["id"]) for c in children if "id" in c]

    results.append(
        AllenStructure(
            allen_id=int(node["id"]),
            acronym=node.get("acronym", ""),
            name=node.get("safe_name") or node.get("name", ""),
            parent_id=parent_id,
            color_hex=node.get("color_hex_triplet", ""),
            graph_order=int(node.get("graph_order") or 0),
            st_level=int(node.get("st_level") or 0),
            atlas_id=atlas_id,
            children_ids=children_ids,
        )
    )

    for child in children:
        _flatten_node(child, atlas_id, int(node["id"]), results)


def flatten_structure_tree(tree: dict[str, Any]) -> list[AllenStructure]:
    """Recursively flatten the nested Allen structure tree into a flat list.

    Expects the raw API response dict (with a ``msg`` list at the top level).
    Returns an empty list if the tree is empty or malformed.
    """
    msg = tree.get("msg") or []
    results: list[AllenStructure] = []
    for root in msg:
        _flatten_node(root, root.get("graph_id", 1), None, results)
    return results


def fetch_and_flatten(atlas_id: int) -> list[AllenStructure]:
    """Convenience: fetch + flatten in one call."""
    raw = fetch_allen_structure_graph(atlas_id)
    if not raw:
        return []
    return flatten_structure_tree(raw)


def save_structures(structures: list[AllenStructure], output_path: Path) -> None:
    """Serialize a list of AllenStructure instances to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(s) for s in structures]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %d structures to %s", len(structures), output_path)


def load_structures(json_path: Path) -> list[AllenStructure]:
    """Load AllenStructure instances from a previously saved JSON file."""
    if not json_path.exists():
        return []
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return [AllenStructure(**item) for item in raw]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    for atlas_id, name in ATLAS_NAMES.items():
        logger.info("Fetching Allen structure graph: atlas_id=%d (%s)", atlas_id, name)
        structures = fetch_and_flatten(atlas_id)
        if not structures:
            logger.warning("No structures returned for atlas_id=%d; skipping save.", atlas_id)
            continue
        out = ARTIFACTS_DIR / f"allen_{name.removeprefix('allen_')}_structures.json"
        save_structures(structures, out)
        logger.info("Wrote %d structures → %s", len(structures), out)
