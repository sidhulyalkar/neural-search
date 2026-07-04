"""Flag datasets whose NWB file predates a recent schema version as
reprocessing candidates.

Reuses `artifacts/validation/top_suggestions_file_validation.jsonl` -- the
same artifact `evidence_tier_upgrader.py` reads -- rather than a new script
or live-network pass. `scripts/validate_top_reanalysis_suggestions.py`
already streams `nwb_version` inside each DANDI row's `raw_result` (read via
the zero-download `HttpRangeFile` header inspection in
`dandi_nwb_validator.py`), so this costs nothing extra to compute.

Modeled as a **node property**, not an edge, following the same reasoning
already used for `attach_retraction_status`: an old NWB schema version is a
fact about that one dataset, not a relationship between two things. A
self-referencing edge would be an schema oddity for no benefit.

The version threshold below is a heuristic trigger for "worth checking
whether reprocessing would add fields (units, epochs, extensions added in
later schema revisions)", never a claim that older data is wrong or unusable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neural_search.graph.schema import KnowledgeGraph

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT / "artifacts" / "validation" / "top_suggestions_file_validation.jsonl"
)

# NWB 2.6.0 (2023) is the cutoff used here: update this as the schema moves
# on. This is a heuristic staleness signal, not a correctness cutoff.
CURRENT_NWB_SCHEMA_VERSION_THRESHOLD = (2, 6, 0)


def _parse_version(version: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(part) for part in version.strip().split("."))
    except (ValueError, AttributeError):
        return None


def _is_stale(version: str) -> bool:
    parsed = _parse_version(version)
    if parsed is None:
        return False
    return parsed < CURRENT_NWB_SCHEMA_VERSION_THRESHOLD


def _threshold_label() -> str:
    return ".".join(str(part) for part in CURRENT_NWB_SCHEMA_VERSION_THRESHOLD)


def attach_reprocessing_candidate_status(
    graph: KnowledgeGraph,
    validation_path: Path = DEFAULT_VALIDATION_PATH,
) -> KnowledgeGraph:
    """Set `properties["reprocessing_candidate"]` on dataset nodes whose
    validated NWB file predates `CURRENT_NWB_SCHEMA_VERSION_THRESHOLD`.

    Immutable: returns a new KnowledgeGraph, does not mutate the input.
    A no-op if the validation artifact doesn't exist yet, or a given row
    isn't for a dataset already present in `graph` -- matches the house
    style of `attach_retraction_status` / `apply_file_validation_upgrades`.
    Only ever adds a property to an existing node; never creates new nodes
    or edges, so this cannot introduce a dangling-edge or graph_degree risk.
    """

    if not validation_path.exists():
        log.info(
            "reprocessing_candidate_builder: %s not found, skipping.", validation_path
        )
        return graph

    stale_by_dataset: dict[str, dict[str, Any]] = {}
    with validation_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("validator") != "dandi":
                continue
            raw_result = row.get("raw_result") or {}
            nwb_version = raw_result.get("nwb_version")
            if not nwb_version or not _is_stale(nwb_version):
                continue
            dataset_node_id = row.get("dataset_node_id")
            if dataset_node_id:
                stale_by_dataset[dataset_node_id] = {
                    "nwb_version": nwb_version,
                    "threshold": _threshold_label(),
                    "asset_path": raw_result.get("asset_path"),
                }

    if not stale_by_dataset:
        return graph

    updated_nodes = dict(graph.nodes)
    upgraded = 0
    for node_id, status in stale_by_dataset.items():
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        updated_nodes[node_id] = node.model_copy(
            update={"properties": {**node.properties, "reprocessing_candidate": status}}
        )
        upgraded += 1

    log.info("reprocessing_candidate_builder: %d datasets flagged.", upgraded)
    return graph.model_copy(update={"nodes": updated_nodes})
