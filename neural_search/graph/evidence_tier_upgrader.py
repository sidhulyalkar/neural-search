"""Apply live file-validation results as evidence_tier upgrades on graph edges.

Reads `artifacts/validation/top_suggestions_file_validation.jsonl` (produced
by `scripts/validate_top_reanalysis_suggestions.py`) and upgrades matching
`dataset_old_dataset_new_method_candidate` / `dataset_reanalysis_bridge_dataset`
edges from their original tier to `file_validated`, per
`neural_search.kg.schemas.evidence_tier`.

This is applied as a build-time layer (see `scripts/build_real_corpus_graph.py`)
rather than a one-off patch to the graph file, so the upgrade survives a full
graph rebuild — the validation JSONL is the durable artifact, not the graph.
Never downgrades: if an edge's existing tier is somehow already stronger than
`file_validated`, it is left unchanged (see `evidence_tier.upgrade_tier`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neural_search.graph.schema import KnowledgeGraphEdge
from neural_search.kg.schemas.evidence_tier import EvidenceTier, upgrade_tier

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DEFAULT_VALIDATION_PATH = (
    PROJECT_ROOT / "artifacts" / "validation" / "top_suggestions_file_validation.jsonl"
)


def load_confirmed_validations(
    path: Path | str = DEFAULT_VALIDATION_PATH,
) -> dict[tuple[str, str], dict[str, Any]]:
    """(dataset_node_id, analysis_family) -> validation row, confirmed rows only."""

    path = Path(path)
    if not path.exists():
        return {}
    confirmed: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("confirmed"):
                continue
            key = (row["dataset_node_id"], row.get("analysis_family"))
            confirmed[key] = row
    return confirmed


def apply_file_validation_upgrades(
    edges: dict[str, KnowledgeGraphEdge],
    validation_path: Path | str = DEFAULT_VALIDATION_PATH,
) -> tuple[dict[str, KnowledgeGraphEdge], int]:
    """Return a new edges dict with confirmed suggestions upgraded to file_validated.

    Immutable: does not mutate the input dict or its edge objects.
    """

    confirmed = load_confirmed_validations(validation_path)
    if not confirmed:
        return edges, 0

    upgraded_edges = dict(edges)
    upgrade_count = 0
    for edge_id, edge in edges.items():
        if edge.edge_type not in (
            "dataset_old_dataset_new_method_candidate",
            "dataset_reanalysis_bridge_dataset",
        ):
            continue
        analysis_family = edge.properties.get("analysis_family") or edge.properties.get("method")
        key = (edge.source_node_id, analysis_family)
        row = confirmed.get(key)
        if row is None:
            continue

        current_tier = edge.properties.get("evidence_tier", EvidenceTier.HEURISTIC_CANDIDATE.value)
        new_tier = upgrade_tier(current_tier, EvidenceTier.FILE_VALIDATED)
        if new_tier.value == current_tier:
            continue  # already at or above file_validated

        new_properties = dict(edge.properties)
        new_properties["evidence_tier"] = new_tier.value
        new_properties["file_validation"] = {
            "validator": row["validator"],
            "source_id": row["source_id"],
        }
        upgraded_edges[edge_id] = edge.model_copy(update={"properties": new_properties})
        upgrade_count += 1

    log.info("Evidence tier upgrades applied: %d", upgrade_count)
    return upgraded_edges, upgrade_count
