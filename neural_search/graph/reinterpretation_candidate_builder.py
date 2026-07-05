"""Populate `dataset_reinterpretation_candidate` edges from real literature contradictions.

Data source found this session: `artifacts/literature/relationships/finding_edges.jsonl`
(200,000 rows, produced by `neural_search/literature/relationship_kg_builder.py`)
already contains 117,475+ real `contradicts` relationships between typed
findings extracted from different papers -- e.g. one paper reporting an
"increase" and another reporting a "decrease" for the same region/task
(`contradiction_subtype: "opposite_direction"`).

Joined with `artifacts/literature/paper_dataset_links.jsonl` (dataset -> paper
match), this answers: "this dataset's linked paper makes a claim that is
directly contradicted by another paper -- reanalyzing/reinterpreting this
dataset through the contradicting paper's framing is a genuine, literature-
motivated opportunity," which is a stronger, more specific claim than the
generic reanalysis-candidate heuristic.

Both edge endpoints are dataset nodes (same reasoning as
`reanalysis_bridge_builder.py`: no method-node reference, no dangling-edge
risk since dataset nodes always exist). Emitted symmetrically (both
directions) since a contradiction is informative in both directions, unlike
the precedent/candidate asymmetry of the reanalysis bridge.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neural_search.graph.reanalysis_bridge_builder import (
    _dataset_node_id,
    load_dataset_paper_matches,
)
from neural_search.graph.schema import GraphEvidence, KnowledgeGraph, KnowledgeGraphEdge
from neural_search.kg.schemas.evidence_tier import EvidenceTier

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
FINDING_EDGES_PATH = (
    PROJECT_ROOT / "artifacts" / "literature" / "relationships" / "finding_edges.jsonl"
)
BUILDER_NAME = "reinterpretation_candidate_builder"
BUILDER_VERSION = "v1.0.0"


def load_paper_contradictions(path: Path | str = FINDING_EDGES_PATH) -> list[dict[str, Any]]:
    """Strongest (highest-confidence) contradiction per unordered paper pair."""

    best_by_pair: dict[frozenset[str], dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("edge_type") != "contradicts":
                continue
            paper_a = row.get("paper_id_a")
            paper_b = row.get("paper_id_b")
            if not paper_a or not paper_b or paper_a == paper_b:
                continue
            pair = frozenset((paper_a, paper_b))
            current = best_by_pair.get(pair)
            if current is None or row.get("confidence", 0) > current.get("confidence", 0):
                best_by_pair[pair] = row
    return list(best_by_pair.values())


def _openalex_id_to_paper_node_key(dataset_paper_matches: dict[str, dict[str, Any]]) -> dict[str, list[tuple[str, float]]]:
    """paper_openalex_id -> [(dataset_record_id, match_confidence), ...]."""

    reverse: dict[str, list[tuple[str, float]]] = {}
    for dataset_record_id, match in dataset_paper_matches.items():
        reverse.setdefault(match["paper_openalex_id"], []).append(
            (dataset_record_id, match["confidence"])
        )
    return reverse


def build_reinterpretation_candidate_edges(
    graph: KnowledgeGraph,
    dataset_paper_matches: dict[str, dict[str, Any]] | None = None,
    contradictions: list[dict[str, Any]] | None = None,
) -> list[KnowledgeGraphEdge]:
    if dataset_paper_matches is None:
        dataset_paper_matches = load_dataset_paper_matches()
    if contradictions is None:
        contradictions = load_paper_contradictions()

    papers_to_datasets = _openalex_id_to_paper_node_key(dataset_paper_matches)

    edges: list[KnowledgeGraphEdge] = []
    seen_edge_ids: set[str] = set()
    for contradiction in contradictions:
        paper_a = contradiction["paper_id_a"].rsplit(":", 1)[-1]
        paper_b = contradiction["paper_id_b"].rsplit(":", 1)[-1]
        datasets_a = papers_to_datasets.get(paper_a, [])
        datasets_b = papers_to_datasets.get(paper_b, [])
        if not datasets_a or not datasets_b:
            continue  # honest gap: one or both sides have no matched corpus dataset

        contradiction_confidence = contradiction.get("confidence", 0.5)
        for record_a, match_conf_a in datasets_a:
            for record_b, match_conf_b in datasets_b:
                node_a = _dataset_node_id(record_a)
                node_b = _dataset_node_id(record_b)
                if node_a not in graph.nodes or node_b not in graph.nodes:
                    continue
                confidence = contradiction_confidence * match_conf_a * match_conf_b
                for source_node, target_node, own_direction, other_direction in (
                    (node_a, node_b, contradiction.get("direction_a"), contradiction.get("direction_b")),
                    (node_b, node_a, contradiction.get("direction_b"), contradiction.get("direction_a")),
                ):
                    edge_id = (
                        f"edge:dataset_reinterpretation:{source_node}:{target_node}:"
                        f"{paper_a}:{paper_b}"
                    )
                    if edge_id in seen_edge_ids:
                        continue
                    seen_edge_ids.add(edge_id)
                    evidence = GraphEvidence(
                        evidence_id=f"evidence:{BUILDER_NAME}:{source_node}:{target_node}",
                        source_type=BUILDER_NAME,
                        source_id=source_node,
                        evidence_text=(
                            f"Finding from linked paper contradicts a finding linked to "
                            f"{target_node} ({contradiction.get('contradiction_subtype')})"
                        ),
                        confidence=confidence,
                        extractor_name=BUILDER_NAME,
                        extractor_version=BUILDER_VERSION,
                    )
                    edges.append(
                        KnowledgeGraphEdge(
                            edge_id=edge_id,
                            source_node_id=source_node,
                            target_node_id=target_node,
                            edge_type="dataset_reinterpretation_candidate",
                            confidence=confidence,
                            evidence=[evidence],
                            properties={
                                "relationship_type": "claim_contradiction_reinterpretation",
                                "contradiction_subtype": contradiction.get("contradiction_subtype"),
                                "direction_self": own_direction,
                                "direction_other": other_direction,
                                "shared_regions": contradiction.get("shared_regions", []),
                                "shared_tasks": contradiction.get("shared_tasks", []),
                                "requires_human_review": True,
                                "evidence_tier": EvidenceTier.EVIDENCE_BACKED_BRIDGE.value,
                            },
                        )
                    )
    log.info(
        "Reinterpretation candidates: %d edges from %d contradictions",
        len(edges),
        len(contradictions),
    )
    return edges
