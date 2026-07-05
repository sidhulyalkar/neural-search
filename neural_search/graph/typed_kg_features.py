"""Isolated typed-relationship/claim-layer scoring feature for retrieval.

Added 2026-06-23 to answer a question none of the existing ablation rungs
could: does the typed finding-relationship layer (Phases 0-6b — supports/
contradicts edges, qualified consensus tiers) move retrieval quality, or is
it only useful for browsing/explanation? ``hybrid_graph`` mixes this signal
in with linked-paper counts, affordances, ontology-dimension matches, and
dataset-similarity edges, so a gain or loss there can't be attributed to the
typed layer specifically.

This module deliberately does NOT touch ``graph_context_score`` or the full
``KnowledgeGraph`` — the typed finding/claim edges aren't merged into
``data/graph/neural_search_graph.real_corpus.json`` yet (that graph predates
Phase 0-6b). Instead it joins three small, already-existing JSONL artifacts
directly:

  dataset_record_id --(paper_dataset_links.jsonl)--> paper_id
  paper_id          --(finding_edges.jsonl)--------> supports/contradicts edges
  brain_regions     --(consensus_summaries_qualified.jsonl)--> qualified consensus

so the "is this dataset's literature context typed-relationship-rich"
question can be answered without a graph rebuild.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Bounds chosen to match graph_context_score's 0.0-0.25 range (search_features.py)
# so typed_kg and hybrid_graph are comparable in magnitude, not just in direction.
MAX_TYPED_KG_SCORE = 0.25
WEIGHT_SUPPORTS = 0.03
WEIGHT_CONTRADICTS = 0.02
WEIGHT_QUALIFIED_BONUS = 0.05
MAX_EDGES_COUNTED = 5


@dataclass(frozen=True)
class TypedKGIndex:
    """Precomputed lookup tables for typed_kg_score. Build once per ablation run."""

    dataset_to_papers: dict[str, list[str]] = field(default_factory=dict)
    paper_to_edges: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    region_to_qualified_consensus: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def from_files(
        cls,
        paper_dataset_links_path: Path,
        finding_edges_path: Path,
        qualified_consensus_path: Path | None = None,
    ) -> TypedKGIndex:
        return cls(
            dataset_to_papers=_load_dataset_to_papers(paper_dataset_links_path),
            paper_to_edges=_load_paper_to_edges(finding_edges_path),
            region_to_qualified_consensus=(
                _load_region_to_qualified_consensus(qualified_consensus_path)
                if qualified_consensus_path is not None
                else {}
            ),
        )

    @property
    def has_qualified_consensus(self) -> bool:
        return bool(self.region_to_qualified_consensus)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_dataset_to_papers(path: Path) -> dict[str, list[str]]:
    """dataset_record_id -> [paper_id] (``paper:openalex:<id>`` form), confidently linked only."""
    index: dict[str, list[str]] = {}
    for row in _read_jsonl(path):
        if row.get("match_method") == "not_found":
            continue
        confidence = float(row.get("confidence") or 0.0)
        openalex_id = str(row.get("paper_openalex_id") or "").strip()
        dataset_id = str(row.get("dataset_record_id") or "").strip()
        if confidence <= 0.0 or not openalex_id or not dataset_id:
            continue
        paper_id = f"paper:openalex:{openalex_id}"
        index.setdefault(dataset_id, []).append(paper_id)
    return index


def _load_paper_to_edges(path: Path) -> dict[str, list[dict[str, Any]]]:
    """paper_id -> list of finding-relationship edges (supports/contradicts) touching it."""
    index: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        for key in ("paper_id_a", "paper_id_b"):
            paper_id = row.get(key)
            if paper_id:
                index.setdefault(str(paper_id), []).append(row)
    return index


def _load_region_to_qualified_consensus(path: Path) -> dict[str, list[dict[str, Any]]]:
    """region (lowercased) -> qualified ConsensusRecord dicts mentioning it, n_papers>=2 only."""
    index: dict[str, list[dict[str, Any]]] = {}
    for row in _read_jsonl(path):
        if int(row.get("n_papers", 0) or 0) < 2:
            continue
        region = str(row.get("region") or "").strip().lower()
        if region:
            index.setdefault(region, []).append(row)
    return index


def typed_kg_score(
    dataset_id: str,
    index: TypedKGIndex,
    *,
    record: dict[str, Any] | None = None,
    qualified: bool = False,
) -> float:
    """Score how richly a dataset's literature context is characterized by the
    typed finding-relationship layer. Agnostic to query context, deliberately:
    this isolates "does typed evidence density help retrieval at all" before
    layering in query-specific matching (which graph_context_score already does
    for the aggregate graph signal).

    Returns 0.0 for datasets with no confidently-linked paper or no typed edges —
    this is the common case (only 393/7,171 corpus datasets have a confident
    paper link as of 2026-06-23), so most candidates score 0.0 on this rung,
    same as they score 0.0 on graph_context_score's linked-paper term today.
    """
    papers = index.dataset_to_papers.get(dataset_id, [])
    if not papers:
        return 0.0

    n_supports = 0
    n_contradicts = 0
    for paper_id in papers:
        for edge in index.paper_to_edges.get(paper_id, []):
            if edge.get("edge_type") == "supports":
                n_supports += 1
            elif edge.get("edge_type") == "contradicts":
                n_contradicts += 1

    score = (
        min(n_supports, MAX_EDGES_COUNTED) * WEIGHT_SUPPORTS
        + min(n_contradicts, MAX_EDGES_COUNTED) * WEIGHT_CONTRADICTS
    )

    if qualified and record is not None and index.has_qualified_consensus:
        regions = {str(r).strip().lower() for r in (record.get("brain_regions") or []) if str(r).strip()}
        if regions & set(index.region_to_qualified_consensus.keys()):
            score += WEIGHT_QUALIFIED_BONUS

    return round(min(score, MAX_TYPED_KG_SCORE), 4)
