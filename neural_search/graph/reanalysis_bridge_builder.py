"""Populate `dataset_reanalysis_bridge_dataset` edges from real paper evidence.

Unlike `dataset_old_dataset_new_method_candidate` (a heuristic: "this dataset's
profile matches a technique's requirements"), this edge type answers a
stronger question: "a similar dataset was actually analyzed with method X, per
a real paper; this dataset hasn't been (as far as we can tell) — it could be."

Built from two existing artifacts, joined for the first time for this purpose:

- `artifacts/literature/paper_dataset_links.jsonl` — dataset -> OpenAlex paper
  match (only `doi_exact`/`title_fuzzy_local` rows are real matches; most rows
  are `not_found`).
- `artifacts/ner/ner_kg.jsonl` — `paper_uses_method` edges extracted by
  `neural_search.ingestion.ner_builder` (`paper:openalex:{id} -> method:{id}`).

As of 2026-07-01 this join yields real evidence for 82 of 7,171 datasets — a
small, honest reflection of current dataset-paper linkage coverage
(393/7,171 datasets have any OpenAlex match at all), not a bug to paper over.

Both edge endpoints are `dataset` nodes (the `method` is a string property,
not a graph edge to a method node — confirmed against
`neural_search.graph.search_features._relationship_summaries`, which reads
`properties["method"]`/`["relationship_type"]`/`["explanation"]`), so unlike
the Methodology Registry's edges there is no cross-builder node-id
reconciliation needed and no dangling-edge risk: dataset nodes always exist.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from neural_search.graph.schema import GraphEvidence, KnowledgeGraph, KnowledgeGraphEdge
from neural_search.kg.schemas.evidence_tier import EvidenceTier

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
PAPER_DATASET_LINKS_PATH = PROJECT_ROOT / "artifacts" / "literature" / "paper_dataset_links.jsonl"
NER_KG_PATH = PROJECT_ROOT / "artifacts" / "ner" / "ner_kg.jsonl"

REAL_MATCH_METHODS = {"doi_exact", "title_fuzzy_local"}
BUILDER_NAME = "reanalysis_bridge_builder"
BUILDER_VERSION = "v1.0.0"


def _method_label(method_id: str) -> str:
    return method_id.replace("_", " ").title()


def _dataset_node_id(dataset_record_id: str) -> str:
    source, _, source_id = dataset_record_id.partition(":")
    return f"node:dataset:{source}:{source_id}"


def load_dataset_paper_matches(path: Path | str = PAPER_DATASET_LINKS_PATH) -> dict[str, dict[str, Any]]:
    """dataset_record_id -> {paper_openalex_id, confidence} for real matches only."""

    matches: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("match_method") not in REAL_MATCH_METHODS:
                continue
            paper_id = row.get("paper_openalex_id")
            if not paper_id:
                continue
            matches[row["dataset_record_id"]] = {
                "paper_openalex_id": paper_id,
                "confidence": row.get("confidence", 0.5),
            }
    return matches


OTHER_SOURCE_FILES: dict[str, tuple[Path, set[str]]] = {
    "datacite": (
        PROJECT_ROOT / "artifacts" / "literature" / "paper_dataset_links.datacite.jsonl",
        {"datacite_related_identifier"},
    ),
    "crossref": (
        PROJECT_ROOT / "artifacts" / "literature" / "paper_dataset_links.crossref.jsonl",
        {"crossref_doi_exact", "crossref_title_fuzzy"},
    ),
    "pubmed": (
        PROJECT_ROOT / "artifacts" / "literature" / "paper_dataset_links.pubmed.jsonl",
        {"pubmed_doi_exact", "pubmed_title_fuzzy", "biorxiv_doi_exact"},
    ),
}


def _build_doi_to_openalex_index(path: Path | str = PAPER_DATASET_LINKS_PATH) -> dict[str, str]:
    """DOI -> OpenAlex ID, built only from the OpenAlex link file (the only source that
    carries a real OpenAlex ID). Used to resolve DataCite/Crossref/PubMed matches (which
    record `paper_doi` but leave `paper_openalex_id` empty) back to an OpenAlex ID, so they
    can still join against `ner_kg.jsonl`'s OpenAlex-keyed `paper_uses_method` mentions.
    """

    index: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            doi = row.get("paper_doi")
            openalex_id = row.get("paper_openalex_id")
            if doi and openalex_id:
                index[doi.lower()] = openalex_id
    return index


def load_dataset_paper_matches_multi_source(
    openalex_path: Path | str = PAPER_DATASET_LINKS_PATH,
    other_sources: dict[str, tuple[Path, set[str]]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Real dataset->paper matches combined across OpenAlex, DataCite, Crossref, and PubMed.

    `load_dataset_paper_matches` (OpenAlex-only) undercounts real linkage: as of 2026-07-04
    the corpus has 2,510/7,171 real paper links across 5 sources, but only 403 of those are
    OpenAlex matches, and the reanalysis-bridge/reinterpretation-candidate builders only ever
    read the OpenAlex file. A DataCite/Crossref/PubMed match is only *usable* here if it
    resolves to an OpenAlex ID (needed to join against `ner_kg.jsonl`'s method mentions) --
    resolved via the DOI both records share, since the OpenAlex file itself carries `paper_doi`
    alongside `paper_openalex_id`.

    **Measured result (2026-07-04, see `reports/reanalysis_insight_report.md`): this resolves
    only 8 additional matches (401 vs. 393) and adds zero new bridge edges (2,517 either way).**
    The other ~2,100 DataCite/Crossref/PubMed matches are for papers OpenAlex never ingested at
    all, so their DOIs don't appear in the OpenAlex file regardless of dataset overlap. The real
    bottleneck is not paper-dataset linkage breadth -- it's that `ner_kg.jsonl`'s method-mention
    extraction (`neural_search/ingestion/ner_builder.py`) has only ever run against
    OpenAlex-ingested paper text. Closing this gap for real needs NER extraction against
    Crossref/PubMed/DataCite paper records directly, not a smarter join. Kept as a small,
    correct, tested utility for that future work rather than reverted, since the null result
    itself is worth having on record precisely (not guessed at).
    """

    if other_sources is None:
        other_sources = OTHER_SOURCE_FILES

    matches = load_dataset_paper_matches(openalex_path)
    doi_to_openalex = _build_doi_to_openalex_index(openalex_path)

    for source_name, (path, real_methods) in other_sources.items():
        if not Path(path).exists():
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("match_method") not in real_methods:
                    continue
                dataset_record_id = row.get("dataset_record_id")
                doi = (row.get("paper_doi") or "").lower()
                if not dataset_record_id or not doi:
                    continue
                openalex_id = doi_to_openalex.get(doi)
                if not openalex_id:
                    continue  # real link, but not resolvable to an OpenAlex ID yet
                confidence = row.get("confidence", 0.5)
                existing = matches.get(dataset_record_id)
                if existing is not None and existing["confidence"] >= confidence:
                    continue
                matches[dataset_record_id] = {
                    "paper_openalex_id": openalex_id,
                    "confidence": confidence,
                    "resolved_via": source_name,
                }
    return matches


def load_paper_method_mentions(path: Path | str = NER_KG_PATH) -> dict[str, dict[str, float]]:
    """paper_openalex_id -> {method_id: confidence} from paper_uses_method edges."""

    mentions: dict[str, dict[str, float]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") != "edge":
                continue
            edge = row["edge"]
            if edge.get("edge_type") != "paper_uses_method":
                continue
            paper_id = edge["source_node_id"].rsplit(":", 1)[-1]
            method_id = edge["target_node_id"].split(":", 1)[-1]
            mentions.setdefault(paper_id, {})[method_id] = edge.get("confidence", 0.72)
    return mentions


def load_dataset_method_evidence(
    paper_links_path: Path | str = PAPER_DATASET_LINKS_PATH,
    ner_kg_path: Path | str = NER_KG_PATH,
    *,
    dataset_papers: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """dataset_record_id -> {method_id: {paper_openalex_id, paper_confidence, method_confidence}}.

    Pass `dataset_papers=load_dataset_paper_matches_multi_source()` to draw matches from all
    5 literature sources instead of just OpenAlex (the default via `paper_links_path`).
    """

    if dataset_papers is None:
        dataset_papers = load_dataset_paper_matches(paper_links_path)
    paper_methods = load_paper_method_mentions(ner_kg_path)

    evidence: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_record_id, paper_match in dataset_papers.items():
        paper_id = paper_match["paper_openalex_id"]
        methods = paper_methods.get(paper_id)
        if not methods:
            continue
        evidence[dataset_record_id] = {
            method_id: {
                "paper_openalex_id": paper_id,
                "paper_confidence": paper_match["confidence"],
                "method_confidence": method_confidence,
            }
            for method_id, method_confidence in methods.items()
        }
    return evidence


def _similarity_neighbors(
    graph: KnowledgeGraph, node_id: str
) -> list[tuple[str, KnowledgeGraphEdge]]:
    """(neighbor_node_id, edge) pairs via existing dataset_similar_to_dataset edges."""

    neighbors: list[tuple[str, KnowledgeGraphEdge]] = []
    for edge in graph.edges.values():
        if edge.edge_type != "dataset_similar_to_dataset":
            continue
        if edge.source_node_id == node_id:
            neighbors.append((edge.target_node_id, edge))
        elif edge.target_node_id == node_id:
            neighbors.append((edge.source_node_id, edge))
    return neighbors


def build_reanalysis_bridge_edges(
    graph: KnowledgeGraph,
    evidence: dict[str, dict[str, dict[str, Any]]] | None = None,
) -> list[KnowledgeGraphEdge]:
    """Build dataset_reanalysis_bridge_dataset edges: candidate (no evidence) -> precedent (has evidence)."""

    if evidence is None:
        evidence = load_dataset_method_evidence()

    # node_id (of the precedent, e.g. node:dataset:dandi:000166) -> its evidence dict
    evidence_by_node_id = {
        _dataset_node_id(record_id): methods for record_id, methods in evidence.items()
    }

    edges: list[KnowledgeGraphEdge] = []
    seen_edge_ids: set[str] = set()
    for precedent_node_id, methods in evidence_by_node_id.items():
        if precedent_node_id not in graph.nodes:
            continue
        for candidate_node_id, sim_edge in _similarity_neighbors(graph, precedent_node_id):
            if candidate_node_id not in graph.nodes:
                continue
            candidate_methods = evidence_by_node_id.get(candidate_node_id, {})
            for method_id, method_evidence in methods.items():
                if method_id in candidate_methods:
                    continue  # candidate already has evidence for this method too
                edge_id = (
                    f"edge:dataset_reanalysis_bridge:{candidate_node_id}:"
                    f"{method_id}:{precedent_node_id}"
                )
                if edge_id in seen_edge_ids:
                    continue
                seen_edge_ids.add(edge_id)

                confidence = (
                    method_evidence["paper_confidence"]
                    * method_evidence["method_confidence"]
                    * sim_edge.confidence
                )
                method_label = _method_label(method_id)
                gevidence = GraphEvidence(
                    evidence_id=f"evidence:{BUILDER_NAME}:{candidate_node_id}:{method_id}",
                    source_type=BUILDER_NAME,
                    source_id=precedent_node_id,
                    evidence_text=(
                        f"OpenAlex paper {method_evidence['paper_openalex_id']} "
                        f"links {precedent_node_id} to {method_label}"
                    ),
                    confidence=confidence,
                    extractor_name=BUILDER_NAME,
                    extractor_version=BUILDER_VERSION,
                )
                edges.append(
                    KnowledgeGraphEdge(
                        edge_id=edge_id,
                        source_node_id=candidate_node_id,
                        target_node_id=precedent_node_id,
                        edge_type="dataset_reanalysis_bridge_dataset",
                        confidence=confidence,
                        evidence=[gevidence],
                        properties={
                            "relationship_type": "method_reanalysis_bridge",
                            "method": method_label,
                            "explanation": (
                                f"A similar dataset was analyzed with {method_label} "
                                f"(per OpenAlex {method_evidence['paper_openalex_id']}); "
                                f"this dataset has no such evidence and shares "
                                f"{sim_edge.properties.get('cross_type', 'similarity')}."
                            ),
                            "precedent_paper_openalex_id": method_evidence["paper_openalex_id"],
                            "similarity_basis": sim_edge.properties.get("cross_type"),
                            "requires_human_review": True,
                            "evidence_tier": EvidenceTier.EVIDENCE_BACKED_BRIDGE.value,
                        },
                    )
                )
    log.info(
        "Reanalysis bridge: %d edges from %d precedent datasets",
        len(edges),
        len(evidence_by_node_id),
    )
    return edges
