"""Generate reports/reanalysis_insight_report.md: the reanalysis-insight-synthesizer's output.

Unlike `generate_reanalysis_candidates_report.py` (aggregate counts by data
form/analysis family/technique) and `generate_methodology_coverage_report.py`
-style reports (raw coverage gaps), this synthesizes across both plus the
production graph's live edges to produce a ranked, actionable "what to reuse
next" list -- ordered by evidence strength and, for the heuristic-candidate
tier, by how genuinely unexplored the dataset looks (no linked papers at all,
not just no evidence for this specific method).

Also records the multi-source paper-linkage finding from
`neural_search/graph/reanalysis_bridge_builder.py` (measured null result: DOI-
resolving DataCite/Crossref/PubMed matches into the bridge builder adds only
8 matches and 0 new bridge edges) so the reason `dataset_reinterpretation_candidate`
is still 0 is on record precisely, not re-guessed at by a future run.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from neural_search.graph.reanalysis_bridge_builder import (
    load_dataset_paper_matches,
    load_dataset_paper_matches_multi_source,
)
from neural_search.graph.schema import KnowledgeGraph, read_graph_json

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
GRAPH_PATH = PROJECT_ROOT / "data" / "graph" / "neural_search_graph.real_corpus.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "reanalysis_insight_report.md"

TOP_N_BRIDGE = 15
TOP_N_UNEXPLORED_CANDIDATES = 15
HIGH_CONFIDENCE_THRESHOLD = 0.85


MAX_ROWS_PER_PRECEDENT = 2


def _top_bridge_opportunities(graph: KnowledgeGraph, top_n: int) -> list[dict[str, Any]]:
    """Highest-confidence dataset_reanalysis_bridge_dataset edges, one row per candidate dataset
    (its single strongest precedent), ranked by confidence descending, capped at
    `MAX_ROWS_PER_PRECEDENT` rows per precedent dataset.

    Confidence is `paper_confidence * method_confidence * sim_edge.confidence` -- a small,
    discrete set of input confidences, so many edges land on identical values, and a single
    highly-connected precedent (e.g. a widely-cited NeuroMorpho reconstruction) can otherwise
    fill the entire top-N with near-duplicate rows. The per-precedent cap trades a small amount
    of ranking purity for a list that actually surfaces variety.
    """

    best_by_candidate: dict[str, Any] = {}
    for edge in graph.edges.values():
        if edge.edge_type != "dataset_reanalysis_bridge_dataset":
            continue
        existing = best_by_candidate.get(edge.source_node_id)
        if existing is None or edge.confidence > existing.confidence:
            best_by_candidate[edge.source_node_id] = edge

    by_confidence = sorted(best_by_candidate.values(), key=lambda e: e.confidence, reverse=True)
    ranked: list[Any] = []
    rows_per_precedent: Counter[str] = Counter()
    for edge in by_confidence:
        if rows_per_precedent[edge.target_node_id] >= MAX_ROWS_PER_PRECEDENT:
            continue
        ranked.append(edge)
        rows_per_precedent[edge.target_node_id] += 1
        if len(ranked) >= top_n:
            break

    rows = []
    for edge in ranked:
        candidate_node = graph.nodes.get(edge.source_node_id)
        precedent_node = graph.nodes.get(edge.target_node_id)
        rows.append(
            {
                "candidate_label": candidate_node.label if candidate_node else edge.source_node_id,
                "candidate_id": edge.source_node_id,
                "precedent_label": precedent_node.label if precedent_node else edge.target_node_id,
                "method": edge.properties.get("method"),
                "confidence": round(edge.confidence, 3),
                "explanation": edge.properties.get("explanation"),
            }
        )
    return rows


MAX_ROWS_PER_TECHNIQUE = 3


def _unexplored_high_confidence_candidates(
    graph: KnowledgeGraph, corpus_linked_paper_datasets: set[str], top_n: int
) -> list[dict[str, Any]]:
    """High-confidence dataset_old_dataset_new_method_candidate edges on datasets with
    zero linked papers at all (not just no evidence for this specific method) -- the
    strongest proxy available today for 'genuinely unexplored, worth a fresh look.'
    Capped at `MAX_ROWS_PER_TECHNIQUE` rows per technique for the same diversity reason
    as `_top_bridge_opportunities` -- one technique/analysis-family pairing (e.g. FFT for
    time_frequency) can otherwise dominate every high-confidence tie."""

    best_by_candidate: dict[str, Any] = {}
    for edge in graph.edges.values():
        if edge.edge_type != "dataset_old_dataset_new_method_candidate":
            continue
        if edge.confidence < HIGH_CONFIDENCE_THRESHOLD:
            continue
        if edge.properties.get("has_linked_papers"):
            continue
        if edge.source_node_id in corpus_linked_paper_datasets:
            continue
        existing = best_by_candidate.get(edge.source_node_id)
        if existing is None or edge.confidence > existing.confidence:
            best_by_candidate[edge.source_node_id] = edge

    by_confidence = sorted(best_by_candidate.values(), key=lambda e: e.confidence, reverse=True)
    ranked: list[Any] = []
    rows_per_technique: Counter[str] = Counter()
    for edge in by_confidence:
        technique = edge.target_node_id
        if rows_per_technique[technique] >= MAX_ROWS_PER_TECHNIQUE:
            continue
        ranked.append(edge)
        rows_per_technique[technique] += 1
        if len(ranked) >= top_n:
            break

    rows = []
    for edge in ranked:
        node = graph.nodes.get(edge.source_node_id)
        rows.append(
            {
                "dataset_label": node.label if node else edge.source_node_id,
                "dataset_id": edge.source_node_id,
                "technique": edge.target_node_id.removeprefix("method:"),
                "analysis_family": edge.properties.get("analysis_family"),
                "confidence": round(edge.confidence, 3),
                "rationale": edge.properties.get("rationale"),
            }
        )
    return rows


def _paper_linked_dataset_node_ids() -> set[str]:
    """node:dataset:* ids for every dataset with a real OpenAlex paper match, so the
    unexplored-candidate ranking can exclude them even though the candidate edge's own
    `has_linked_papers` property only reflects the corpus's `linked_papers` field, a
    different (much sparser) signal -- see reanalysis_candidates.py's own docstring."""

    from neural_search.graph.reanalysis_bridge_builder import _dataset_node_id

    return {_dataset_node_id(record_id) for record_id in load_dataset_paper_matches()}


def _multi_source_linkage_finding() -> dict[str, Any]:
    old_matches = load_dataset_paper_matches()
    new_matches = load_dataset_paper_matches_multi_source()
    return {
        "openalex_only_matches": len(old_matches),
        "multi_source_resolved_matches": len(new_matches),
        "delta": len(new_matches) - len(old_matches),
    }


def build_report() -> str:
    logging.basicConfig(level=logging.INFO)
    log.info("Loading production graph from %s", GRAPH_PATH)
    graph = read_graph_json(GRAPH_PATH)
    log.info("Loaded %d nodes / %d edges", len(graph.nodes), len(graph.edges))

    bridge_rows = _top_bridge_opportunities(graph, TOP_N_BRIDGE)
    paper_linked_ids = _paper_linked_dataset_node_ids()
    unexplored_rows = _unexplored_high_confidence_candidates(
        graph, paper_linked_ids, TOP_N_UNEXPLORED_CANDIDATES
    )
    linkage_finding = _multi_source_linkage_finding()

    reinterp_count = sum(
        1 for e in graph.edges.values() if e.edge_type == "dataset_reinterpretation_candidate"
    )
    bridge_count = sum(
        1 for e in graph.edges.values() if e.edge_type == "dataset_reanalysis_bridge_dataset"
    )
    candidate_count = sum(
        1
        for e in graph.edges.values()
        if e.edge_type == "dataset_old_dataset_new_method_candidate"
    )

    lines = [
        "# Reanalysis Insight Report",
        "",
        "Synthesized by the `reanalysis-insight-synthesizer` agent "
        "(`artifacts/agents/playbooks/reanalysis_insight_synthesizer.md`) from the live "
        "production graph. Ranks existing reanalysis/reinterpretation signal instead of "
        "just counting it -- see `reports/reanalysis_candidates_report.md` and "
        "`reports/methodology_coverage_report.md` for the raw aggregate counts this builds on.",
        "",
        f"- `dataset_old_dataset_new_method_candidate` edges live: {candidate_count}",
        f"- `dataset_reanalysis_bridge_dataset` edges live: {bridge_count}",
        f"- `dataset_reinterpretation_candidate` edges live: {reinterp_count}",
        "",
        f"## Top {len(bridge_rows)} evidence-backed reuse opportunities "
        "(`dataset_reanalysis_bridge_dataset`, ranked by confidence)",
        "",
        "A similar dataset was actually analyzed with the named method, per a real linked "
        "paper; the candidate dataset has no such evidence yet, per the corpus's own "
        "linked-paper coverage.",
        "",
        "| Candidate dataset | Precedent dataset | Method | Confidence |",
        "|---|---|---|---|",
    ]
    for row in bridge_rows:
        lines.append(
            f"| {row['candidate_label']} | {row['precedent_label']} | {row['method']} | "
            f"{row['confidence']} |"
        )

    lines.extend(
        [
            "",
            f"## Top {len(unexplored_rows)} high-confidence, genuinely unexplored candidates "
            f"(confidence >= {HIGH_CONFIDENCE_THRESHOLD}, zero linked papers)",
            "",
            "Heuristic (`dataset_old_dataset_new_method_candidate`), but the strongest current "
            "proxy for 'nobody has published anything on this dataset yet, and its profile "
            "strongly matches this technique's requirements.' Still `requires_human_review=True`.",
            "",
            "| Dataset | Technique | Analysis family | Confidence |",
            "|---|---|---|---|",
        ]
    )
    for row in unexplored_rows:
        lines.append(
            f"| {row['dataset_label']} | {row['technique']} | {row['analysis_family']} | "
            f"{row['confidence']} |"
        )

    lines.extend(
        [
            "",
            "## Why `dataset_reinterpretation_candidate` is still 0, precisely",
            "",
            f"Growing paper-dataset linkage from 403 to 2,510 real matches (5 sources) did "
            "**not** change this. Re-measured 2026-07-04: resolving DataCite/Crossref/PubMed "
            f"matches to an OpenAlex ID by shared DOI finds only "
            f"{linkage_finding['delta']} additional usable matches "
            f"({linkage_finding['openalex_only_matches']} -> "
            f"{linkage_finding['multi_source_resolved_matches']}), and adds zero new "
            "`dataset_reanalysis_bridge_dataset` edges (2,517 either way). The bottleneck is "
            "not paper-dataset linkage breadth; it's that `artifacts/ner/ner_kg.jsonl`'s "
            "method-mention extraction has only ever run against OpenAlex-ingested paper text. "
            "Closing this needs NER extraction against Crossref/PubMed/DataCite paper records "
            "directly -- a real, scoped, not-yet-attempted next step, not a re-run of existing "
            "linkers.",
            "",
            "## Cross-reference: methodology registry gaps limiting candidate generation",
            "",
            "See `reports/methodology_coverage_report.md` for the full list. As of the last "
            "coverage run: 10/27 analysis families have no technique mapping yet (including "
            "`biomarker_discovery`, `cell_type_mapping`, `phenotyping`, `spatial_mapping`), and "
            "`intracellular_ephys`/`molecular` data forms have zero candidate-eligible analysis "
            "families at all -- any dataset in those data forms cannot generate a candidate "
            "edge regardless of its actual analysis potential, until the registry is extended.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    report = build_report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
