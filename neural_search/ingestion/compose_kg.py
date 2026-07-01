"""Compose all KG builder layers into a single merged KnowledgeGraph.

Runs every builder that has local data available, merges their nodes and
edges (deduplicating by ID), and writes the result to:
    artifacts/kg/composed_kg.jsonl

Usage:
    python -m neural_search.ingestion.compose_kg [--include-neurosynth] [--include-ner]

The composed KG is the authoritative graph used by the search API and
frontend visualisations.

Edge counts by layer (approximate, depends on data present):
    concept     :  ~217  concept hierarchy + method links
    disorder    :  ~228  disorder→circuit + biomarker edges
    allen       :   ~34  region_projects_to curated mouse circuits
    paradigm    :  ~400  paradigm→circuit + topic edges
    oscillation :  ~300  region→oscillation edges
    homology    :  ~200  cross-species region homology
    hcp         :  ~150  structural connectivity (FA-weighted)
    neurosynth  : ~5000  topic→region activation edges (if pickle exists)
    ner         :  ~10K+  paper→region/disorder/method (if enabled)
    citation    : ~500K  paper_cites_paper (if enabled)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from neural_search.graph.schema import (
    KnowledgeGraph,
    KnowledgeGraphNode,
    KnowledgeGraphEdge,
    write_graph_jsonl,
)

log = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent.parent / "artifacts" / "kg"
OUTPUT_PATH = OUTPUT_DIR / "composed_kg.jsonl"


def _merge_into(
    base_nodes: dict[str, KnowledgeGraphNode],
    base_edges: dict[str, KnowledgeGraphEdge],
    layer_kg: KnowledgeGraph,
    layer_name: str,
) -> tuple[int, int]:
    """Merge layer KG into base dicts. Returns (new_nodes, new_edges) counts."""
    new_nodes = 0
    new_edges = 0
    for node_id, node in layer_kg.nodes.items():
        if node_id not in base_nodes:
            base_nodes[node_id] = node
            new_nodes += 1
    for edge_id, edge in layer_kg.edges.items():
        if edge_id not in base_edges:
            base_edges[edge_id] = edge
            new_edges += 1
    log.info(
        "Layer %-20s  +%4d nodes  +%5d edges  (total: %d nodes, %d edges)",
        layer_name, new_nodes, new_edges, len(base_nodes), len(base_edges),
    )
    return new_nodes, new_edges


def compose_kg(
    include_neurosynth: bool = True,
    include_ner: bool = False,
    include_citations: bool = False,
    ner_max_papers: int = 50_000,
    citation_max_edges: int = 500_000,
) -> KnowledgeGraph:

    all_nodes: dict[str, KnowledgeGraphNode] = {}
    all_edges: dict[str, KnowledgeGraphEdge] = {}

    # ── Core static layers (always run) ──────────────────────────────────────
    layers: list[tuple[str, Any]] = []

    try:
        from neural_search.ingestion.concept_builder import build_concept_kg
        layers.append(("concept", build_concept_kg))
    except Exception as exc:
        log.warning("concept_builder unavailable: %s", exc)

    try:
        from neural_search.ingestion.disorder_builder import build_disorder_kg
        layers.append(("disorder", build_disorder_kg))
    except Exception as exc:
        log.warning("disorder_builder unavailable: %s", exc)

    try:
        from neural_search.ingestion.allen_connectivity_builder import build_allen_connectivity_kg
        layers.append(("allen_connectivity", build_allen_connectivity_kg))
    except Exception as exc:
        log.warning("allen_connectivity_builder unavailable: %s", exc)

    try:
        from neural_search.ingestion.paradigm_builder import build_paradigm_kg
        layers.append(("paradigm", build_paradigm_kg))
    except Exception as exc:
        log.warning("paradigm_builder unavailable: %s", exc)

    try:
        from neural_search.ingestion.oscillation_builder import build_oscillation_kg
        layers.append(("oscillation", build_oscillation_kg))
    except Exception as exc:
        log.warning("oscillation_builder unavailable: %s", exc)

    try:
        from neural_search.ingestion.species_homology_builder import build_species_homology_kg
        layers.append(("species_homology", build_species_homology_kg))
    except Exception as exc:
        log.warning("species_homology_builder unavailable: %s", exc)

    try:
        from neural_search.ingestion.hcp_connectivity import build_hcp_kg
        layers.append(("hcp_connectivity", build_hcp_kg))
    except Exception as exc:
        log.warning("hcp_connectivity unavailable: %s", exc)

    try:
        from neural_search.ingestion.scholarpedia_builder import build_scholarpedia_kg
        layers.append(("scholarpedia", build_scholarpedia_kg))
    except Exception as exc:
        log.warning("scholarpedia_builder unavailable: %s", exc)

    # Run all core layers
    for name, builder_fn in layers:
        try:
            log.info("Building layer: %s…", name)
            kg = builder_fn()
            _merge_into(all_nodes, all_edges, kg, name)
        except Exception as exc:
            log.warning("Layer %s failed: %s", name, exc)

    # ── Optional: NeuroSynth (requires pickle) ────────────────────────────────
    if include_neurosynth:
        try:
            from neural_search.ingestion.neurosynth_builder import build_neurosynth_kg
            log.info("Building layer: neurosynth…")
            kg = build_neurosynth_kg()
            _merge_into(all_nodes, all_edges, kg, "neurosynth")
        except Exception as exc:
            log.warning("neurosynth_builder failed: %s", exc)

    # ── Optional: NER (slow unless cache exists) ─────────────────────────────
    if include_ner:
        try:
            from neural_search.ingestion.ner_builder import (
                build_ner_kg, load_cached_ner_kg, save_ner_kg, NER_ARTIFACT_PATH,
            )
            cached = load_cached_ner_kg()
            if cached is not None:
                log.info("NER: using cached artifact (%d nodes, %d edges)", len(cached.nodes), len(cached.edges))
                _merge_into(all_nodes, all_edges, cached, "ner")
            else:
                log.info("NER: no cache found, running extraction (max_papers=%d)…", ner_max_papers)
                kg = build_ner_kg(use_spacy=False, max_papers=ner_max_papers)
                save_ner_kg(kg)
                log.info("NER: cache saved to %s", NER_ARTIFACT_PATH)
                _merge_into(all_nodes, all_edges, kg, "ner")
        except Exception as exc:
            log.warning("ner_builder failed: %s", exc)

    # ── Optional: Citation graph (large) ─────────────────────────────────────
    if include_citations:
        try:
            from neural_search.ingestion.citation_builder import build_citation_kg
            log.info("Building layer: citations (max_edges=%d)…", citation_max_edges)
            kg = build_citation_kg(max_edges=citation_max_edges)
            _merge_into(all_nodes, all_edges, kg, "citations")
        except Exception as exc:
            log.warning("citation_builder failed: %s", exc)

    # ── Build final graph (lenient validation — partial layers OK) ────────────
    # We build without cross-layer edge validation since edges may reference
    # nodes defined in another layer that isn't included in this run.
    composed = KnowledgeGraph(nodes=all_nodes, edges=all_edges)

    log.info(
        "Composed KG: %d total nodes, %d total edges",
        len(composed.nodes), len(composed.edges),
    )
    return composed


def _summarise(kg: KnowledgeGraph) -> None:
    from collections import Counter
    node_types = Counter(n.node_type for n in kg.nodes.values())
    edge_types = Counter(e.edge_type for e in kg.edges.values())
    print(f"\n=== Composed KG Summary ===")
    print(f"Total nodes: {len(kg.nodes):,}")
    for nt, count in node_types.most_common():
        print(f"  {nt:<30} {count:>6,}")
    print(f"\nTotal edges: {len(kg.edges):,}")
    for et, count in edge_types.most_common():
        print(f"  {et:<45} {count:>6,}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-neurosynth", action="store_true", default=True,
                        help="Include NeuroSynth layer (requires pickle, default: on)")
    parser.add_argument("--skip-neurosynth", action="store_true",
                        help="Skip NeuroSynth even if pickle exists")
    parser.add_argument("--include-ner", action="store_true",
                        help="Run NER over corpus (slow, ~20 min for 255K papers)")
    parser.add_argument("--include-citations", action="store_true",
                        help="Include citation graph (500K edges)")
    parser.add_argument("--ner-max-papers", type=int, default=50_000)
    parser.add_argument("--citation-max-edges", type=int, default=500_000)
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Output JSONL path")
    args = parser.parse_args()

    kg = compose_kg(
        include_neurosynth=not args.skip_neurosynth,
        include_ner=args.include_ner,
        include_citations=args.include_citations,
        ner_max_papers=args.ner_max_papers,
        citation_max_edges=args.citation_max_edges,
    )

    _summarise(kg)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_graph_jsonl(kg, out_path)
    size_mb = out_path.stat().st_size / (1024 ** 2)
    print(f"\nWritten -> {out_path}  ({size_mb:.1f} MB)")
