#!/usr/bin/env python3
"""Generate docs/whitepaper/generated_stats.tex from the artifact manifest.

Run `scripts/build_artifact_manifest.py` first (or via `main()` here, which
calls it automatically) so the manifest reflects the current files, then this
script emits `\\newcommand` macros the whitepaper's `\\input{generated_stats.tex}`
pulls in. This is what makes whitepaper numbers regenerate instead of
drifting stale — see the comment above that `\\input` in
docs/whitepaper/neural_search_whitepaper.tex for the incident that motivated
this (abstract/conclusion/appendix citing a two-reconnection-sessions-old
graph size).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = PROJECT_ROOT / "reports" / "eval" / "current_artifact_manifest.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "whitepaper" / "generated_stats.tex"


def _fmt(n: int | float) -> str:
    if isinstance(n, float):
        return f"{n:,.4f}".rstrip("0").rstrip(".")
    return f"{n:,}"


def _macro(name: str, value: Any) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def build_macros(manifest: dict[str, Any]) -> list[str]:
    corpus = manifest.get("corpus", {})
    graph = manifest.get("knowledge_graph", {})
    node_counts = graph.get("node_type_counts", {})
    edge_counts = graph.get("edge_type_counts", {})
    display_edge_counts = graph.get("display_edge_counts", {})
    paper_links = manifest.get("paper_dataset_links", {})
    match_methods = paper_links.get("match_method_counts", {})
    qrels = manifest.get("qrels", {})
    reanalysis = manifest.get("reanalysis_edges", {})
    ablation = manifest.get("ablation_benchmark", {})
    rungs = ablation.get("rungs", {})

    cross_dataset_edges = display_edge_counts.get(
        "same_region_cross_modality", 0
    ) + display_edge_counts.get("same_task_cross_species", 0)
    base_dataset_concept_edges = (
        edge_counts.get("dataset_has_modality", 0)
        + edge_counts.get("dataset_records_region", 0)
        + edge_counts.get("dataset_has_species", 0)
        + edge_counts.get("dataset_has_task", 0)
    )

    def ndcg(rung: str) -> str:
        value = rungs.get(rung, {}).get("metrics", {}).get("ndcg@10")
        return _fmt(value) if value is not None else "N/A"

    macros = [
        "% Corpus",
        _macro("CorpusRowCount", _fmt(corpus.get("row_count", "N/A"))),
        _macro("CorpusUniqueSourceIds", _fmt(corpus.get("unique_source_ids", "N/A"))),
        "% Knowledge graph totals",
        _macro("KGNodeCount", _fmt(graph.get("total_nodes", "N/A"))),
        _macro("KGEdgeCount", _fmt(graph.get("total_edges", "N/A"))),
        _macro("StubNodeCount", _fmt(graph.get("stub_node_count", "N/A"))),
        "% Knowledge graph node breakdown",
        _macro("DatasetNodeCount", _fmt(node_counts.get("dataset", 0))),
        _macro("BrainRegionNodeCount", _fmt(node_counts.get("brain_region", 0))),
        _macro("TaskNodeCount", _fmt(node_counts.get("task", 0))),
        _macro("ModalityNodeCount", _fmt(node_counts.get("modality", 0))),
        _macro("SpeciesNodeCount", _fmt(node_counts.get("species", 0))),
        _macro("MethodNodeCount", _fmt(node_counts.get("method", 0))),
        _macro("DisorderNodeCount", _fmt(node_counts.get("disorder", 0))),
        _macro("ParadigmNodeCount", _fmt(node_counts.get("paradigm", 0))),
        _macro("OscillationNodeCount", _fmt(node_counts.get("oscillation", 0))),
        "% Knowledge graph edge breakdown",
        _macro("BaseDatasetConceptEdgeCount", _fmt(base_dataset_concept_edges)),
        _macro("CrossDatasetEdgeCount", _fmt(cross_dataset_edges)),
        _macro(
            "SameRegionCrossModalityCount",
            _fmt(display_edge_counts.get("same_region_cross_modality", 0)),
        ),
        _macro(
            "SameTaskCrossSpeciesCount", _fmt(display_edge_counts.get("same_task_cross_species", 0))
        ),
        _macro("DatasetHasModalityCount", _fmt(edge_counts.get("dataset_has_modality", 0))),
        _macro("DatasetRecordsRegionCount", _fmt(edge_counts.get("dataset_records_region", 0))),
        _macro("DatasetHasSpeciesCount", _fmt(edge_counts.get("dataset_has_species", 0))),
        _macro("DatasetHasTaskCount", _fmt(edge_counts.get("dataset_has_task", 0))),
        _macro("CitationEdgeCount", _fmt(edge_counts.get("paper_cites_paper", 0))),
        "% Reanalysis edge counts",
        _macro(
            "ReanalysisCandidateEdgeCount",
            _fmt(reanalysis.get("dataset_old_dataset_new_method_candidate", 0)),
        ),
        _macro(
            "ReanalysisBridgeEdgeCount", _fmt(reanalysis.get("dataset_reanalysis_bridge_dataset", 0))
        ),
        _macro(
            "ReinterpretationCandidateEdgeCount",
            _fmt(reanalysis.get("dataset_reinterpretation_candidate", 0)),
        ),
        "% Paper-dataset linkage",
        _macro("PaperLinkTotalRows", _fmt(paper_links.get("total_rows", "N/A"))),
        _macro("PaperLinkDoiExact", _fmt(match_methods.get("doi_exact", 0))),
        _macro("PaperLinkFuzzyTitle", _fmt(match_methods.get("title_fuzzy_local", 0))),
        _macro("PaperLinkFuzzyLive", _fmt(match_methods.get("title_fuzzy", 0))),
        _macro("PaperLinkNotFound", _fmt(match_methods.get("not_found", 0))),
        _macro("PaperLinkRealMatches", _fmt(paper_links.get("real_matches", 0))),
        _macro(
            "PaperLinkDataciteRealMatches",
            _fmt(paper_links.get("by_source", {}).get("datacite", {}).get("real_matches", 0)),
        ),
        _macro(
            "PaperLinkCombinedRealMatches",
            _fmt(paper_links.get("combined_datasets_with_real_link", paper_links.get("real_matches", 0))),
        ),
        "% Qrels tiers",
        _macro("QrelsGoldRows", _fmt(qrels.get("gold", {}).get("rows", 0))),
        _macro("QrelsSilverRows", _fmt(qrels.get("silver", {}).get("rows", 0))),
        _macro("QrelsBronzeRows", _fmt(qrels.get("bronze", {}).get("rows", 0))),
        _macro("QrelsCanonicalRows", _fmt(qrels.get("canonical_llm_silver", {}).get("rows", 0))),
        "% Ablation benchmark (canonical LLM-silver qrels, 317 queries)",
        _macro("NdcgHybridRrf", ndcg("hybrid_rrf")),
        _macro("NdcgHybridGraph", ndcg("hybrid_graph")),
        _macro("NdcgTypedKg", ndcg("typed_kg")),
        _macro("NdcgFull", ndcg("full")),
        "% Manifest provenance",
        _macro("ManifestGeneratedAt", manifest.get("generated_at", "N/A")),
    ]
    return macros


def main() -> int:
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_artifact_manifest.py")],
        check=True,
    )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    macros = build_macros(manifest)

    header = (
        "% AUTO-GENERATED by scripts/generate_whitepaper_stats.py "
        f"from {MANIFEST_PATH.relative_to(PROJECT_ROOT).as_posix()}.\n"
        "% Do not hand-edit; re-run the script instead.\n"
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(header + "\n".join(macros) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
