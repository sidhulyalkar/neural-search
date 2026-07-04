#!/usr/bin/env python3
"""Compute the current-state artifact manifest directly from files on disk.

Replaces the hand-maintained `reports/eval/current_artifact_manifest.json`
(last hand-edited 2026-06-24, which drifted to citing a stale graph size —
7,593/31,920 nodes/edges — while the live production graph had grown to
7,946/144,428). This was flagged as backlog since that file's own
`reconciliation_note` in 2026-06-24: "No build_artifact_manifest.py script
exists in the repo... writing that script is queued as backlog."

Every field here is computed by reading the actual artifact, not carried
over from a previous manifest or asserted from memory. If a file is missing,
that section reports `available: false` rather than silently omitting itself
or guessing — a stale-but-present number is worse than an honest gap.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
CORPUS_PATH = (
    PROJECT_ROOT / "data" / "corpus" / "normalized" / "combined_corpus.jsonl" / "full_corpus_v09.jsonl"
)
GRAPH_PATH = PROJECT_ROOT / "data" / "graph" / "neural_search_graph.real_corpus.json"
PAPER_LINKS_PATH = PROJECT_ROOT / "artifacts" / "literature" / "paper_dataset_links.jsonl"
# Additional per-source literature-linking outputs (2026-07-02 expansion:
# neural_search/literature/datacite.py, crossref.py, semantic_scholar.py,
# pubmed.py) -- each is a standalone file; see
# neural_search/graph/paper_node_builder.py's docstring for why no formal
# merge step is required on the production-graph build path (though
# neural_search.literature.merge_links.merge_link_sources() exists as an
# optional reporting utility).
ADDITIONAL_PAPER_LINKS_PATHS = {
    "datacite": PROJECT_ROOT / "artifacts" / "literature" / "paper_dataset_links.datacite.jsonl",
    "crossref": PROJECT_ROOT / "artifacts" / "literature" / "paper_dataset_links.crossref.jsonl",
    "semantic_scholar": PROJECT_ROOT / "artifacts" / "literature" / "paper_dataset_links.semantic_scholar.jsonl",
    "pubmed": PROJECT_ROOT / "artifacts" / "literature" / "paper_dataset_links.pubmed.jsonl",
}
ABLATION_REPORT_PATH = PROJECT_ROOT / "reports" / "eval" / "ablation_ladder_report.partial.json"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "eval" / "current_artifact_manifest.json"

QRELS_FILES = {
    "gold": PROJECT_ROOT / "artifacts" / "qrels_gold.jsonl",
    "silver": PROJECT_ROOT / "artifacts" / "qrels_silver.jsonl",
    "bronze": PROJECT_ROOT / "artifacts" / "qrels_bronze.jsonl",
    "field_state_adjudicated": PROJECT_ROOT / "artifacts" / "field_state" / "adjudicated_qrels.jsonl",
    "canonical_llm_silver": PROJECT_ROOT / "data" / "qrels" / "qrels.canonical.jsonl",
}


def _rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _count_lines(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def _corpus_section() -> dict[str, Any]:
    if not CORPUS_PATH.exists():
        return {"available": False, "path": _rel(CORPUS_PATH)}

    source_ids: set[str] = set()
    dataset_ids: set[str] = set()
    row_count = 0
    with CORPUS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row_count += 1
            row = json.loads(line)
            source = row.get("source")
            source_id = row.get("source_id")
            if source and source_id:
                source_ids.add(f"{source}:{source_id}")
            if row.get("dataset_id"):
                dataset_ids.add(row["dataset_id"])
    return {
        "available": True,
        "path": _rel(CORPUS_PATH),
        "row_count": row_count,
        "unique_source_ids": len(source_ids),
        "unique_dataset_ids": len(dataset_ids),
    }


def _graph_section() -> dict[str, Any]:
    if not GRAPH_PATH.exists():
        return {"available": False, "path": _rel(GRAPH_PATH)}

    payload = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    nodes = payload.get("nodes", {})
    edges = payload.get("edges", {})
    node_values = nodes.values() if isinstance(nodes, dict) else nodes
    edge_values = edges.values() if isinstance(edges, dict) else edges

    node_type_counts = Counter(n.get("node_type", "unknown") for n in node_values)
    edge_type_counts = Counter(e.get("edge_type", "unknown") for e in edge_values)
    # scripts/build_real_corpus_graph.py's own summary printout breaks
    # dataset_similar_to_dataset down by its properties["cross_type"]
    # (same_region_cross_modality / same_task_cross_species) rather than by
    # edge_type — the whitepaper cites these cross_type labels as if they
    # were edge types, so replicate that same display convention here rather
    # than silently reporting 0 for them (a real bug caught while wiring
    # this up: edge_type_counts alone doesn't have these keys at all).
    display_edge_counts = Counter(
        e.get("properties", {}).get("cross_type") or e.get("edge_type", "unknown")
        for e in edge_values
    )
    stub_node_count = sum(
        1 for n in (nodes.values() if isinstance(nodes, dict) else nodes)
        if n.get("properties", {}).get("stub") is True
    )

    return {
        "available": True,
        "path": _rel(GRAPH_PATH),
        "total_nodes": sum(node_type_counts.values()),
        "total_edges": sum(edge_type_counts.values()),
        "node_type_counts": dict(sorted(node_type_counts.items(), key=lambda kv: -kv[1])),
        "edge_type_counts": dict(sorted(edge_type_counts.items(), key=lambda kv: -kv[1])),
        "display_edge_counts": dict(sorted(display_edge_counts.items(), key=lambda kv: -kv[1])),
        "stub_node_count": stub_node_count,
        "metadata": payload.get("metadata", {}),
    }


def _qrels_section() -> dict[str, Any]:
    section: dict[str, Any] = {}
    for tier, path in QRELS_FILES.items():
        count = _count_lines(path)
        section[tier] = {
            "path": _rel(path),
            "available": count is not None,
            "rows": count if count is not None else 0,
        }
    return section


_NOT_REAL_MATCH_METHODS = {"not_found", "not_applicable_no_dataset_doi"}


def _read_paper_links_file(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    match_methods: Counter[str] = Counter()
    dataset_ids: set[str] = set()
    total = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            match_methods[row.get("match_method", "unknown")] += 1
            if row.get("match_method") not in _NOT_REAL_MATCH_METHODS:
                dataset_ids.add(row["dataset_record_id"])

    real_matches = sum(
        count for method, count in match_methods.items() if method not in _NOT_REAL_MATCH_METHODS
    )
    return {
        "path": _rel(path),
        "total_rows": total,
        "match_method_counts": dict(match_methods),
        "real_matches": real_matches,
        "real_match_rate": round(real_matches / total, 4) if total else 0.0,
        "_dataset_ids": dataset_ids,
    }


def _paper_links_section() -> dict[str, Any]:
    openalex = _read_paper_links_file(PAPER_LINKS_PATH)
    if openalex is None:
        return {"available": False, "path": _rel(PAPER_LINKS_PATH)}

    by_source = {"openalex": {k: v for k, v in openalex.items() if k != "_dataset_ids"}}
    all_real_dataset_ids: set[str] = set(openalex["_dataset_ids"])

    for source_name, path in ADDITIONAL_PAPER_LINKS_PATHS.items():
        parsed = _read_paper_links_file(path)
        if parsed is None:
            continue
        by_source[source_name] = {k: v for k, v in parsed.items() if k != "_dataset_ids"}
        all_real_dataset_ids |= parsed["_dataset_ids"]

    return {
        "available": True,
        "by_source": by_source,
        # Union across sources -- a dataset with both an OpenAlex and a
        # DataCite match counts once here, matching "how many datasets have
        # at least one real literature link" rather than summing rows.
        "combined_datasets_with_real_link": len(all_real_dataset_ids),
        # Backward-compat top-level fields (pre-2026-07-02 consumers read
        # these directly off this section for the OpenAlex-only figure).
        "path": _rel(PAPER_LINKS_PATH),
        "total_rows": openalex["total_rows"],
        "match_method_counts": openalex["match_method_counts"],
        "real_matches": openalex["real_matches"],
        "real_match_rate": openalex["real_match_rate"],
    }


def _ablation_section() -> dict[str, Any]:
    if not ABLATION_REPORT_PATH.exists():
        return {"available": False, "path": _rel(ABLATION_REPORT_PATH)}

    payload = json.loads(ABLATION_REPORT_PATH.read_text(encoding="utf-8"))
    rungs = {
        rung["rung"]: {"status": rung["status"], "metrics": rung.get("metrics", {})}
        for rung in payload.get("rungs", [])
    }
    skipped = [name for name, r in rungs.items() if r["status"] == "skipped"]
    return {
        "available": True,
        "path": _rel(ABLATION_REPORT_PATH),
        "is_partial_run": bool(skipped),
        "skipped_rungs": skipped,
        "rungs": rungs,
    }


def build_manifest() -> dict[str, Any]:
    graph = _graph_section()
    paper_links = _paper_links_section()
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "generated_by": "scripts/build_artifact_manifest.py",
        "note": (
            "Computed directly from files on disk at generation time. "
            "Re-run this script rather than hand-editing this file."
        ),
        "corpus": _corpus_section(),
        "knowledge_graph": graph,
        "reanalysis_edges": {
            "dataset_old_dataset_new_method_candidate": graph.get("edge_type_counts", {}).get(
                "dataset_old_dataset_new_method_candidate", 0
            ),
            "dataset_reanalysis_bridge_dataset": graph.get("edge_type_counts", {}).get(
                "dataset_reanalysis_bridge_dataset", 0
            ),
            "dataset_reinterpretation_candidate": graph.get("edge_type_counts", {}).get(
                "dataset_reinterpretation_candidate", 0
            ),
            "dataset_reprocessing_candidate": graph.get("edge_type_counts", {}).get(
                "dataset_reprocessing_candidate", 0
            ),
        },
        "qrels": _qrels_section(),
        "paper_dataset_links": paper_links,
        "ablation_benchmark": _ablation_section(),
    }


def main() -> int:
    manifest = build_manifest()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=False), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")
    print(f"  Corpus rows: {manifest['corpus'].get('row_count')}")
    print(
        f"  Graph: {manifest['knowledge_graph'].get('total_nodes')} nodes / "
        f"{manifest['knowledge_graph'].get('total_edges')} edges "
        f"({manifest['knowledge_graph'].get('stub_node_count')} stub nodes)"
    )
    combined = manifest["paper_dataset_links"].get("combined_datasets_with_real_link")
    print(f"  Paper links: {manifest['paper_dataset_links'].get('real_matches')} real matches (OpenAlex only)"
          + (f", {combined} datasets combined across all sources" if combined is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
