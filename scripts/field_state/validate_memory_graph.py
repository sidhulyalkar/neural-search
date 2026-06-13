#!/usr/bin/env python3
"""Validate the field-state memory graph and compute quality metrics.

Usage::

    python scripts/field_state/validate_memory_graph.py \\
        --nodes artifacts/field_state/memory_graph_nodes.jsonl \\
        --edges artifacts/field_state/memory_graph_edges.jsonl \\
        --out-dir reports/field_state

Outputs:
    reports/field_state/memory_graph_validation.md
    reports/field_state/memory_graph_validation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.field_state.graph_store import FieldStateGraphStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("validate_memory_graph")

DEFAULT_NODES = Path("artifacts/field_state/memory_graph_nodes.jsonl")
DEFAULT_EDGES = Path("artifacts/field_state/memory_graph_edges.jsonl")
DEFAULT_OUT_DIR = Path("reports/field_state")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nodes", type=Path, default=DEFAULT_NODES)
    p.add_argument("--edges", type=Path, default=DEFAULT_EDGES)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def compute_quality_metrics(store: FieldStateGraphStore) -> dict:
    datasets = store.query_datasets()

    missing_modality: list[str] = []
    missing_species: list[str] = []
    missing_region: list[str] = []
    missing_task: list[str] = []
    missing_description: list[str] = []
    missing_raw_evidence: list[str] = []

    # Build adjacency by node_id for faster lookup
    neighbors_by_node: dict[str, list[str]] = defaultdict(list)
    for edge in store._edges.values():
        neighbors_by_node[edge.source_node_id].append(edge.edge_type)
        neighbors_by_node[edge.target_node_id].append(edge.edge_type)

    for ds in datasets:
        nid = ds.node_id
        edge_types = neighbors_by_node.get(nid, [])
        ds_id = ds.properties.get("dataset_id", nid)

        if "dataset_has_modality" not in edge_types:
            missing_modality.append(ds_id)
        if "dataset_has_species" not in edge_types:
            missing_species.append(ds_id)
        if "dataset_records_region" not in edge_types:
            missing_region.append(ds_id)
        if "dataset_has_task" not in edge_types:
            missing_task.append(ds_id)
        if not ds.properties.get("description"):
            missing_description.append(ds_id)
        has_raw = "dataset_has_raw_signal" in edge_types
        has_processed = "dataset_has_processed_signal" in edge_types
        if not has_raw and not has_processed:
            missing_raw_evidence.append(ds_id)

    # Orphan nodes (no edges)
    connected_node_ids: set[str] = set()
    for edge in store._edges.values():
        connected_node_ids.add(edge.source_node_id)
        connected_node_ids.add(edge.target_node_id)
    orphans = [nid for nid in store._nodes if nid not in connected_node_ids]

    # Node/edge counts by type
    node_counts: dict[str, int] = defaultdict(int)
    for node in store._nodes.values():
        node_counts[node.node_type] += 1
    edge_counts: dict[str, int] = defaultdict(int)
    for edge in store._edges.values():
        edge_counts[edge.edge_type] += 1

    # Average degree by source archive
    degree_by_source: dict[str, list[int]] = defaultdict(list)
    for ds in datasets:
        source = ds.properties.get("source", "unknown")
        degree_by_source[source].append(len(neighbors_by_node.get(ds.node_id, [])))

    avg_degree_by_source = {
        src: sum(degrees) / len(degrees) if degrees else 0.0
        for src, degrees in degree_by_source.items()
    }

    # Top disconnected datasets (missing multiple labels)
    disconnection_score: list[tuple[str, int]] = []
    for ds in datasets:
        ds_id = ds.properties.get("dataset_id", ds.node_id)
        edge_types = neighbors_by_node.get(ds.node_id, [])
        score = sum([
            "dataset_has_modality" not in edge_types,
            "dataset_has_species" not in edge_types,
            "dataset_records_region" not in edge_types,
            "dataset_has_task" not in edge_types,
            "dataset_has_raw_signal" not in edge_types and "dataset_has_processed_signal" not in edge_types,
            not ds.properties.get("description"),
        ])
        if score >= 3:
            disconnection_score.append((ds_id, score))
    top_disconnected = sorted(disconnection_score, key=lambda x: -x[1])[:10]

    return {
        "node_counts_by_type": dict(node_counts),
        "edge_counts_by_type": dict(edge_counts),
        "total_nodes": store.node_count,
        "total_edges": store.edge_count,
        "total_datasets": len(datasets),
        "orphan_node_count": len(orphans),
        "orphan_node_ids": orphans[:20],
        "datasets_missing_modality": len(missing_modality),
        "datasets_missing_species": len(missing_species),
        "datasets_missing_region": len(missing_region),
        "datasets_missing_task": len(missing_task),
        "datasets_missing_description": len(missing_description),
        "datasets_missing_raw_or_processed_evidence": len(missing_raw_evidence),
        "average_degree_by_source": avg_degree_by_source,
        "top_disconnected_datasets": top_disconnected,
        "high_value_curation_targets": [ds_id for ds_id, _ in top_disconnected[:5]],
    }


def validate_guardrails(store: FieldStateGraphStore) -> list[str]:
    """Check semantic guardrail violations."""
    violations: list[str] = []

    for node in store._nodes.values():
        # No human_gold in judgment nodes
        if node.node_type == "neuro_judge_judgment":
            if node.properties.get("label_provenance") == "human_gold":
                violations.append(f"GUARDRAIL: judgment {node.node_id} has human_gold provenance")
            note = node.properties.get("provenance_note", "")
            if "silver" not in note:
                violations.append(f"WARN: judgment {node.node_id} missing silver provenance note")

        # Feedback must be downstream signal, not gold
        if node.node_type == "feedback_signal":
            prov = node.properties.get("provenance", "")
            if prov == "human_gold":
                violations.append(f"GUARDRAIL: feedback {node.node_id} marked as human_gold")

    return violations


def render_markdown_report(metrics: dict, errors: list[str], guardrails: list[str]) -> str:
    lines = ["# Memory Graph Validation Report", ""]
    lines += [f"**Total nodes:** {metrics['total_nodes']}  "]
    lines += [f"**Total edges:** {metrics['total_edges']}  "]
    lines += [f"**Total datasets:** {metrics['total_datasets']}  "]
    lines += [f"**Orphan nodes:** {metrics['orphan_node_count']}  "]
    lines += [""]
    lines += ["## Completeness Issues"]
    lines += [f"- Missing modality: {metrics['datasets_missing_modality']}"]
    lines += [f"- Missing species: {metrics['datasets_missing_species']}"]
    lines += [f"- Missing region: {metrics['datasets_missing_region']}"]
    lines += [f"- Missing task: {metrics['datasets_missing_task']}"]
    lines += [f"- Missing description: {metrics['datasets_missing_description']}"]
    lines += [f"- Missing raw/processed evidence: {metrics['datasets_missing_raw_or_processed_evidence']}"]
    lines += [""]
    lines += ["## Node Counts by Type"]
    for ntype, count in sorted(metrics["node_counts_by_type"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{ntype}`: {count}")
    lines += [""]
    lines += ["## Edge Counts by Type"]
    for etype, count in sorted(metrics["edge_counts_by_type"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{etype}`: {count}")
    lines += [""]
    lines += ["## Average Graph Degree by Source"]
    for src, deg in sorted(metrics["average_degree_by_source"].items(), key=lambda x: -x[1]):
        lines.append(f"- {src}: {deg:.1f}")
    lines += [""]
    if metrics["top_disconnected_datasets"]:
        lines += ["## Top Disconnected Datasets (Curation Targets)"]
        for ds_id, score in metrics["top_disconnected_datasets"]:
            lines.append(f"- `{ds_id}` (missing {score}/6 label categories)")
        lines += [""]
    if errors:
        lines += ["## Invariant Errors"]
        for err in errors[:30]:
            lines.append(f"- {err}")
        lines += [""]
    if guardrails:
        lines += ["## Guardrail Violations"]
        for v in guardrails[:30]:
            lines.append(f"- **{v}**")
        lines += [""]
    else:
        lines += ["## Guardrail Violations", "None — all guardrails passed."]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    if not args.nodes.exists():
        log.error("Nodes file not found: %s", args.nodes)
        log.error("Run scripts/field_state/build_memory_graph.py first")
        sys.exit(1)

    log.info("Loading graph from %s / %s", args.nodes, args.edges)
    store = FieldStateGraphStore.from_jsonl(args.nodes, args.edges)
    log.info("Loaded: %s", store)

    invariant_errors = store.validate_invariants()
    guardrail_violations = validate_guardrails(store)
    metrics = compute_quality_metrics(store)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "memory_graph_validation.json"
    report = {
        "metrics": metrics,
        "invariant_errors": invariant_errors,
        "guardrail_violations": guardrail_violations,
        "passed": len(invariant_errors) == 0 and len(guardrail_violations) == 0,
    }
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote validation JSON → %s", json_path)

    md_path = out_dir / "memory_graph_validation.md"
    md_path.write_text(
        render_markdown_report(metrics, invariant_errors, guardrail_violations),
        encoding="utf-8",
    )
    log.info("Wrote validation report → %s", md_path)

    print(f"\n{'PASS' if report['passed'] else 'FAIL'} — {len(invariant_errors)} invariant error(s), {len(guardrail_violations)} guardrail violation(s)")
    print(f"  Nodes: {metrics['total_nodes']}, Edges: {metrics['total_edges']}, Datasets: {metrics['total_datasets']}")


if __name__ == "__main__":
    main()
