#!/usr/bin/env python3
"""Compare two field-state snapshots and report drift.

Usage::

    python scripts/field_state/compare_snapshots.py \\
        --old artifacts/field_state/snapshots/20260611T000000Z \\
        --new artifacts/field_state/snapshots/20260612T000000Z \\
        --out-dir reports/field_state

Outputs:
    reports/field_state/snapshot_diff.md
    reports/field_state/snapshot_diff.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("compare_snapshots")

DEFAULT_OUT_DIR = Path("reports/field_state")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True, help="Old snapshot directory")
    p.add_argument("--new", type=Path, required=True, help="New snapshot directory")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return p.parse_args()


def load_manifest(snapshot_dir: Path, filename: str) -> dict:
    path = snapshot_dir / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def diff_record_hashes(old_hashes: dict[str, str], new_hashes: dict[str, str]) -> dict:
    old_ids = set(old_hashes)
    new_ids = set(new_hashes)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    shared = old_ids & new_ids
    changed = sorted(did for did in shared if old_hashes[did] != new_hashes[did])
    unchanged = sorted(did for did in shared if old_hashes[did] == new_hashes[did])
    return {
        "added": added,
        "removed": removed,
        "changed": changed,
        "unchanged_count": len(unchanged),
        "total_old": len(old_ids),
        "total_new": len(new_ids),
    }


def diff_graph(old_manifest: dict, new_manifest: dict) -> dict:
    def _counts(m: dict, key: str) -> dict[str, int]:
        return m.get(key, {})

    old_nodes = _counts(old_manifest, "node_counts_by_type")
    new_nodes = _counts(new_manifest, "node_counts_by_type")
    old_edges = _counts(old_manifest, "edge_counts_by_type")
    new_edges = _counts(new_manifest, "edge_counts_by_type")

    all_node_types = set(old_nodes) | set(new_nodes)
    all_edge_types = set(old_edges) | set(new_edges)

    node_diff = {t: new_nodes.get(t, 0) - old_nodes.get(t, 0) for t in all_node_types}
    edge_diff = {t: new_edges.get(t, 0) - old_edges.get(t, 0) for t in all_edge_types}

    return {
        "total_nodes_old": old_manifest.get("total_nodes", 0),
        "total_nodes_new": new_manifest.get("total_nodes", 0),
        "total_edges_old": old_manifest.get("total_edges", 0),
        "total_edges_new": new_manifest.get("total_edges", 0),
        "node_diff_by_type": {k: v for k, v in node_diff.items() if v != 0},
        "edge_diff_by_type": {k: v for k, v in edge_diff.items() if v != 0},
    }


def render_markdown(diff: dict) -> str:
    corpus = diff.get("corpus", {})
    graph = diff.get("graph", {})
    lines = ["# Snapshot Comparison Report", ""]
    lines += [f"**Old snapshot:** `{diff.get('old_snapshot')}`  "]
    lines += [f"**New snapshot:** `{diff.get('new_snapshot')}`  "]
    lines += [""]
    lines += ["## Corpus Changes"]
    lines += [f"- Datasets added: **{len(corpus.get('added', []))}**"]
    lines += [f"- Datasets removed: **{len(corpus.get('removed', []))}**"]
    lines += [f"- Datasets changed: **{len(corpus.get('changed', []))}**"]
    lines += [f"- Unchanged: {corpus.get('unchanged_count', 0)}"]
    lines += [f"- Total old: {corpus.get('total_old', 0)}, Total new: {corpus.get('total_new', 0)}"]
    lines += [""]
    if corpus.get("added"):
        lines += ["### Added Datasets (first 10)"]
        for did in corpus["added"][:10]:
            lines.append(f"- `{did}`")
        lines += [""]
    if corpus.get("removed"):
        lines += ["### Removed Datasets (first 10)"]
        for did in corpus["removed"][:10]:
            lines.append(f"- `{did}`")
        lines += [""]
    if corpus.get("changed"):
        lines += ["### Changed Datasets (first 10)"]
        for did in corpus["changed"][:10]:
            lines.append(f"- `{did}`")
        lines += [""]
    lines += ["## Graph Changes"]
    lines += [f"- Nodes: {graph.get('total_nodes_old', 0)} → {graph.get('total_nodes_new', 0)}"]
    lines += [f"- Edges: {graph.get('total_edges_old', 0)} → {graph.get('total_edges_new', 0)}"]
    node_diff = graph.get("node_diff_by_type", {})
    if node_diff:
        lines += ["### Node Count Changes by Type"]
        for ntype, delta in sorted(node_diff.items(), key=lambda x: -abs(x[1])):
            sign = "+" if delta > 0 else ""
            lines.append(f"- `{ntype}`: {sign}{delta}")
    warnings = diff.get("warnings", [])
    if warnings:
        lines += ["", "## Warnings"]
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    old_dir: Path = args.old
    new_dir: Path = args.new

    if not old_dir.exists():
        log.error("Old snapshot not found: %s", old_dir)
        sys.exit(1)
    if not new_dir.exists():
        log.error("New snapshot not found: %s", new_dir)
        sys.exit(1)

    old_corpus = load_manifest(old_dir, "corpus_manifest.json")
    new_corpus = load_manifest(new_dir, "corpus_manifest.json")
    old_graph = load_manifest(old_dir, "memory_graph_manifest.json")
    new_graph = load_manifest(new_dir, "memory_graph_manifest.json")

    corpus_diff = diff_record_hashes(
        old_corpus.get("record_hashes", {}),
        new_corpus.get("record_hashes", {}),
    )
    graph_diff = diff_graph(old_graph, new_graph)

    # Detect large unexpected changes
    warnings: list[str] = []
    added_count = len(corpus_diff["added"])
    removed_count = len(corpus_diff["removed"])
    total_old = corpus_diff.get("total_old", 1) or 1
    if removed_count / total_old > 0.1:
        warnings.append(f"{removed_count} datasets removed ({removed_count/total_old:.0%} of prior corpus) — unexpected large removal")
    if added_count > 500:
        warnings.append(f"{added_count} datasets added in a single update — verify this is expected")

    diff_report = {
        "old_snapshot": str(old_dir),
        "new_snapshot": str(new_dir),
        "corpus": corpus_diff,
        "graph": graph_diff,
        "warnings": warnings,
    }

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "snapshot_diff.json"
    json_path.write_text(json.dumps(diff_report, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Wrote diff JSON → %s", json_path)

    md_path = out_dir / "snapshot_diff.md"
    md_path.write_text(render_markdown(diff_report), encoding="utf-8")
    log.info("Wrote diff report → %s", md_path)

    print(f"\nSnapshot diff: {corpus_diff['total_old']} → {corpus_diff['total_new']} datasets")
    print(f"  Added: {len(corpus_diff['added'])}, Removed: {len(corpus_diff['removed'])}, Changed: {len(corpus_diff['changed'])}")
    if warnings:
        print(f"  Warnings: {len(warnings)}")
        for w in warnings:
            print(f"    ⚠  {w}")


if __name__ == "__main__":
    main()
