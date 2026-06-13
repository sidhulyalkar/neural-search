#!/usr/bin/env python3
"""Incremental field-state update pipeline.

Detects changed/new/removed corpus records by content hash, rebuilds affected
graph nodes and edges, and writes a new snapshot manifest.

Usage::

    python scripts/field_state/update_field_state.py [options]

Options:
    --corpus              Path to normalized corpus JSONL
    --out-dir             Artifact output directory (default: artifacts/field_state)
    --snapshots-dir       Snapshot directory (default: artifacts/field_state/snapshots)
    --dry-run             Detect changes but do not write outputs
    --since DATE          Only process records created/updated after DATE
    --force               Force full rebuild even if no changes detected
    --changed-only        Skip records that have not changed
    --skip-embeddings     Skip embedding index updates
    --skip-neuro-judge    Skip neuro-judge evidence packet rebuild
    --skip-feedback       Skip feedback signal ingestion
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neural_search.field_state.memory_graph import MemoryGraphBuilder, _load_jsonl

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("update_field_state")

DEFAULT_CORPUS = Path("data/corpus/normalized/combined_corpus.jsonl")
DEFAULT_OUT_DIR = Path("artifacts/field_state")
DEFAULT_SNAPSHOTS_DIR = Path("artifacts/field_state/snapshots")
DEFAULT_EVIDENCE_PACKETS = Path("artifacts/field_state/neuro_judge_evidence_packets.jsonl")
DEFAULT_JUDGMENTS = Path("artifacts/field_state/neuro_qrels_consensus.jsonl")
DEFAULT_FEEDBACK = Path("artifacts/frontend/retrieval_feedback.jsonl")
DEFAULT_CONCEPTS = Path("artifacts/field_state/concept_memory/concepts.jsonl")
CURRENT_MANIFEST = Path("artifacts/field_state/current_manifest.json")


def _record_hash(rec: dict) -> str:
    key = {k: rec.get(k) for k in ("title", "description", "source_id", "source")}
    return hashlib.sha256(json.dumps(key, sort_keys=True, default=str).encode()).hexdigest()[:16]


def load_previous_hashes(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return manifest.get("record_hashes", {})
    except Exception:
        return {}


def detect_changes(
    records: list[dict],
    previous_hashes: dict[str, str],
) -> tuple[list[dict], list[dict], list[str]]:
    """Return (new_records, changed_records, removed_dataset_ids)."""
    current_ids = {rec["dataset_id"] for rec in records if rec.get("dataset_id")}
    removed = [did for did in previous_hashes if did not in current_ids]

    new_records: list[dict] = []
    changed_records: list[dict] = []

    for rec in records:
        did = rec.get("dataset_id", "")
        if not did:
            continue
        h = _record_hash(rec)
        if did not in previous_hashes:
            new_records.append(rec)
        elif previous_hashes[did] != h:
            changed_records.append(rec)

    return new_records, changed_records, removed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--snapshots-dir", type=Path, default=DEFAULT_SNAPSHOTS_DIR)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--since", type=str, default=None, help="ISO date cutoff")
    p.add_argument("--force", action="store_true")
    p.add_argument("--changed-only", action="store_true")
    p.add_argument("--skip-embeddings", action="store_true")
    p.add_argument("--skip-neuro-judge", action="store_true")
    p.add_argument("--skip-feedback", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.monotonic()

    # Load corpus
    corpus_path: Path = args.corpus
    if not corpus_path.exists():
        log.error("Corpus not found: %s", corpus_path)
        sys.exit(1)
    records = _load_jsonl(corpus_path)
    log.info("Loaded %d corpus records from %s", len(records), corpus_path)

    # Filter by --since if provided
    if args.since:
        records = [
            r for r in records
            if r.get("created_at", "9999") >= args.since
        ]
        log.info("After --since %s filter: %d records", args.since, len(records))

    # Detect changes
    prev_hashes = {} if args.force else load_previous_hashes(CURRENT_MANIFEST)
    new_records, changed_records, removed_ids = detect_changes(records, prev_hashes)

    log.info(
        "Change detection: %d new, %d changed, %d removed",
        len(new_records), len(changed_records), len(removed_ids),
    )

    if args.changed_only:
        build_records = new_records + changed_records
        log.info("--changed-only: building %d records", len(build_records))
    else:
        build_records = records

    if not build_records and not args.force:
        log.info("No changes detected — skipping rebuild (use --force to override)")
        return

    if args.dry_run:
        log.info("--dry-run: detected %d new, %d changed, %d removed — not writing", len(new_records), len(changed_records), len(removed_ids))
        return

    # Build graph
    builder = MemoryGraphBuilder()
    store = builder.build(
        corpus_records=build_records,
        evidence_packets_path=None if args.skip_neuro_judge else DEFAULT_EVIDENCE_PACKETS,
        judgments_path=None if args.skip_neuro_judge else DEFAULT_JUDGMENTS,
        feedback_path=None if args.skip_feedback else DEFAULT_FEEDBACK,
        concept_memory_path=DEFAULT_CONCEPTS,
    )
    log.info("Graph built: %s", store)

    elapsed = time.monotonic() - t0

    # Write artifacts
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    store.export_jsonl(out_dir / "memory_graph_nodes.jsonl", out_dir / "memory_graph_edges.jsonl")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    build_id = f"update_{timestamp}"
    store.write_manifest(out_dir / "memory_graph_manifest.json", build_id=build_id)

    # Snapshot
    snapshot_dir = args.snapshots_dir / timestamp
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Write snapshot manifests
    record_hashes = {rec["dataset_id"]: _record_hash(rec) for rec in records if rec.get("dataset_id")}
    corpus_manifest = {
        "build_id": build_id,
        "timestamp": timestamp,
        "corpus_path": str(corpus_path),
        "record_count": len(records),
        "record_hashes": record_hashes,
        "new_count": len(new_records),
        "changed_count": len(changed_records),
        "removed_count": len(removed_ids),
        "removed_ids": removed_ids,
    }
    (snapshot_dir / "corpus_manifest.json").write_text(
        json.dumps(corpus_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    store.write_manifest(snapshot_dir / "memory_graph_manifest.json", build_id=build_id)

    index_manifest: dict = {
        "build_id": build_id,
        "skip_embeddings": args.skip_embeddings,
        "skip_neuro_judge": args.skip_neuro_judge,
        "skip_feedback": args.skip_feedback,
    }
    (snapshot_dir / "index_manifest.json").write_text(
        json.dumps(index_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Update report
    update_report = f"""# Field-State Update Report

**Build ID:** {build_id}
**Timestamp:** {timestamp}
**Elapsed:** {elapsed:.2f}s

## Change Summary
- New records: {len(new_records)}
- Changed records: {len(changed_records)}
- Removed records: {len(removed_ids)}

## Graph
- Total nodes: {store.node_count}
- Total edges: {store.edge_count}

## Options
- skip_embeddings: {args.skip_embeddings}
- skip_neuro_judge: {args.skip_neuro_judge}
- skip_feedback: {args.skip_feedback}
- changed_only: {args.changed_only}
- force: {args.force}
"""
    (snapshot_dir / "update_report.md").write_text(update_report, encoding="utf-8")
    log.info("Snapshot written → %s", snapshot_dir)

    # Update current_manifest.json
    current = {
        "build_id": build_id,
        "timestamp": timestamp,
        "snapshot_dir": str(snapshot_dir),
        "record_hashes": record_hashes,
        "graph": {
            "total_nodes": store.node_count,
            "total_edges": store.edge_count,
        },
    }
    CURRENT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_MANIFEST.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
    log.info("Updated current_manifest → %s", CURRENT_MANIFEST)

    print(f"\nField-state updated in {elapsed:.2f}s ({build_id})")
    print(f"  New: {len(new_records)}, Changed: {len(changed_records)}, Removed: {len(removed_ids)}")
    print(f"  Graph: {store.node_count} nodes, {store.edge_count} edges")
    print(f"  Snapshot: {snapshot_dir}")


if __name__ == "__main__":
    main()
