#!/usr/bin/env python3
"""Rebuild the knowledge graph from the full v09 flat corpus.

Reads data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl,
converts each flat record to NormalizedDatasetRecord, builds the base
provenance graph, then enriches it with cross-dataset relationship edges:

  - same_region_cross_modality: datasets sharing a brain region but using
    different recording modalities (multimodal reuse candidates)
  - same_task_cross_species: datasets sharing a task across different species
    (cross-species generalization candidates)
  - same_region_same_task: datasets sharing both a region AND a task
    (tightest methodological siblings)

Cross-dataset edges are capped per concept (MAX_PAIRS_PER_CONCEPT=15) to
avoid edge explosion on high-frequency concepts like "hippocampus".

Output: data/graph/neural_search_graph.real_corpus.json

Usage
-----
    python scripts/rebuild_full_corpus_graph.py
    python scripts/rebuild_full_corpus_graph.py --dry-run
    python scripts/rebuild_full_corpus_graph.py --min-confidence 0.4 --limit 500
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("rebuild_full_corpus_graph")

CORPUS_PATH = ROOT / "data" / "corpus" / "normalized" / "combined_corpus.jsonl" / "full_corpus_v09.jsonl"
GRAPH_OUT = ROOT / "data" / "graph" / "neural_search_graph.real_corpus.json"

MAX_PAIRS_PER_CONCEPT = 15  # cap cross-dataset edges per shared concept


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _to_evidence_labels(values: Any, label_type: str) -> list:
    """Convert a flat list of strings to EvidenceLabel objects."""
    from neural_search.schemas import EvidenceLabel

    if not values:
        return []
    labels = []
    for v in values:
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                v = ast.literal_eval(v)
            except (SyntaxError, ValueError):
                pass
        if isinstance(v, str) and v.strip():
            val = v.strip()
            labels.append(
                EvidenceLabel(
                    id=val,
                    label=val,
                    label_type=label_type,
                    confidence=1.0,
                    source_field=label_type,
                    source_value=val,
                )
            )
        elif isinstance(v, dict):
            # Already in EvidenceLabel shape
            try:
                labels.append(EvidenceLabel.model_validate(v))
            except Exception:
                pass
    return labels


def _to_analysis_affordances(values: Any) -> list:
    from neural_search.schemas import AnalysisAffordance

    if not values:
        return []
    affordances = []
    for v in values:
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                v = ast.literal_eval(v)
            except (SyntaxError, ValueError):
                pass
        if isinstance(v, dict):
            try:
                affordances.append(AnalysisAffordance.model_validate(v))
            except Exception:
                pass
    return affordances


def flat_to_normalized(rec: dict[str, Any]):
    """Convert a flat corpus record to NormalizedDatasetRecord.

    Handles two flat schemas:
    - v09 (new): has ``dataset_id`` field like ``"dataset:dandi:000785"``
    - legacy (old): only has ``source`` + ``source_id``; ``dataset_id`` is
      synthesized as ``"dataset:<source>:<source_id>"``
    """
    from neural_search.schemas import NormalizedDatasetRecord, UsabilityFlags

    source = str(rec.get("source", "unknown")).strip() or "unknown"
    source_id = str(rec.get("source_id", "")).strip() or "unknown"
    dataset_id = (
        str(rec["dataset_id"]).strip()
        if rec.get("dataset_id")
        else f"dataset:{source}:{source_id}"
    )

    # Legacy records store usability as top-level booleans, not nested dict
    usability_raw = rec.get("usability_flags") or {
        "has_behavior": rec.get("has_behavior"),
        "has_trials": rec.get("has_trials"),
        "has_raw_data": rec.get("has_raw_data"),
        "has_processed_data": rec.get("has_processed_data"),
    }
    usability = UsabilityFlags.model_validate(usability_raw) if usability_raw else UsabilityFlags()

    title = str(rec.get("title", "") or dataset_id).strip() or dataset_id

    return NormalizedDatasetRecord(
        dataset_id=dataset_id,
        source=source,
        source_id=source_id,
        title=title,
        description=rec.get("description"),
        url=rec.get("url"),
        species=_to_evidence_labels(rec.get("species", []), "species"),
        modalities=_to_evidence_labels(rec.get("modalities", []), "modality"),
        recording_scales=_to_evidence_labels(rec.get("recording_scales", []), "recording_scale"),
        brain_regions=_to_evidence_labels(rec.get("brain_regions", []), "brain_region"),
        tasks=_to_evidence_labels(rec.get("tasks", []), "task"),
        behavioral_events=_to_evidence_labels(rec.get("behavioral_events", []), "behavioral_event"),
        analysis_goals=_to_evidence_labels(rec.get("analysis_goals", []), "analysis_goal"),
        data_standards=_to_evidence_labels(rec.get("data_standards", []), "data_standard"),
        file_formats=_to_evidence_labels(rec.get("file_formats", []), "file_format"),
        linked_papers=list(rec.get("linked_papers", [])),
        usability_flags=usability,
        missing_fields=list(rec.get("missing_fields", [])),
        analysis_affordances=_to_analysis_affordances(rec.get("analysis_affordances", [])),
        created_at=rec.get("created_at", _now()),
        extractor_version=rec.get("extractor_version", "v0.3.0"),
    )


def load_corpus(path: Path, limit: int | None) -> list[dict]:
    records: list[dict] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if limit and len(records) >= limit:
                    break
    log.info("Loaded %d raw records from %s", len(records), path)
    return records


def _add_cross_dataset_edges(
    graph,
    datasets: list,
) -> int:
    """Add cross-dataset relationship edges to an existing graph.

    Returns number of edges added.
    """
    from neural_search.graph.builder import dataset_node_id
    from neural_search.graph.schema import (
        GraphEvidence,
        KnowledgeGraphEdge,
        make_edge_id,
    )

    # Build indexes: concept_id → list of (dataset_node_id, modality_set, task_set, species_set)
    region_to_datasets: dict[str, list[str]] = defaultdict(list)
    modality_to_datasets: dict[str, list[str]] = defaultdict(list)
    task_to_datasets: dict[str, list[str]] = defaultdict(list)
    species_to_datasets: dict[str, list[str]] = defaultdict(list)

    dataset_regions: dict[str, set[str]] = {}
    dataset_modalities: dict[str, set[str]] = {}
    dataset_tasks: dict[str, set[str]] = {}
    dataset_species: dict[str, set[str]] = {}

    for ds in datasets:
        nid = dataset_node_id(ds)
        if nid not in graph.nodes:
            continue

        regions = {lbl.id for lbl in ds.brain_regions}
        modalities = {lbl.id for lbl in ds.modalities}
        tasks = {lbl.id for lbl in ds.tasks}
        species = {lbl.id for lbl in ds.species}

        dataset_regions[nid] = regions
        dataset_modalities[nid] = modalities
        dataset_tasks[nid] = tasks
        dataset_species[nid] = species

        for r in regions:
            region_to_datasets[r].append(nid)
        for m in modalities:
            modality_to_datasets[m].append(nid)
        for t in tasks:
            task_to_datasets[t].append(nid)
        for s in species:
            species_to_datasets[s].append(nid)

    n_added = 0
    seen_pairs: set[tuple[str, str, str]] = set()

    def _add_edge(
        src: str,
        tgt: str,
        edge_type: str,
        context: str,
        *,
        properties: dict[str, Any] | None = None,
    ) -> None:
        nonlocal n_added
        pair_key = (min(src, tgt), max(src, tgt), edge_type)
        if pair_key in seen_pairs:
            return
        seen_pairs.add(pair_key)
        edge_id = make_edge_id(src, edge_type, tgt)
        if edge_id in graph.edges:
            return
        evidence = GraphEvidence(
            evidence_id=f"evidence:cross_dataset:{edge_type}:{src}:{tgt}",
            source_type="cross_dataset_analysis",
            source_id=f"{src}:{tgt}",
            source_field=edge_type,
            evidence_text=context,
            confidence=0.7,
            extractor_name="rebuild_full_corpus_graph",
            extractor_version="v1.0.0",
        )
        edge = KnowledgeGraphEdge(
            edge_id=edge_id,
            edge_type=edge_type,
            source_node_id=src,
            target_node_id=tgt,
            evidence=[evidence],
            confidence=0.7,
            properties={
                "context": context,
                **(properties or {}),
            },
            created_at=_now(),
        )
        graph.edges[edge_id] = edge
        n_added += 1

    # same_region_cross_modality: datasets sharing a region with different modalities
    for region_id, ds_ids in region_to_datasets.items():
        if len(ds_ids) < 2:
            continue
        pairs_added = 0
        for i, a in enumerate(ds_ids):
            if pairs_added >= MAX_PAIRS_PER_CONCEPT:
                break
            for b in ds_ids[i + 1:]:
                if pairs_added >= MAX_PAIRS_PER_CONCEPT:
                    break
                mods_a = dataset_modalities.get(a, set())
                mods_b = dataset_modalities.get(b, set())
                if mods_a and mods_b and not mods_a.intersection(mods_b):
                    _add_edge(
                        a, b,
                        "same_region_cross_modality",
                        f"shared_region:{region_id} modalities:{sorted(mods_a)}x{sorted(mods_b)}",
                        properties={
                            "relationship_type": "same_region_cross_modality",
                            "shared_region": region_id,
                            "source_modalities": sorted(mods_a),
                            "target_modalities": sorted(mods_b),
                        },
                    )
                    _add_edge(
                        a, b,
                        "dataset_reanalysis_bridge_dataset",
                        f"reanalysis_bridge:shared_region:{region_id}:cross_modality",
                        properties={
                            "relationship_type": "multimodal_reanalysis_bridge",
                            "shared_region": region_id,
                            "reason": "shared brain region with different modalities",
                        },
                    )
                    pairs_added += 1

    # same_task_cross_species: datasets sharing a task with different species
    for task_id, ds_ids in task_to_datasets.items():
        if len(ds_ids) < 2:
            continue
        pairs_added = 0
        for i, a in enumerate(ds_ids):
            if pairs_added >= MAX_PAIRS_PER_CONCEPT:
                break
            for b in ds_ids[i + 1:]:
                if pairs_added >= MAX_PAIRS_PER_CONCEPT:
                    break
                sp_a = dataset_species.get(a, set())
                sp_b = dataset_species.get(b, set())
                if sp_a and sp_b and not sp_a.intersection(sp_b):
                    _add_edge(
                        a, b,
                        "same_task_cross_species",
                        f"shared_task:{task_id} species:{sorted(sp_a)}x{sorted(sp_b)}",
                        properties={
                            "relationship_type": "same_task_cross_species",
                            "shared_task": task_id,
                            "source_species": sorted(sp_a),
                            "target_species": sorted(sp_b),
                        },
                    )
                    _add_edge(
                        a, b,
                        "dataset_reinterpretation_candidate",
                        f"reinterpretation_candidate:shared_task:{task_id}:cross_species",
                        properties={
                            "relationship_type": "cross_species_reinterpretation",
                            "shared_task": task_id,
                            "reason": "shared task across species can support generalization checks",
                        },
                    )
                    pairs_added += 1

    # same_region_same_task: datasets sharing both region AND task (methodological siblings)
    for region_id, r_ds_ids in region_to_datasets.items():
        if len(r_ds_ids) < 2:
            continue
        r_set = set(r_ds_ids)
        for task_id, t_ds_ids in task_to_datasets.items():
            overlap = r_set.intersection(t_ds_ids)
            if len(overlap) < 2:
                continue
            overlap_list = sorted(overlap)
            pairs_added = 0
            for i, a in enumerate(overlap_list):
                if pairs_added >= MAX_PAIRS_PER_CONCEPT:
                    break
                for b in overlap_list[i + 1:]:
                    if pairs_added >= MAX_PAIRS_PER_CONCEPT:
                        break
                    _add_edge(
                        a, b,
                        "same_region_same_task",
                        f"shared_region:{region_id} shared_task:{task_id}",
                        properties={
                            "relationship_type": "same_region_same_task",
                            "shared_region": region_id,
                            "shared_task": task_id,
                        },
                    )
                    _add_edge(
                        a, b,
                        "dataset_reprocessing_candidate",
                        f"reprocessing_candidate:shared_region:{region_id}:shared_task:{task_id}",
                        properties={
                            "relationship_type": "methodological_sibling_reprocessing",
                            "shared_region": region_id,
                            "shared_task": task_id,
                            "reason": "same task and region can support harmonized reprocessing",
                        },
                    )
                    pairs_added += 1

    log.info("Cross-dataset edges added: %d", n_added)
    return n_added


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--out", type=Path, default=GRAPH_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.corpus.exists():
        log.error("Corpus not found: %s", args.corpus)
        return 1

    raw_records = load_corpus(args.corpus, args.limit)

    log.info("Converting %d records to NormalizedDatasetRecord ...", len(raw_records))
    datasets = []
    skipped = 0
    for rec in raw_records:
        try:
            datasets.append(flat_to_normalized(rec))
        except Exception as e:
            log.debug("Skip %s: %s", rec.get("dataset_id", "?"), e)
            skipped += 1

    log.info("Converted: %d | Skipped: %d", len(datasets), skipped)

    if args.dry_run:
        log.info("Dry run — not building graph")
        return 0

    from neural_search.graph.builder import build_graph_from_records
    from neural_search.graph.schema import write_graph_json

    log.info("Building base graph (min_confidence=%.2f) ...", args.min_confidence)
    graph = build_graph_from_records(datasets, papers=[], min_confidence=args.min_confidence)

    base_nodes = len(graph.nodes)
    base_edges = len(graph.edges)
    log.info("Base graph: %d nodes, %d edges", base_nodes, base_edges)

    log.info("Adding cross-dataset relationship edges ...")
    n_cross = _add_cross_dataset_edges(graph, datasets)

    log.info(
        "Final graph: %d nodes, %d edges (+%d cross-dataset)",
        len(graph.nodes), len(graph.edges), n_cross,
    )

    graph.metadata.update({
        "corpus_path": str(args.corpus),
        "built_at": _now(),
        "total_records": len(raw_records),
        "converted_records": len(datasets),
        "skipped_records": skipped,
        "min_confidence": args.min_confidence,
        "cross_dataset_edges": n_cross,
        "builder_script": "scripts/rebuild_full_corpus_graph.py",
        "builder_version": "v1.0.0",
    })

    write_graph_json(graph, args.out)
    log.info("Graph written to %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
