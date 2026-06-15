"""
End-to-end integration demo for neural-search v0.9.

Exercises:
  1. Domain precision  — specific neuro queries that should resolve cleanly
  2. Cross-modal       — queries where modality specificity matters
  3. Analysis-ready    — queries targeting analysis affordances
  4. Provenance chain  — verify graph evidence flows to frontend payload
  5. Contrastive       — queries where a wrong dataset would be contraindicated
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CORPUS_PATH = Path("data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl")


def _load_corpus() -> tuple[list[dict[str, Any]], dict[str, dict]]:
    records = [json.loads(line) for line in CORPUS_PATH.read_text().strip().split("\n") if line.strip()]
    ds_map: dict[str, dict] = {}
    for r in records:
        ds_map[r["dataset_id"]] = r
        if r.get("source_id"):
            ds_map[r["source_id"]] = r
    return records, ds_map


def _resolve_title(r: Any, ds_map: dict) -> str:
    ds = ds_map.get(r.dataset_id) or ds_map.get(r.dataset_id.split(":")[-1], {})
    return (ds.get("title") or r.dataset_id)[:75]


def _extract_ids(field: Any) -> list[str]:
    if not field:
        return []
    if isinstance(field, list):
        return [(v.get("id") if isinstance(v, dict) else v) for v in field if v]
    return [str(field)]


def _print_result(i: int, r: Any, ds_map: dict) -> None:
    title = _resolve_title(r, ds_map)
    ds = ds_map.get(r.dataset_id) or ds_map.get(r.dataset_id.split(":")[-1], {})
    src = ds.get("source", "?")
    mods = _extract_ids(ds.get("modalities"))[:3]
    spc = _extract_ids(ds.get("species"))[:2]
    reg = _extract_ids(ds.get("brain_regions"))[:2]

    bd = getattr(r, "score_breakdown", {}) or {}
    mg = bd.get("memory_graph_score", 0)
    ev = getattr(r, "memory_graph_evidence", {}) or {}
    graph_hits = (
        ev.get("modality_matches", []) +
        ev.get("species_matches", []) +
        ev.get("region_matches", []) +
        ev.get("affordance_matches", [])
    )[:5]

    print(f"  {i}. [{r.score:.2f}] [{src}] {title}")
    print(f"     mods={mods}  spc={spc}  reg={reg}")
    if mg != 0 or graph_hits:
        graph_str = f"graph_score={mg:+.3f}"
        if graph_hits:
            graph_str += f"  hits={graph_hits}"
        print(f"     {graph_str}")
    why = (getattr(r, "why_matched", None) or [])[:2]
    for w in why:
        print(f"     ↳ {w}")


def run_suite(records: list, ds_map: dict) -> None:
    from neural_search.search.core import search_datasets

    QUERIES = [
        # ── Domain precision ────────────────────────────────────────────
        (
            "Domain precision",
            [
                "two-photon calcium imaging mouse hippocampal CA1 place cells",
                "neuropixels silicon probe multi-region recording mouse decision making",
                "patch-clamp single-unit auditory cortex frequency tuning",
            ],
        ),
        # ── Species + modality specificity ──────────────────────────────
        (
            "Cross-modal specificity",
            [
                "rat LFP prefrontal cortex theta oscillations",
                "macaque single-unit area MT visual motion",
                "human ECoG speech production Broca area",
            ],
        ),
        # ── Analysis affordances ─────────────────────────────────────────
        (
            "Analysis-readiness",
            [
                "NWB formatted dataset with behavioral events for spike sorting",
                "trial-aligned population dynamics suitable for dimensionality reduction",
                "BIDS fMRI dataset with task design matrix for GLM modeling",
            ],
        ),
        # ── Graph evidence / contrastive ────────────────────────────────
        (
            "Contrastive / graph evidence",
            [
                "calcium imaging NOT spike sorting NOT electrophysiology",
                "ephys spike trains dorsal striatum mouse reinforcement learning",
            ],
        ),
    ]

    for section, queries in QUERIES:
        print(f"\n{'═'*70}")
        print(f"  {section}")
        print(f"{'═'*70}")
        for query in queries:
            print(f"\nQ: {query}")
            results_obj = search_datasets(query, datasets=records, limit=5)
            results = results_obj.results if hasattr(results_obj, "results") else results_obj
            for i, r in enumerate(results[:3], 1):
                _print_result(i, r, ds_map)


if __name__ == "__main__":
    import logging
    logging.disable(logging.WARNING)

    print("Loading corpus and search engine…")
    records, ds_map = _load_corpus()
    print(f"Corpus: {len(records)} records")

    run_suite(records, ds_map)
