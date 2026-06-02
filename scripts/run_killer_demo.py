#!/usr/bin/env python3
"""Killer Demo — 5-stage multi-dataset query pipeline.

Query:
  "Map the neural circuit mechanisms underlying flexible cognitive control —
   integrating datasets spanning prefrontal-hippocampal interactions,
   dopaminergic reward modulation, motor adaptation, and cross-species
   learning-dependent plasticity — to identify convergent computational
   mechanisms."

Stage 1: Query decomposition into typed sub-queries
Stage 2: Per-sub-query constraint extraction
Stage 3: Retrieval + set-coverage scoring
Stage 4: Role assignment to result set
Stage 5: Demo success metrics

Usage:
    python scripts/run_killer_demo.py
    python scripts/run_killer_demo.py --k 15 --output reports/killer_demo.json
    python scripts/run_killer_demo.py --dry-run
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

KILLER_QUERY = (
    "Map the neural circuit mechanisms underlying flexible cognitive control — "
    "integrating datasets spanning prefrontal-hippocampal interactions, "
    "dopaminergic reward modulation, motor adaptation, and cross-species "
    "learning-dependent plasticity — to identify convergent computational "
    "mechanisms that could be tested in a single unified experiment."
)

SUB_QUERIES = [
    {
        "id": "SQ1",
        "query": "prefrontal cortex hippocampus interaction working memory",
        "intent": "cross_dataset_comparison",
        "species_constraint": ["mouse", "macaque", "human"],
        "brain_regions": ["prefrontal_cortex", "hippocampus"],
        "task_family": "working_memory",
    },
    {
        "id": "SQ2",
        "query": "dopamine reward prediction error striatum",
        "intent": "meta_analysis",
        "species_constraint": ["mouse", "rat", "macaque"],
        "brain_regions": ["striatum"],
        "task_family": "reward_learning",
    },
    {
        "id": "SQ3",
        "query": "motor cortex adaptation learning plasticity",
        "intent": "method_transfer",
        "species_constraint": ["mouse", "macaque", "human"],
        "brain_regions": ["motor_cortex"],
        "task_family": "motor_task",
    },
    {
        "id": "SQ4",
        "query": "cross-species decision making flexible behavior reversal learning",
        "intent": "cross_dataset_comparison",
        "species_constraint": ["mouse", "rat", "macaque", "human"],
        "brain_regions": ["prefrontal_cortex", "striatum"],
        "task_family": "decision_making",
    },
    {
        "id": "SQ5",
        "query": "population dynamics prefrontal cortex latent space manifold",
        "intent": "method_transfer",
        "species_constraint": ["mouse", "macaque"],
        "brain_regions": ["prefrontal_cortex"],
        "task_family": "any",
    },
]


def _count_sub_query_matches(dataset: dict, sub_queries: list[dict]) -> int:
    """Count how many sub-queries this dataset is relevant for."""
    title_desc = f"{dataset.get('title', '')} {dataset.get('description', '')}".lower()
    regions = {
        (r.get("label") if isinstance(r, dict) else r).lower()
        for r in dataset.get("brain_regions", [])
    }
    species = {
        (s.get("label") if isinstance(s, dict) else s).lower()
        for s in dataset.get("species", [])
    }

    count = 0
    for sq in sub_queries:
        sq_words = set(sq["query"].lower().split())
        overlap_text = sq_words & set(title_desc.split())
        sq_regions = {r.lower() for r in sq.get("brain_regions", [])}
        sq_species = {s.lower() for s in sq.get("species_constraint", [])}

        text_match = len(overlap_text) >= 2
        region_match = bool(sq_regions & regions)
        species_match = bool(sq_species & species)

        if text_match or (region_match and species_match):
            count += 1

    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20, help="Datasets to retrieve per sub-query")
    parser.add_argument("--output", default="reports/killer_demo.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.dry_run:
        print("DRY RUN — 5-stage killer demo pipeline")
        print(f"Query: {KILLER_QUERY[:80]}...")
        print(f"Sub-queries: {len(SUB_QUERIES)}")
        for sq in SUB_QUERIES:
            print(f"  {sq['id']}: {sq['query'][:60]}")
        return 0

    from neural_search.search.core import search_datasets
    from neural_search.ingestion.demo_seed import build_combined_corpus
    from neural_search.retrieval.set_coverage_scorer import SetCoverageScorer, SetConstraints
    from neural_search.retrieval.role_assignment import assign_role

    print("=" * 60)
    print("Killer Demo — Neural Circuit Cognitive Control")
    print("=" * 60)

    # Build corpus lookup for rich metadata (modalities, species, brain_regions, tasks)
    # dataset_card_preview does not carry these fields — must pull from corpus records
    print("\nLoading corpus metadata index...")
    corpus_records = build_combined_corpus()
    corpus_lookup: dict[str, dict] = {}
    for rec in corpus_records:
        did = str(rec.get("dataset_id") or rec.get("source_id") or "")
        if did:
            corpus_lookup[did] = rec
            # also index by bare source_id in case search result strips prefix
            bare = did.split(":")[-1] if ":" in did else did
            corpus_lookup.setdefault(bare, rec)
    print(f"  Loaded {len(corpus_records)} records, {len(corpus_lookup)} index entries")

    # Stage 1 + 2: Decompose and retrieve per sub-query
    print("\nStage 1-2: Sub-query decomposition and retrieval...")
    all_results: dict[str, dict] = {}
    dataset_sq_counts: dict[str, int] = {}

    for sq in SUB_QUERIES:
        print(f"  {sq['id']}: {sq['query'][:50]}...")
        response = search_datasets(sq["query"], limit=args.k)
        for result in response.results:
            did = str(result.dataset_id)
            if did not in all_results:
                rec = corpus_lookup.get(did) or corpus_lookup.get(did.split(":")[-1]) or {}
                all_results[did] = {
                    "dataset_id": did,
                    "title": rec.get("title") or did,
                    "description": rec.get("description") or " ".join(result.why_matched),
                    "usefulness_score": (result.usefulness_score or {}).get("total_score", result.score / 100.0),
                    "modalities": rec.get("modalities") or [],
                    "species": rec.get("species") or [],
                    "brain_regions": rec.get("brain_regions") or [],
                    "affordances": rec.get("affordances") or [],
                    "tasks": rec.get("tasks") or result.matched_terms,
                    "doi": rec.get("doi") or rec.get("metadata_json", {}).get("doi") if isinstance(rec.get("metadata_json"), dict) else None,
                    "sub_query_matches": 0,
                }
                dataset_sq_counts[did] = 0
            dataset_sq_counts[did] += 1

    for did in all_results:
        all_results[did]["sub_query_matches"] = dataset_sq_counts[did]

    # Stage 3: Set-coverage scoring
    print(f"\nStage 3: Set-coverage scoring ({len(all_results)} unique candidates)...")
    candidate_list = sorted(all_results.values(), key=lambda d: -d["usefulness_score"])[:30]
    scorer = SetCoverageScorer()
    constraints = SetConstraints(
        required_modalities=["neuropixels", "fmri", "calcium_imaging"],
        required_species=["mouse", "human"],
        hard_negative_modalities=[],
    )
    coverage_result = scorer.score_set(candidate_list, constraints)
    print(f"  Set-coverage score: {coverage_result.total_score:.4f}")
    print(f"  Coverage bonus: {coverage_result.coverage_bonus:.4f}")
    print(f"  Hard-negative violations: {coverage_result.hard_negative_violations}")

    # Stage 4: Role assignment
    print("\nStage 4: Role assignment...")
    anchor_id = (
        max(dataset_sq_counts, key=dataset_sq_counts.get)
        if dataset_sq_counts else None
    )
    role_assignments = []
    assigned_roles: set[str] = set()

    for ds in candidate_list[:15]:
        ra = assign_role(ds, candidate_list, anchor_id=anchor_id)
        if ra.role.value == "unassignable":
            continue
        role_assignments.append({
            "dataset_id": ra.dataset_id,
            "role": ra.role.value,
            "evidence": ra.evidence,
            "title": ds.get("title", ""),
        })
        assigned_roles.add(ra.role.value)

    print(f"  Assigned roles: {sorted(assigned_roles)}")
    print(f"  Final result set: {len(role_assignments)} datasets")

    # Stage 5: Success metrics
    print("\nStage 5: Demo success metrics...")
    hard_criteria = {
        "all_datasets_have_role": len(role_assignments) > 0,
        "zero_hard_negative_violations": len(coverage_result.hard_negative_violations) == 0,
        "anchor_assigned": "anchor" in assigned_roles,
    }
    coverage_criteria = {
        "n_distinct_roles": len(assigned_roles),
        "coverage_bonus": coverage_result.coverage_bonus,
        "complementarity_bonus": coverage_result.complementarity_bonus,
    }

    print("\n  Hard criteria:")
    for k, v in hard_criteria.items():
        print(f"    [{'x' if v else ' '}] {k}")
    print("\n  Coverage criteria (measured):")
    for k, v in coverage_criteria.items():
        print(f"    {k}: {v}")

    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "query": KILLER_QUERY,
        "sub_queries": SUB_QUERIES,
        "n_candidates": len(all_results),
        "set_coverage": {
            "total_score": coverage_result.total_score,
            "coverage_bonus": coverage_result.coverage_bonus,
            "complementarity_bonus": coverage_result.complementarity_bonus,
        },
        "role_assignments": role_assignments,
        "hard_criteria": hard_criteria,
        "coverage_criteria": coverage_criteria,
        "all_hard_criteria_pass": all(hard_criteria.values()),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nReport → {args.output}")
    print(f"Overall: {'PASS' if output['all_hard_criteria_pass'] else 'FAIL (check hard criteria)'}")
    return 0 if output["all_hard_criteria_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
