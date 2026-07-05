# Playbook: file-validation-runner

**Registry entry:** `file-validation-runner` in `artifacts/agents/registry.yaml`
**Gate type:** `soft_review` (network-dependent; a spike in errors should pause
scale-up next run, not be silently retried at the same or larger scope).

## Why this exists

`dataset_old_dataset_new_method_candidate` and `dataset_reanalysis_bridge_dataset`
edges are heuristic/evidence-backed *candidates* -- `requires_human_review=True`,
`evidence_tier` at most `heuristic_candidate`/`evidence_backed_bridge`. Live file
validation (`neural_search/graph/dandi_nwb_validator.py`,
`neural_search/graph/openneuro_bids_validator.py`) is the only way to move a
candidate up to `file_validated`: it reads real NWB/BIDS metadata (electrode
tables, task/session counts) directly from DANDI/OpenNeuro without downloading
full files, and checks it against the suggested analysis family's core
requirement (`neural_search/graph/file_validation_requirements.py`). As of
2026-07-04 this has been run at `--top-n 300` (up from an initial `--top-n 50`),
confirming 172+ datasets live -- see `reports/top_suggestions_validation_report.md`.

## When to run

- Weekly (registry `schedule_cadence: weekly`), or on demand after a large batch
  of new candidate edges is generated (e.g. methodology registry coverage grows).

## Steps

1. Read the agent context brief (`artifacts/agents/context_brief.md`, regenerate
   via `python -m neural_search.agents.context_brief` if stale) for the current
   `dataset_old_dataset_new_method_candidate` / `dataset_reanalysis_bridge_dataset`
   edge counts and the last run's scale (`--top-n`).
2. Check `neural_search.agents.ledger.last_run_for_agent("file-validation-runner")`
   for the previous run's errored-count and `--top-n`. **Do not scale up
   `--top-n` past the previous run if the previous run's errored/validator-none
   ratio was elevated** -- per the registry's own `gate_note`, a spike should
   pause escalation, not be silently retried at the same or larger scale.
3. Run:
   ```
   python scripts/validate_top_reanalysis_suggestions.py --top-n <N>
   ```
   Start at the same `--top-n` as the last successful run; only increase it if
   the last run's error rate was low and DANDI/OpenNeuro's share of candidate
   datasets in the current top-N still leaves room to validate more (sources
   without a live validator -- figshare, gin, crcns, etc. -- are recorded as
   `validator: "none"`, not an error, and don't count against the error rate).
4. Read the two outputs it writes:
   - `artifacts/validation/top_suggestions_file_validation.jsonl` (per-dataset
     confirmed/errored/none rows)
   - `reports/top_suggestions_validation_report.md` (human-readable summary)
5. Confirm `neural_search/graph/evidence_tier_upgrader.py`'s
   `apply_file_validation_upgrades()` step is wired into
   `scripts/build_real_corpus_graph.py` (it should already be, per the
   `file_validation_and_evidence_tiers` project memory) -- if a rebuild is
   needed to apply the new upgrades, that graph rebuild must go through
   **benchmark-gatekeeper** before being considered committed, same as any
   other graph-modifying agent.
6. Append a ledger entry with `outcome: "ok"` (or `"error"` if the run itself
   crashed) and `findings` listing: datasets validated, confirmed count,
   errored count, `validator: "none"` count, and the `--top-n` used.
7. Write a note under `obsidian_vault/11_Agent_Runs/` summarizing the result.

## What "done" looks like

A ledger row + Obsidian note exist. The confirmed/errored/none counts are
recorded precisely (not summarized as "some passed"). If a graph rebuild was
needed to apply upgrades, benchmark-gatekeeper has confirmed no NDCG@10
regression.
