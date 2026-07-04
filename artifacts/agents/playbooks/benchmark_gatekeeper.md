# Playbook: benchmark-gatekeeper

**Registry entry:** `benchmark-gatekeeper` in `artifacts/agents/registry.yaml`
**Gate type:** `hard_block` -- no other graph-modifying agent's output counts
as committed until this passes.

## Why this exists

This exact check (re-run the ablation ladder, compare to the last known
NDCG@10) has caught three real regressions in this project's history: two
from a new densely-populated edge type dominating the `graph_degree`
feature, one from a score weight (`linked_paper`) that had been configured
speculatively years before the data it scored ever existed. All three
looked completely fine by every other measure (zero dangling edges, clean
build, correct edge counts) until this specific check ran. See the
`ablation_results` project memory for the full history -- read it before
assuming "no dangling edges" is sufficient; it is necessary but not
sufficient.

## When to run

- Immediately after any `scripts/build_real_corpus_graph.py` run that
  changed anything (new builder wired in, new artifact consumed, edge
  weight tweaked).
- Do not skip this because the change "only adds a node property" or
  "only touches an unrelated edge type" -- the three historical regressions
  above all looked exactly that innocuous beforehand.

## Steps

1. Confirm the graph was actually rebuilt: check the mtime of
   `data/graph/neural_search_graph.real_corpus.json` is newer than the
   change you're gating.
2. Run:
   ```
   python scripts/eval/run_ablation_ladder.py --skip-rungs bm25 bm25_structured dense_bge hybrid_rrf
   ```
   (Skipping the non-graph rungs is safe here -- they can't be affected by
   a KG-only change, and skipping them cuts a ~10 minute run to a few
   minutes.) Full run (no `--skip-rungs`) if you have reason to suspect a
   non-graph component also changed.
3. Read `reports/eval/ablation_ladder_report.partial.json`'s `hybrid_graph`
   and `full` rung `ndcg@10` values.
4. Compare against the last recorded baseline in the `ablation_results`
   project memory (or the most recent `benchmark-gatekeeper` ledger row via
   `neural_search.agents.ledger.last_run_for_agent("benchmark-gatekeeper")`
   if more recent).
5. **If unchanged (within floating-point noise, effectively exact):** outcome
   `ok`. Nothing further needed.
6. **If regressed:** outcome `regression`. Do not guess the cause from what
   worked last time -- isolate it:
   - What edge types changed in count since the last good graph? (compare
     `reports/eval/current_artifact_manifest.json`'s edge_type_counts,
     regenerate via `python scripts/build_artifact_manifest.py` if stale)
   - Is a newly-populated edge type missing from
     `neural_search/graph/search_features.py`'s `graph_degree` exclusion
     list (`_DEGREE_EXCLUDED_EDGE_TYPES` or equivalent)?
   - Is a score weight configured for data that didn't exist until this
     change (search `search_features.py` / scoring config for weights on
     the new edge/node type)?
   - Monkeypatch the suspected weight to 0 or add the suspected exclusion,
     re-run the ladder, confirm the exact baseline number returns before
     committing the fix -- isolate, don't assume.
7. Append a ledger entry (`neural_search.agents.ledger.append_ledger_entry`)
   with `outcome`, `eval_delta` (before/after NDCG@10 per rung), and
   `gate_result` (`"pass"` or `"fail: <cause>"`).
8. Write a note under `obsidian_vault/11_Agent_Runs/` summarizing the
   result -- especially the cause, if this was a regression. Future
   sessions (human or agent) should not have to re-derive a cause that's
   already been found once.

## What "done" looks like

A ledger row + Obsidian note exist, and the graph's NDCG@10 is either
confirmed unchanged or restored to baseline with the cause documented.
"Ran the script" is not done; "confirmed the number" is done.
