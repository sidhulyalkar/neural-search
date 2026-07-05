# Playbook: reanalysis-insight-synthesizer

**Registry entry:** `reanalysis-insight-synthesizer` in `artifacts/agents/registry.yaml`
**Gate type:** `none` (read-only synthesis over edges already live in the
production graph; proposes nothing that changes the graph itself).

## Why this exists

`reports/reanalysis_candidates_report.md` and `reports/methodology_coverage_report.md`
report raw counts (59,126 heuristic candidates, 2,517 evidence-backed bridge
edges, coverage gaps) but never rank them into an actionable "what to look at
first" list, and never cross-check whether a load-bearing assumption (e.g.
"more paper linkage means more reanalysis-bridge evidence") still holds as the
underlying data grows. This playbook does both: it ranks the existing signal
by confidence with diversity caps (a naive top-N by confidence degenerates
into near-duplicate rows from one highly-connected precedent dataset or one
dominant technique -- confirmed empirically 2026-07-04, see
`scripts/generate_reanalysis_insight_report.py`'s `MAX_ROWS_PER_PRECEDENT`/
`MAX_ROWS_PER_TECHNIQUE`), and it re-tests the paper-linkage-coverage
assumption against the current graph every run rather than trusting a stale
prior conclusion.

## When to run

- After any `literature-linker` run (paper-linkage coverage changed).
- After any `file-validation-runner` run (evidence tiers changed).
- After a `method_registry.yaml` update (candidate-eligible analysis families
  changed).
- Otherwise weekly, chained after `kg-connectivity-auditor` and
  `benchmark-gatekeeper` in the scheduled routine, since it's cheap
  (single graph load, no network) and benefits from running against a
  graph already confirmed non-regressed by benchmark-gatekeeper that cycle.

## Steps

1. Read the agent context brief (`artifacts/agents/context_brief.md`,
   regenerate via `python -m neural_search.agents.context_brief` if its
   `manifest_generated_at` predates the last graph rebuild) for current
   `dataset_old_dataset_new_method_candidate` / `dataset_reanalysis_bridge_dataset`
   / `dataset_reinterpretation_candidate` edge counts and the last
   `benchmark-gatekeeper` outcome -- **do not run this against a graph that
   benchmark-gatekeeper hasn't cleared**, since a regressed graph's edge
   counts may reflect a bug, not real signal.
2. Run:
   ```
   python scripts/generate_reanalysis_insight_report.py
   ```
   This loads the production graph once (`data/graph/neural_search_graph.real_corpus.json`,
   ~5s), and writes `reports/reanalysis_insight_report.md` with:
   - Top evidence-backed reuse opportunities (`dataset_reanalysis_bridge_dataset`,
     confidence-ranked, capped per precedent dataset).
   - Top high-confidence, genuinely-unexplored candidates (`dataset_old_dataset_new_method_candidate`,
     confidence >= 0.85, zero linked papers, capped per technique).
   - A live re-check of whether growing paper-linkage coverage has changed
     the `dataset_reinterpretation_candidate` count (it re-measures the
     multi-source DOI-resolution delta from
     `neural_search.graph.reanalysis_bridge_builder.load_dataset_paper_matches_multi_source()`
     every run -- if a future NER-extraction expansion changes this, this
     report is what will surface it, not a note someone has to remember to
     re-check by hand).
   - A cross-reference to the current methodology-registry gaps limiting
     candidate generation.
3. Read the generated report. If either ranked list is unexpectedly short
   (fewer rows than requested, e.g. exhausted by diversity caps) that is a
   genuine finding about signal breadth, not a bug -- report it as such
   rather than treating a short list as an error.
4. If the reinterpretation-candidate re-check shows any change from the
   0-edges baseline (see `reports/reanalysis_insight_report.md`'s history via
   git blame / prior ledger rows), that is a significant finding: flag it
   prominently, since it means literature-source-expansion work finally
   surfaced a real reinterpretation opportunity.
5. Append a ledger entry with `outcome: "ok"` and `findings` listing: edge
   counts synthesized, size of each ranked list produced, and whether the
   reinterpretation-candidate re-check changed from the last run.
6. Write a note under `obsidian_vault/11_Agent_Runs/` linking to
   `reports/reanalysis_insight_report.md` and summarizing the top 2-3 ranked
   opportunities inline (so a human scanning Obsidian doesn't have to open
   the full report to get the headline).

## What "done" looks like

A ledger row + Obsidian note + `reports/reanalysis_insight_report.md` exist.
The ranked lists are genuinely diverse (not dominated by one precedent or
technique), and the reinterpretation-candidate assumption has been re-tested
against current data, not assumed unchanged from a prior session's finding.
