# Playbook: literature-linker

**Registry entry:** `literature-linker` in `artifacts/agents/registry.yaml`
**Gate type:** `soft_review` (rate-limited external APIs; `trigger: manual`
only -- not safe to run unattended on a fixed schedule yet).

## Why this exists

Paper-dataset linkage is the precondition for the strongest reuse signals in
the graph (`dataset_reanalysis_bridge_dataset`, `dataset_reinterpretation_candidate`,
`paper` node trust signals). As of 2026-07-04, combined real linkage across 5
sources (OpenAlex, DataCite, Crossref, PubMed, bioRxiv) is 2,510/7,171 (35%),
up from 403/7,171 (5.6%) before this project's literature-source-expansion
work. Semantic Scholar remains blocked (HTTP 429 on the unauthenticated tier,
no `SEMANTIC_SCHOLAR_API_KEY` configured) -- a real external constraint, not a
code defect.

**Important, non-obvious limitation found 2026-07-04**: growing per-source
linkage does *not* automatically grow the reanalysis-bridge/reinterpretation
signal. `neural_search/graph/reanalysis_bridge_builder.py` and
`reinterpretation_candidate_builder.py` need a paper's OpenAlex ID (to join
against `ner_kg.jsonl`'s method-mention extraction, which has only ever run
against OpenAlex-ingested paper text). A DataCite/Crossref/PubMed match with no
corresponding OpenAlex ID can't feed those two builders even after
`neural_search.graph.reanalysis_bridge_builder.load_dataset_paper_matches_multi_source()`
resolves it by DOI -- measured result: +8 matches, +0 bridge edges (see
`reports/reanalysis_insight_report.md`). **Running this playbook grows overall
paper-link coverage and paper trust signals (retraction/correction status,
API-surfaced evidence tiers) — it does not by itself grow reanalysis-bridge
edges.** That needs NER extraction against non-OpenAlex paper sources, which is
separate, unscoped future work.

## When to run

- Manually, when corpus growth or a new source becomes available. Do not
  schedule this unattended (registry `trigger: manual`) until per-source
  budget-aware backoff exists — see the OpenAlex budget-wall incident
  (`file_validation_and_evidence_tiers` / `literature_source_expansion`
  project memory): a prior unattended long-running pass silently converted
  budget-exhausted requests into false `not_found` results before the
  `TransientLookupError` fix.

## Steps

1. Read the agent context brief (`artifacts/agents/context_brief.md`) for the
   current combined real-match count and per-source breakdown (from
   `reports/eval/current_artifact_manifest.json`'s `paper_dataset_links.by_source`).
2. Identify which sources have unprocessed corpus records:
   - OpenAlex: `neural_search.literature.linking.link_corpus_to_literature`
     (live, rate/budget-limited) vs `link_corpus_to_local_literature` (local
     tier-1 index only, no network, always safe to re-run).
   - DataCite: `neural_search.literature.datacite.link_corpus_to_datacite`
     (dataset-DOI-only; 1,147/7,171 DANDI/OpenNeuro records have no `doi`
     field at all -- not this layer's gap, a corpus-normalization gap).
   - Crossref: `neural_search.literature.crossref.link_corpus_to_crossref`
     (full-corpus DOI-exact + title-fuzzy).
   - PubMed/bioRxiv: `neural_search.literature.pubmed.link_corpus_to_pubmed`.
   - Semantic Scholar: skip -- confirmed blocked without an API key.
3. Run only the sources with real unprocessed volume and a working budget.
   Every source module raises `neural_search.literature.api_client.TransientLookupError`
   on budget/rate exhaustion or transport failure instead of silently returning
   `not_found` -- if a run aborts with this error, **stop, record how far it
   got, do not retry immediately at the same scale** (matches the honest
   partial-progress handling already built into these modules).
4. After any new linking run, regenerate the combined view (does NOT overwrite
   the OpenAlex-shaped legacy file consumers depend on):
   ```
   python -c "from neural_search.literature.merge_links import merge_link_sources; print(merge_link_sources())"
   ```
5. Run retraction/correction checks on any newly-found DOIs:
   ```
   python scripts/check_paper_retraction_status.py
   ```
6. Regenerate the manifest so the new coverage numbers are visible everywhere
   that reads it:
   ```
   python scripts/build_artifact_manifest.py
   ```
7. If new paper nodes reached production (via `paper_node_builder.py`'s wiring
   into `scripts/build_real_corpus_graph.py`), that graph rebuild must go
   through **benchmark-gatekeeper** before being considered committed.
8. Append a ledger entry with `outcome`, and `findings` listing: new real
   matches by source, combined coverage before/after, any
   `TransientLookupError` aborts and where they stopped.
9. Write a note under `obsidian_vault/11_Agent_Runs/` summarizing the result.

## What "done" looks like

A ledger row + Obsidian note exist. Coverage numbers are exact (not
estimated). If a budget/rate wall was hit, that is recorded as the stopping
reason, not silently absorbed into a lower "not_found" count.
