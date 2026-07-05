---
run_id: 2026-07-04_reanalysis_insight_synthesis
agent: reanalysis-insight-synthesizer
outcome: ok
created: '2026-07-04'
tags:
- agent-run
- reanalysis-insight-synthesizer
- reuse
type: agent_run
---

## First formal run of this playbook

New agent this session: `reanalysis-insight-synthesizer`
(`artifacts/agents/playbooks/reanalysis_insight_synthesizer.md`), registered
in `artifacts/agents/registry.yaml` with `depends_on: [kg-connectivity-auditor,
benchmark-gatekeeper]`. Ran against the production graph
(12,748 nodes / 149,998 edges) already confirmed non-regressed by
`benchmark-gatekeeper` this cycle (NDCG@10 0.8594, unchanged). Full output:
`reports/reanalysis_insight_report.md`.

## Headline: top reuse opportunities (see full report for all rows)

**Evidence-backed** (`dataset_reanalysis_bridge_dataset`, a similar dataset was
actually analyzed with the named method per a real paper):
1. NeuroMorpho: Acsady (13 neurons) — EEG Analysis, precedent: NeuroMorpho
   Timofeev (0.36 confidence)
2. Zhang: Hippocampal-neocortical dialogue during memory consolidation —
   Calcium Imaging Analysis, precedent: barrel-cortex whisker-locomotion
   study (0.342)
3. cai-2 (jRGECO1a/jRCaMP1a characterization) — Coherence, precedent:
   layer-specific V1 physiology study (0.342)

**Genuinely unexplored, high confidence** (zero linked papers,
confidence >= 0.85):
1. eeg-1 (64-channel human scalp EEG, spontaneous thought) — FFT /
   time_frequency (0.9)
2. Allen Cell Types Database Electrophysiology — DCM / connectivity (0.85)
3. Allen Visual Coding Neuropixels sessions — burst_analysis /
   spike_train_analysis (0.85)

## A ranking bug found and fixed in the same run

Naive top-N-by-confidence degenerated badly: the raw top 15 evidence-backed
rows were 15 near-duplicate entries, all the same precedent dataset
(NeuroMorpho: Timofeev) at the identical 0.36 confidence value — a real
artifact of how `confidence = paper_confidence * method_confidence *
sim_edge.confidence` produces ties from a small set of discrete inputs, not a
data quality problem. Fixed by capping rows per precedent dataset (max 2) and
per technique (max 3) in `scripts/generate_reanalysis_insight_report.py` — the
unexplored-candidates list only filled 9/15 slots after capping, which is an
honest signal-breadth finding, not a bug to paper over.

## A real bottleneck identified precisely, not just re-confirmed

Growing paper-dataset linkage from 403 to 2,510 real matches across 5 sources
(this project's literature-source-expansion work, 2026-07-02 through
2026-07-03) did **not** unlock more reanalysis-bridge evidence — re-measured
this run: DOI-resolving DataCite/Crossref/PubMed matches to an OpenAlex ID
(new `load_dataset_paper_matches_multi_source()` in
`neural_search/graph/reanalysis_bridge_builder.py`) finds only 8 additional
usable matches and 0 additional bridge edges. The actual bottleneck:
`artifacts/ner/ner_kg.jsonl`'s method-mention extraction has only ever run
against OpenAlex-ingested paper text, never against the Crossref/PubMed/
DataCite paper records that make up most of the newer linkage growth. This is
now on record precisely (`reports/reanalysis_insight_report.md`) instead of
being re-guessed at by a future session.

## What shipped alongside this run

- `neural_search/agents/context_brief.py` — single current-state briefing
  (production scale, NDCG baseline, per-agent last-run status, open
  questions from prior runs) every playbook should read first, generated via
  `python -m neural_search.agents.context_brief` into
  `artifacts/agents/context_brief.md`.
- Two previously-missing playbooks filled in: `file_validation_runner.md`,
  `literature_linker.md` — all 5 registered agents now have a runnable
  playbook (`kg-connectivity-auditor` and `benchmark-gatekeeper` already did).
- `depends_on` field added to the registry for formal agent sequencing.

## Ledger

See `artifacts/agents/ledger.jsonl`, `agent: "reanalysis-insight-synthesizer"`,
`started_at: "2026-07-04T21:30:00+00:00"`.
