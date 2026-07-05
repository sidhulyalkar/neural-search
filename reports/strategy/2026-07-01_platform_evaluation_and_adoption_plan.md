# Neural Search Platform Evaluation and Adoption Plan

**Date:** 2026-07-01  
**Scope:** Current repository state, technical readiness, scientific credibility, and adoption plan for users dealing with very large integrated neuroscience knowledge graphs.

## 1. Executive Assessment

Neural Search is no longer just a search demo. It is an emerging research-object retrieval platform for neuroscience datasets, papers, findings, claims, affordances, and graph-backed explanations. The strongest current asset is the breadth of infrastructure: ingestion adapters, ontology grounding, hybrid retrieval, typed relationship KG work, evaluation scripts, a frontend, a FastAPI backend, coverage ledgers, and a large corpus of reports.

The main blocker is not lack of architecture. The blocker is trust conversion: turning the large amount of silver, inferred, and diagnostic evidence into calibrated, human-validated product behavior that researchers can rely on. The platform should therefore prioritize validation, curation loops, reproducible artifacts, and focused researcher workflows before adding more KG schema or speculative graph layers.

North-star product statement:

> A researcher describes the experiment or analysis they want to run, and Neural Search returns reusable datasets with evidence-backed match explanations, analysis affordances, source provenance, quality gaps, and a runnable path from query to notebook.

## 2. Current State

### 2.1 Verified Local Signals

- Test discovery works in the local `.venv`: 2,918 tests collected.
- Retrieval smoke suite passed: 22 tests passed in `tests/smoke/test_end_to_end_retrieval.py`.
- API smoke test timed out after 180 seconds when run directly, likely because the default API path loads the full corpus and heavy search artifacts instead of a small demo fixture.
- Release summary regenerated successfully, but reports `release_ready=False`.
- Release checker expects older `demo_v05` and `real_v07` artifacts that are missing locally. This conflicts with the newer full-corpus manifest rather than necessarily indicating the platform has no usable corpus.

### 2.2 Source-of-Truth Manifest

The most current manifest is `reports/eval/current_artifact_manifest.json`.

Key platform state from that manifest:

- Corpus: `data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl`
- Corpus rows: 7,171
- Unique source IDs: 7,121
- Graph: `data/graph/neural_search_graph.real_corpus.json`
- Graph size: 7,593 nodes, 31,920 edges
- Cross-dataset edges: 2,957
- Gold qrels: 0 rows
- Field-state adjudicated qrels: 3 rows
- Canonical LLM-silver qrels: 13,654 pairs across 317 queries
- Vector index: only 625 current on-disk IDs; full 7,171-record rebuild pending
- Literature: 255,940 OpenAlex papers ingested; 190,279 operational findings estimated

### 2.3 Product Surface

The platform currently includes:

- FastAPI backend in `apps/api`.
- React/Vite frontend in `apps/web`.
- Pages for search, results, datasets, ontology, reports, evaluation, graph exploration, coverage, atlas, methods, disorders, and demo views.
- Search responses with score breakdowns, warnings, evidence snippets, neuro-judge snapshots, feedback, memory graph evidence, and linked papers.
- Dataset cards, notebooks, comparison endpoints, ingestion endpoints, coverage APIs, and knowledge graph APIs.

This is enough for an internal alpha and guided external demos. It is not yet ready for broad unsupervised public use.

## 3. What Works

### 3.1 Differentiated Retrieval Stack

The platform has lexical, structured, dense, hybrid, graph, typed-KG, and full retrieval variants. The canonical silver benchmark shows non-degenerate differentiation:

- `bm25`: NDCG@10 around 0.6566 in the reconciled manifest.
- `hybrid_rrf`: NDCG@10 around 0.6667.
- `hybrid_graph/full`: NDCG@10 around 0.6696.
- `dense_bge`: underperforms BM25 in the reported canonical run.

Caveat: these are LLM-silver diagnostics, not human-gold claims.

### 3.2 Rich Neuroscience Semantics

The codebase contains meaningful domain layers:

- Behavioral task ontology.
- Brain region index and atlas references.
- Modality, species, recording scale, event, task, and analysis affordance layers.
- Dataset-paper linking.
- Claim and finding extraction.
- Typed finding fields such as frequency bands, injury models, negation, species, and signal types.
- Relationship KG support and contradiction logic.
- Coverage ledgers and source-specific gap reports.

### 3.3 Honest Evaluation Culture

The reports are unusually candid. Existing strategy docs correctly avoid overclaiming graph improvements and repeatedly identify missing human labels as the main scientific blocker. This honesty should become a product virtue: the system should always show evidence tier, confidence, missingness, and whether a claim is human-reviewed, LLM-silver, inferred, or unreviewed.

## 4. What Is Fragile

### 4.1 Gold Evaluation Is Still Missing

The single most important gap is gold qrels. Current human labels are tiny relative to the ambition:

- Gold qrels: 0.
- Field-state adjudicated: 3.
- Human-labelled pairs reported in qrels progress: 13.
- Publishable target: 100 queries, 1,500 pairs, 2 annotators, agreement stats.

Without gold qrels, the platform can support internal development and guided demos, but it cannot support strong scientific claims.

### 4.2 Artifact Regimes Are Split

There are at least two artifact regimes:

- Older release checker regime: `demo_v05`, `real_v07`, hashing embeddings, release summary.
- Current full-corpus/eval regime: `full_corpus_v09`, full graph, canonical LLM-silver qrels, current artifact manifest.

This creates confusion for contributors and risk for demos. A user should not have to know which artifact tree is authoritative.

### 4.3 API Smoke Path Is Too Heavy

The API smoke test timed out locally. The test itself calls core API methods directly and performs a search, dataset card retrieval, comparison, and report load. The current default API behavior appears too expensive for a fast smoke test. This makes it harder to trust changes and harder to onboard contributors.

### 4.4 Graph Contribution Is Not Yet Strong Enough

Graph/typed KG signals are promising but not yet decisive:

- Full-corpus typed KG report shows tiny or no movement at current weights.
- Relationship edge quality shows helpful rates around coin-flip for several edge types.
- Graph improvements in canonical silver results are directionally positive but within uncertainty.

The lesson is not "remove the graph." The lesson is "use the graph carefully, calibrate it by edge family, and surface its evidence as explanation before treating it as a strong ranker."

### 4.5 Metadata Coverage Has Critical Holes

Coverage report highlights:

- Species value coverage: 72.8%.
- Brain-region value coverage: 48.1%.
- Modality value coverage: 81.9%.
- Task value coverage: 23.3%.
- Behavioral-event value coverage: 1.7%.
- Unknown state slots needing enrichment: 10,576.

Behavioral events, tasks, brain regions, and source-specific metadata gaps directly affect the platform's ability to answer "can I actually run this analysis?"

## 5. Who Benefits Most

The first users should be people with urgent dataset reuse and KG navigation needs, not casual search users.

### 5.1 Primary Early Users

1. Computational neuroscientists doing dataset reuse
   - Need: find datasets suitable for specific analyses such as Q-learning, decoding, event-aligned activity, latent dynamics, sleep staging, seizure detection, or connectivity.
   - Pain: archive search is keyword-oriented and does not expose analysis affordances.
   - Product promise: "Here are datasets that can support your analysis, with evidence and missingness."

2. Labs with large internal and public data portfolios
   - Need: understand what their datasets contain, where metadata is missing, what is reusable, and which papers connect to which data.
   - Pain: internal knowledge lives across archive metadata, papers, lab memory, spreadsheets, and notebooks.
   - Product promise: "A living map of your lab's research objects and evidence."

3. Data curators and archive teams
   - Need: identify metadata gaps, prioritize enrichment, improve discoverability, validate provenance.
   - Pain: coverage work is manual and reactive.
   - Product promise: "A curation dashboard that turns missingness into a ranked worklist."

4. Methods developers and benchmark builders
   - Need: find datasets matching precise modality, task, species, and signal requirements.
   - Pain: generic dataset search cannot distinguish reusable from merely related.
   - Product promise: "A benchmark substrate for methods that need real data."

5. KG and scientific AI researchers
   - Need: test whether typed, provenance-backed scientific KGs improve retrieval and reasoning.
   - Pain: many KGs are large but weakly typed or weakly validated.
   - Product promise: "A testbed for typed scientific retrieval, not generic GraphRAG."

### 5.2 Users To Defer

- Broad public users who expect complete coverage and production reliability.
- Clinicians making safety-critical decisions.
- Users who want a chatbot answer rather than traceable research objects.
- Users who need live, authenticated, multi-tenant workflows before the platform has auth, roles, and audit logs.

## 6. Product Strategy

### 6.1 Make The Core Workflow Unmistakable

The first screen and demo should center one loop:

1. Search by scientific intent.
2. Inspect ranked datasets.
3. See why each dataset matches or fails.
4. Check analysis affordances.
5. Open source archive and evidence.
6. Generate or open a starter notebook.
7. Leave feedback or adjudicate relevance.

Everything else should support that loop.

### 6.2 Package Three Alpha Experiences

1. Dataset Reuse Search
   - Query examples: "mouse reversal learning with fiber photometry for Q-learning", "human iEEG speech decoding", "Neuropixels visual decision task with choice labels".
   - Output: ranked datasets, why matched, missing metadata, source links, affordances, notebook starter.

2. Coverage and Curation Workbench
   - Audience: labs, archives, curators.
   - Output: source coverage, unknown slots, region/modality gaps, priority enrichment queue.

3. KG Evidence Explorer
   - Audience: KG/scientific AI collaborators.
   - Output: typed relationships, claim/finding provenance, contradictions, consensus, relationship quality diagnostics.

Do not market all pages equally. Lead with Dataset Reuse Search. Use the other two as power-user modes.

## 7. Technical Roadmap

### Phase 0: Reconcile The Platform State (1 week)

Goal: remove confusion and make the current platform runnable and inspectable.

Tasks:

- Declare `reports/eval/current_artifact_manifest.json` the current source of truth.
- Update or replace `neural_search.release.check` so it understands `full_corpus_v09` and the full graph, not only old `demo_v05` and `real_v07` artifacts.
- Add a `make doctor` command that reports:
  - Python environment status.
  - Required dependencies.
  - Current corpus path and row count.
  - Graph path and node/edge counts.
  - Qrels counts by tier.
  - Whether API smoke mode can run.
- Add `NEURAL_SEARCH_DEMO_MODE=1` or fixture monkeypatching to `tests/test_api_smoke.py` so API smoke completes in under 15 seconds.
- Add a "current artifact regime" section to the README.

Exit criteria:

- `make doctor` is green on a fresh dev checkout.
- API smoke test passes under 15 seconds.
- Release checker and artifact manifest agree on current corpus/graph availability.

### Phase 1: Trust Foundation (2 weeks)

Goal: produce the first credible human-validated retrieval numbers.

Tasks:

- Build a stratified annotation set from existing 317 canonical queries:
  - 25 strict lookup queries.
  - 25 pipeline reuse queries.
  - 25 reanalysis feasibility queries.
  - 25 exploration or cross-dataset comparison queries.
- Label at least 300 pairs with one human annotator.
- Dual-label at least 75 of those pairs with a second annotator.
- Produce gold qrels and agreement stats.
- Run BM25, hybrid RRF, hybrid graph, typed KG, and full against gold subset.
- Publish a small "Tier A alpha benchmark" report that clearly separates gold from silver.

Exit criteria:

- `artifacts/qrels_gold.jsonl` is non-empty.
- At least 300 human labels exist.
- Agreement stats exist.
- A gold-subset NDCG/MRR/P@5 report exists with confidence caveats.

### Phase 2: High-Value Search Fixes (2 weeks)

Goal: make the platform feel useful to a real researcher.

Tasks:

- Prioritize strict lookup and source-ID handling:
  - DANDI IDs.
  - OpenNeuro `ds*` IDs.
  - known dataset names like Steinmetz, IBL, Allen Visual Coding.
- Ensure source archive URLs are visible on result cards and dataset pages.
- Improve hard-negative handling for explicit exclusions such as "NOT EEG", "not fMRI", "not human".
- Add "why not a match" explanations for top partial results.
- Add a query intent header: strict lookup, pipeline reuse, replication, exploration, reanalysis feasibility, cross-dataset comparison.
- Add result evidence tier labels:
  - Human-reviewed.
  - Source-declared.
  - File-derived.
  - LLM-silver.
  - Inferred.
  - Unreviewed.

Exit criteria:

- Top 20 known-item lookups return the intended dataset in rank 1-3.
- Hard-negative violation rate is measured and trending down on the gold subset.
- Every top result exposes source URL, evidence tier, and missingness.

### Phase 3: Researcher Alpha (3-4 weeks)

Goal: put the system into the hands of 5-10 people who can benefit immediately.

Pilot groups:

- 2 computational neuroscience labs.
- 1 archive/curation team.
- 1 methods/benchmark group.
- 1 KG/scientific AI collaborator.

Pilot package:

- Hosted or local guided demo.
- 10 curated demo queries.
- 3 runnable notebooks tied to real datasets.
- Feedback capture in the frontend.
- Short onboarding document: "What Neural Search is good at, what it is not yet allowed to claim."

Success metrics:

- 70% of pilot users find at least one dataset worth opening.
- 50% produce at least one concrete correction, label, or metadata improvement.
- At least 30 new human relevance labels are collected from pilot usage.
- At least 3 notebooks run successfully from search result to first plot.

### Phase 4: Active Knowledge And Understanding Loop (ongoing)

Goal: make the platform learn without losing epistemic discipline.

Core loop:

1. New corpus items enter ingestion.
2. Metadata is normalized and assigned evidence tiers.
3. Search and KG indices rebuild with manifest hashes.
4. Eval runs detect regressions.
5. Human feedback and annotations update qrels and QA state.
6. Coverage ledgers reprioritize enrichment.
7. Reports update claim status.

Required systems:

- Artifact versioning with corpus hash and graph hash.
- Label provenance: annotator, timestamp, rubric, query intent.
- Edge lifecycle: unreviewed, accepted, deprecated, contradicted.
- Feedback audit queue from frontend.
- "Active knowledge" dashboard showing:
  - What changed since last build.
  - What evidence improved.
  - What claims weakened.
  - What source gaps matter most.

## 8. KG Strategy

### 8.1 What To Keep Building

Build graph features only when they feed one of three outputs:

- Better ranking on validated queries.
- Better explanations for result trust.
- Better curation worklists.

High-value KG work:

- Edge-family calibration: measure which edge types help or harm ranking.
- Typed contradiction and consensus review.
- Dataset-can-test-finding bridge.
- Evidence-span offsets for claims and edges.
- Provenance and evidence-tier display in dataset cards.

### 8.2 What To Pause

Pause:

- New graph schema growth without an eval target.
- Speculative hidden relationship mining.
- Spectral phenotype search wiring unless benchmark queries demand it.
- More LLM-silver expansion before gold calibration.
- Graph database migration unless file-backed graph loading becomes a measured bottleneck.

### 8.3 How To Handle Large Integrated KGs

For extremely large, integrated, complex KGs, the platform should avoid "one graph to rule them all" at query time. Use layered access:

- Retrieval index: fast candidate generation.
- Typed KG feature index: compact edge-derived features for ranking.
- Evidence graph: provenance and explanation after candidate selection.
- Coverage graph: gap analysis and curation planning.
- Literature graph: claim/finding exploration and contradiction analysis.

This keeps search fluid while preserving breadth.

## 9. Adoption Plan

### 9.1 Positioning

Do not position Neural Search as "a neuroscience chatbot" or "a better KG viewer."

Position it as:

> Experiment-aware dataset discovery for neuroscience reuse, backed by provenance, analysis affordances, and active evaluation.

Short pitch:

> Neural Search helps researchers find datasets they can actually use, not just papers or archive records that share keywords.

### 9.2 First Outreach Targets

Start with warm, concrete use cases:

- DANDI/NWB users looking for reusable neurophysiology datasets.
- OpenNeuro users working with BIDS EEG/iEEG/fMRI task datasets.
- Labs doing reversal learning, decision making, speech/iEEG, sleep, seizure, BCI, and Neuropixels work.
- Data curators responsible for metadata completeness.
- Benchmark developers needing real datasets for method evaluation.

### 9.3 Pilot Offer

Offer a "dataset discovery audit" rather than a generic demo:

- User brings 3 research questions.
- Neural Search runs each query.
- Together you inspect top results, evidence, and missingness.
- User labels relevance and identifies what would make results useful.
- The platform records labels and coverage gaps.

This creates value even when search is imperfect.

### 9.4 Public Artifacts

Before broader release, prepare:

- 3-minute demo video.
- 10-query gallery with expected behavior.
- 3 real notebooks.
- One-page limitations note.
- Human-label progress dashboard.
- Reproducible artifact manifest.
- "How to cite / how to contribute labels" guide.

## 10. Success Metrics

### Scientific Credibility

- Gold qrels: 300 pairs in alpha, 1,500 for publishable benchmark.
- Dual annotation: at least 25% of gold pairs.
- Agreement: reported with per-intent breakdown.
- Hard-negative violation rate: under 5% on gold.
- Known-item lookup: intended dataset rank 1-3 for curated known-item set.

### Product Utility

- Time from query to source dataset: under 60 seconds.
- Time from query to first notebook plot: under 10 minutes for curated datasets.
- Pilot task success: 70% of pilot users find at least one useful dataset.
- Feedback conversion: at least 30 labels per pilot cohort.

### Platform Health

- API smoke under 15 seconds.
- Current manifest generated by script, not hand-maintained.
- Full corpus vector index covers all current source IDs.
- Release checker agrees with manifest.
- CI separates quick smoke, full eval, and heavy artifact builds.

## 11. Immediate Next Actions

1. Fix local platform clarity:
   - Update release checker or manifest flow.
   - Add `make doctor`.
   - Make API smoke use demo mode.

2. Start gold label campaign:
   - Select 100-query stratified subset.
   - Label first 300 pairs.
   - Dual-label 75 pairs.

3. Harden the researcher loop:
   - Source URLs everywhere.
   - Evidence tiers everywhere.
   - Known-item lookup tests.
   - Hard-negative tests for explicit exclusions.

4. Package the alpha:
   - 10 query gallery.
   - 3 notebooks.
   - 5 pilot users.
   - Feedback-to-qrels path.

5. Calibrate KG contributions:
   - Edge-family helpfulness by query intent.
   - Typed KG ablation on gold subset.
   - Relationship edge review before further KG expansion.

## 12. Bottom Line

This project has enough technical breadth. The next leap is not another layer of graph machinery; it is converting breadth into active, trusted understanding. That means a system that knows what it knows, shows why, admits what is missing, learns from expert feedback, and gets faster at helping researchers move from a question to a reusable dataset.

The right next move is a trust-centered alpha: reconcile artifacts, make smoke tests fast, gather gold labels, harden source/provenance display, and put the guided workflow in front of the researchers and curators already feeling the pain.
