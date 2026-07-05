# Claude Execution Prompt: Whitepaper, Repo State, Validation, and Frontend Upgrade

Date: 2026-06-20

Use this as a direct prompt for Claude while qrel labeling is running on the Mac and paper/finding extraction is running on the PC/GPU.

## Prompt To Paste Into Claude

You are working in the `neural-search` repository. Your job is to harden the current whitepaper, reconcile repo/artifact state, and upgrade the frontend into a compelling, publication/demo-grade research interface without disrupting active qrel labeling or paper extraction jobs.

Important constraints:

- Do not edit `.env.local`.
- Do not delete, overwrite, truncate, or regenerate active long-running outputs unless explicitly asked:
  - `artifacts/eval/llm_judgments.jsonl`
  - `artifacts/literature/findings_tier1_ollama.jsonl`
  - `artifacts/literature/findings_tier1_ollama.checkpoint.json`
- Avoid heavy rebuilds by default. Do not run full embedding rebuilds, full extraction, or full benchmark jobs unless the user explicitly says to.
- Prefer lightweight audits, focused tests, UI fixes, docs fixes, and small scripts that reconcile manifests.
- Preserve unrelated untracked files and user work.

### Current Observed Repo State

Actual local qrels counts observed:

```text
artifacts/qrels_gold.jsonl                         0 rows
artifacts/qrels_silver.jsonl                     175 rows
artifacts/qrels_bronze.jsonl                     319 rows
artifacts/field_state/adjudicated_qrels.jsonl      3 rows
artifacts/field_state/qrels_reviews.jsonl         10 rows
artifacts/eval/llm_judgments.jsonl                 0 rows
```

Actual current corpus path and count:

```text
data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl = 7,171 rows
data/corpus/normalized/combined_corpus.jsonl is a directory, not a file
```

Known stale or contradictory artifacts:

- `reports/eval/corpus_manifest.json` says corpus size is 10,404 and points to `data/corpus/normalized/combined_corpus.jsonl`; this conflicts with the current 7,171-row nested corpus file.
- `reports/eval/retrieval_loading_diagnostics.md` warns that corpus load failed because `data/corpus/normalized/combined_corpus.jsonl` is a directory.
- `reports/eval/neuro_judge_retrieval_metrics.md` claims 675 annotated pairs, but actual `artifacts/field_state/adjudicated_qrels.jsonl` has only 3 rows and `artifacts/qrels_gold.jsonl` is empty. Treat that report as stale or incompatible until proven otherwise.
- `reports/eval/ndcg_report.md` is based on 8 LLM-qrels pairs across 1 query. It is diagnostic only.
- `artifacts/field_state/current_manifest.json` appears to be a test artifact pointing to `/tmp/pytest-...` with only 701 nodes and 1 edge. It should not be cited as current production state.
- `artifacts/field_state/memory_graph_manifest.json` reports a 629-record reviewed slice with 2,200 nodes and 3,788 edges.
- `data/graph/neural_search_graph.real_corpus.json` appears to be the full-corpus graph cited by the whitepaper: 7,593 nodes / 31,920 edges.
- `apps/web/src/pages/SearchPage.tsx` still displays stale product copy: `v2.0 · 873 datasets` and `Corpus Map — 873 datasets visualized`. This conflicts with the current 7,171-row corpus and should be fixed or made dynamic.

Current git status includes untracked active artifacts:

```text
?? artifacts/eval/llm_judgments.jsonl
?? artifacts/literature/findings_tier1_ollama.checkpoint.json
?? artifacts/literature/findings_tier1_ollama.jsonl
?? docs/superpowers/plans/2026-06-18-knowledge-explorer.md
?? reports/strategy/brainknow_comparison_publishable_plan.md
?? reports/strategy/claude_whitepaper_frontend_execution_prompt.md
```

### High-Level Evaluation

The whitepaper is conceptually strong and unusually honest about validation gaps. Its best current thesis is:

> Neural Search is not a generic RAG or another concept-only brain KG. It is a provenance-backed research-object retrieval system for finding reusable neuroscience datasets, linked papers, structured findings, analysis affordances, and evidence quality.

Keep that. Do not overclaim scientific retrieval performance before gold qrels exist.

The most important paper risk is not lack of ideas. It is artifact inconsistency:

- Some docs/reports cite old 10,404-record corpus numbers.
- The current corpus path is odd: `combined_corpus.jsonl/` is a directory.
- Metrics reports are mixed across historical, silver, smoke, and stale runs.
- The whitepaper itself mostly handles this correctly, but it should be made more reviewer-proof by adding a visible "Artifact Reconciliation" subsection and by quarantining stale reports from paper tables.

The most important product risk is not missing features. The frontend already has many useful surfaces:

- Search page.
- Structured filters.
- Dataset result cards.
- Evidence packet panel.
- Neuro-judge watermark.
- Raw evidence JSON.
- Feedback logging.
- Related findings panel.
- Dataset detail cards.
- Affordance panel.
- Similar datasets panel.
- Knowledge Explorer graph.
- Brain Atlas / Coverage pages.

But the experience does not yet feel like the definitive "Neuronpedia for neuroscience datasets." It needs cohesion, stronger visual storytelling, data-driven counters, and a graph/evidence journey that feels alive.

## Workstream A: Whitepaper Hardening

Primary files:

- `docs/whitepaper/neural_search_whitepaper.tex`
- `reports/strategy/brainknow_comparison_publishable_plan.md`
- `reports/eval/whitepaper_claims_status.md`
- `docs/CLAIM_LEDGER.md`
- `docs/WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md`

Tasks:

1. Add or strengthen an "Artifact Reconciliation and Evidence Tiers" subsection near the beginning of the whitepaper.
   - State current source of truth:
     - corpus: `data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl`
     - gold qrels: empty
     - field-state adjudicated qrels: 3 rows
     - silver/bronze qrels: diagnostic only
     - full-corpus KG: `data/graph/neural_search_graph.real_corpus.json`
     - field-state memory graph: 629-record curation slice
   - Explicitly classify reports into:
     - publishable current-state artifacts,
     - engineering validation,
     - silver diagnostics,
     - stale historical artifacts,
     - future/planned results.

2. Ensure every metric table in the whitepaper uses cautious language.
   - "Current validation artifacts" should not imply final benchmark performance.
   - Keep NDCG/MRR/Recall as pending unless backed by human/adjudicated qrels.
   - Keep vector recall clearly separated from scientific retrieval recall.

3. Add a short related-work contrast with BrainKnow/LBD.
   - BrainKnow: PubMed-scale concept co-occurrence KG, Node2Vec synthesis, strong field-scale map.
   - Neural Search: dataset-linked, provenance-backed, analysis-aware, relation-typed, human-reviewable retrieval.
   - Do not claim Neural Search is larger or more complete.
   - Claim the differentiator is reuse and testability: "which dataset can test/reproduce/contradict this claim?"

4. Add a crisp "Why This Is Needed" section.
   - Repositories expose data but weak scientific intent search.
   - Literature KGs expose concepts but weak dataset reusability.
   - RAG exposes text answers but weak provenance and analysis feasibility.
   - Neural Search bridges datasets, papers, findings, affordances, and evidence status.

5. Add a reviewer-facing "Current Non-Claims" paragraph.
   - No final retrieval superiority claim yet.
   - No final finding-extraction precision claim yet.
   - No gold-qrels benchmark yet.
   - No content-level neural signature search yet.
   - No generalization claim across all neuroscience yet.

6. Update stale wording if present.
   - Search for 10,404, 873, "gold benchmark", "675 annotated", "NDCG >".
   - If a claim cannot be supported by the actual artifacts listed above, downgrade it or mark it historical/stale.

Acceptance criteria:

- Whitepaper can be read by a skeptical reviewer without feeling metrics are inflated.
- Every number has a named artifact and evidence tier.
- The paper's strongest claims are about architecture, implemented infrastructure, reproducible workflow, and the research-object retrieval framing.
- Final scientific retrieval claims are clearly pending qrel completion.

Suggested lightweight commands:

```bash
rg -n "10404|10,404|873|675|gold|NDCG|MRR|Recall|publishable|validated|unproven|pending|stale" docs reports
wc -l artifacts/qrels_gold.jsonl artifacts/qrels_silver.jsonl artifacts/qrels_bronze.jsonl artifacts/field_state/adjudicated_qrels.jsonl
wc -l data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl
```

## Workstream B: Repo Artifact Reconciliation

Primary files/scripts:

- `reports/eval/corpus_manifest.json`
- `reports/eval/retrieval_loading_diagnostics.md`
- `scripts/eval/diagnose_retrieval_loading.py`
- `scripts/eval/freeze_corpus_manifest.py` if present, otherwise create a focused manifest helper
- `neural_search/normalized.py`
- any config that points to `data/corpus/normalized/combined_corpus.jsonl`

Tasks:

1. Fix path assumptions that treat `data/corpus/normalized/combined_corpus.jsonl` as a file.
   - Either update configs to use `data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl`, or make loaders detect directory-with-corpus-file gracefully.
   - Avoid breaking older tests that may use file paths.

2. Generate or update a current lightweight manifest.
   - Include:
     - corpus path
     - row count
     - unique IDs
     - sha256
     - generated timestamp
     - git commit if easily available
     - graph artifact path and counts if cheap to compute
     - qrels counts
   - Write it somewhere explicit, e.g. `reports/eval/current_artifact_manifest.json`.

3. Update diagnostics so they report current state and flag stale reports.
   - If a path points to a directory, diagnostics should say where the current file is.
   - If gold qrels are empty, report "gold qrels pending" rather than error.

4. Add a small test for path resolution.
   - Loader should accept both file and directory forms where appropriate.
   - It should fail clearly if no JSONL corpus file exists.

Acceptance criteria:

- A single manifest can be cited by the whitepaper.
- Retrieval loading diagnostics no longer fail solely because the corpus path is a directory.
- The repo has a clear "current artifact truth" document/report.

## Workstream C: Frontend "Out Of This World" Upgrade

Primary files:

- `apps/web/src/pages/SearchPage.tsx`
- `apps/web/src/pages/ResultsPage.tsx`
- `apps/web/src/components/DatasetCard.tsx`
- `apps/web/src/pages/DatasetPage.tsx`
- `apps/web/src/pages/KnowledgeExplorerPage.tsx`
- `apps/web/src/pages/BrainAtlasPage.tsx`
- `apps/web/src/pages/CoveragePage.tsx`
- `apps/web/src/components/graph/*`
- `apps/web/src/index.css`
- `apps/api/main.py`
- `apps/api/graph_router.py`

Design target:

Make the first 60 seconds feel like a scientific cockpit:

1. Search asks a real scientific intent.
2. Results explain why each dataset is useful.
3. Evidence is inspectable, not hidden.
4. The graph shows how datasets, findings, papers, regions, and methods connect.
5. The UI constantly distinguishes gold/human evidence from silver/model/metadata evidence.

Do not make a marketing landing page. The first screen should remain the usable search experience.

### Frontend Phase 1: Fix Trust And Coherence

Tasks:

- Replace hardcoded `873 datasets` in `SearchPage.tsx`.
  - Prefer dynamic API-backed corpus summary.
  - If dynamic is too much for this pass, use "7,171 dataset records" with a comment pointing to the manifest.
- Add a compact "Evidence Tier" strip on the search page:
  - Corpus records
  - Linked papers
  - Extracted findings
  - Silver qrels
  - Human adjudicated qrels
  - Clearly show which are pending.
- Make all silver labels visually watermarked.
  - Neuro-judge badges should already say preliminary; make the styling unmistakable but not alarming.
- Make qrel/gold state visible in Evaluation page.
  - "Gold qrels: 0"
  - "Field-state adjudicated: 3"
  - "Silver/bronze diagnostic: 175/319"
  - Avoid showing stale 675 annotated pairs unless clearly historical.

Acceptance criteria:

- No stale 873 count remains.
- Users immediately understand what is validated and what is provisional.

### Frontend Phase 2: Results As Evidence Cards

Upgrade `DatasetCard.tsx`.

Tasks:

- Add a compact score-radar or evidence bar using existing score breakdown:
  - task
  - modality
  - region
  - species
  - affordance
  - graph
  - readiness
- Add a "Can test this?" micro-panel when evidence exists:
  - supported dimensions
  - missing dimensions
  - contraindications/hard negatives
  - raw/processed evidence
- Improve feedback controls.
  - Current controls are functional but visually dense.
  - Turn them into a compact segmented control + reason chips + optional note drawer.
- Make "Open evidence" feel like opening a lab notebook:
  - tabs: Match, Evidence, Papers, Findings, Graph, Raw JSON.
  - Keep raw JSON available, but not visually dominant.

Acceptance criteria:

- Top-10 results can be scanned quickly.
- A scientist can tell why a dataset matches without reading raw JSON.
- Evidence packet details remain available.

### Frontend Phase 3: Dataset Detail As A Research Object Page

Upgrade `DatasetPage.tsx`.

Tasks:

- Add a top "Research Object Summary" band:
  - dataset identity
  - source
  - standard
  - species/modality/region/task chips
  - raw data status
  - QA status
  - linked paper count
  - affordance count
- Add an "Evidence Timeline" or "Evidence Stack":
  - repository metadata
  - linked papers
  - extracted findings
  - graph neighbors
  - human reviews
  - feedback signals
- Add a "What Can I Do With This Dataset?" panel:
  - high-confidence analyses
  - medium/low analyses
  - missing requirements
  - generate notebook button per top analysis.
- Upgrade Similar Datasets panel:
  - group by relation:
    - same region + same task
    - same region + cross modality
    - same task + cross species
  - show why each neighbor matters.
- Add "Claims this dataset may test" placeholder if findings linkage is missing.
  - If no linked findings, show a useful empty state: "No paper/finding bridge yet; link papers to activate claim testing."

Acceptance criteria:

- Dataset page feels like a reusable scientific dossier, not metadata dump.
- Similar datasets and affordances are meaningfully explained.

### Frontend Phase 4: Knowledge Explorer As The Signature Experience

Upgrade `KnowledgeExplorerPage.tsx` and graph components.

Current state:

- Has galaxy view.
- Has explorer graph.
- Has filters, suggested views, ontology tree, legend, node details.
- Uses artifacts:
  - `artifacts/graph/galaxy_points.json`
  - `artifacts/graph/cluster_graph.json`
  - `artifacts/literature/findings_tier1_ollama.jsonl`
  - `artifacts/literature/paper_dataset_links.jsonl`

Tasks:

- Add layer switch with clearer labels:
  - Dataset Corpus
  - Literature Findings
  - Paper-Dataset Bridges
  - Validation / Qrels
  - Coverage Gaps
- Add visual encoding:
  - Node color = type
  - Node ring = evidence tier
  - Node size = evidence count / dataset count
  - Edge color = relation type
  - Edge opacity = confidence
  - Dashed edges = silver/provisional
- Add "story views" as preset tours:
  - Hippocampal navigation
  - Reward learning
  - Cross-species decision making
  - Human BCI / ECoG
  - Negative evidence / hard negatives
  - Coverage gaps
- Add click-to-search bridge:
  - Clicking a region/task/finding should offer "Search datasets for this context."
- Add graph-to-dataset bridge:
  - Clicking a dataset node opens a right panel with:
    - dataset title
    - source
    - top labels
    - linked papers
    - affordances
    - "Open dataset card"
- Add "evidence lens":
  - toggle between all edges, human-reviewed only, silver only, missing evidence.

Acceptance criteria:

- The graph is no longer just a visualization; it becomes a navigable discovery surface.
- It visibly communicates provenance and uncertainty.
- It creates a clear path from concept/finding -> datasets -> evidence -> notebook.

### Frontend Phase 5: Brain Atlas And Coverage

Upgrade `BrainAtlasPage.tsx` and `CoveragePage.tsx`.

Tasks:

- Brain atlas should feel like a map of what the corpus can see.
- Add heatmap layers:
  - dataset count
  - modality diversity
  - species diversity
  - validated region labels
  - missing/uncertain labels
- Add dark-pair explorer:
  - region x modality gaps
  - region x species gaps
  - task x modality gaps
- Add "Acquisition target" cards:
  - why this gap matters
  - candidate source to ingest
  - expected impact
- Add "coverage confidence" so sparse or metadata-only regions do not look over-certain.

Acceptance criteria:

- Atlas view answers: "What parts of neuroscience can this system currently search well?"
- It also answers: "Where is the corpus blind?"

## Workstream D: Needed-Piece Validation

Make the project defensible as a needed piece of infrastructure.

Add a document or whitepaper subsection answering:

### Why Existing Tools Are Not Enough

- Archive search: finds records but poorly understands scientific intent, analysis feasibility, and hard negatives.
- Literature search: finds papers but not reusable datasets.
- Knowledge graphs like BrainKnow: map concepts at scale but mostly do not link to analysis-ready downloadable datasets.
- Generic embeddings/RAG: retrieves text but does not enforce provenance, constraints, source quality, or analysis affordances.

### What Neural Search Uniquely Enables

- Query by experimental intent.
- Retrieve datasets, not just documents.
- Explain why a dataset matches.
- Show what analyses are supported.
- Link datasets to papers and extracted findings.
- Track provenance and evidence tier.
- Support human qrels and feedback loops.
- Identify coverage gaps and acquisition targets.

### What Must Be Proven Next

- Human qrels benchmark shows improved NDCG/MRR/Recall over BM25/dense/co-occurrence baselines.
- Hard-negative violation rate improves.
- Affordance labels predict actual analysis feasibility.
- Paper-dataset links are precise enough to trust.
- Finding extraction precision is acceptable after manual audit.
- Users can find a reusable dataset faster than archive-native search.

### Recommended Validation Ladder

1. Smoke: exact lookup and known-item queries.
2. Silver: LLM/neuro-judge diagnostic labels for triage only.
3. Human qrels: 100 queries, 1,500 pooled pairs, two annotators.
4. Extraction audits: 100 findings, 100 paper-dataset links, 100 affordance labels.
5. User study: 5-10 neuroscience users compare Neural Search vs DANDI/OpenNeuro/Google Scholar for dataset discovery tasks.
6. Prospective case studies: produce notebooks for 3 datasets found through the system.

## Implementation Priorities

Do this order:

1. Reconcile artifact truth and stale report warnings.
2. Patch obvious stale frontend counts/copy.
3. Add current artifact manifest and diagnostics.
4. Tighten whitepaper wording around evidence tiers.
5. Improve results cards and dataset pages.
6. Then enhance Knowledge Explorer/Atlas.

Avoid this until user says so:

- Full extraction rebuild.
- Full embedding rebuild.
- Full qrels metrics as "gold."
- Deleting stale reports.

## Suggested Verification

Lightweight checks:

```bash
python -m pytest tests/test_normalized_schema.py tests/test_graph_schema.py tests/test_api_smoke.py -q
npm --prefix apps/web run build
```

If `npm` build fails because dependencies are missing or network is required, report that rather than installing anything without approval.

Manual QA:

- Search page has no stale counts.
- Results page still searches.
- Dataset card evidence opens.
- Feedback logging still posts.
- Dataset page loads affordances and similar datasets.
- Knowledge Explorer loads galaxy and explorer modes.

## Final Deliverable

Return:

- A concise summary of whitepaper changes.
- A list of reconciled artifact counts and remaining stale reports.
- Frontend changes with screenshots if Playwright/browser checks are available.
- Tests/builds run and results.
- Remaining blockers tied to qrels/extraction jobs.

