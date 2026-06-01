# Frontend Product Plan

## Demo goal

A scientist should understand within 60 seconds that this is not a generic search box. It is a reuse-oriented dataset cockpit.

## Main pages

### Search page

Add:

- Suggested queries grouped by task/modality.
- Structured filters for task, behavior, modality, region, species, source, standard, readiness.
- Clear empty/error state.
- “What this searches” explainer.

### Results page

Each result should show:

- Dataset title/source.
- Score and readiness.
- Why matched.
- Matched ontology terms.
- Missing metadata warnings.
- Linked papers/evidence snippets.
- Buttons: view card, compare, generate notebook, mark reviewed/trusted/rejected.

### Dataset detail page

Sections:

- Overview.
- Provenance/source archive.
- Experimental labels.
- Analysis readiness.
- Missing metadata.
- Linked papers.
- Dataset card JSON/Markdown.
- Starter notebook.
- QA status.

### Comparison drawer/page

Must answer:

- Which dataset is best for my analysis?
- Which metadata fields are shared or missing?
- Which modalities/tasks/regions overlap?
- Which one has stronger provenance/readiness?

### Evaluation page

Show:

- Benchmark summary.
- Per-query pass/fail.
- Top missed labels.
- Hard-negative violations.
- Regression vs baseline.

### Corpus dashboard

Show:

- Dataset count by source.
- Modalities coverage.
- Tasks coverage.
- Brain regions coverage.
- QA status distribution.
- Missing metadata heatmap.

## Design tone

Keep the existing dark neural aesthetic, but make information hierarchy cleaner:

- Cards should be readable under demo pressure.
- Use badges for task/modality/region/source.
- Use warnings sparingly but clearly.
- Avoid giant blobs of JSON unless behind a disclosure.
- Make “why matched” the star of the show.

## Quick wins

1. Fix TypeScript build.
2. Add loading/error/empty states to all API-backed pages.
3. Add top-level corpus/eval stats to the layout or homepage.
4. Make comparison summary use all data it fetches.
5. Add a “copy CLI command” for generated notebooks and ingestion queries.
