# Neural Search Demo Walkthrough

This walkthrough is designed for a public technical demo. The goal is to make the system's thesis obvious: Neural Search is experiment-aware dataset discovery, not generic RAG.

## Setup

```bash
python -m pip install -e ".[all]"
cd apps/web && npm install && cd ../..
make demo
```

Run the app:

```bash
# Terminal 1
make api

# Terminal 2
make web
```

Open http://localhost:5173.

If the web app shows an API error, check that `make api` is still running on port 8000. The frontend dev server runs on port 5173 and proxies `/api` requests to the backend.

## Demo Assets

Placeholder visuals live in `docs/demo_media/` and can be replaced with screenshots or GIFs before recording:

- `search_ui_placeholder.svg`
- `dataset_card_placeholder.svg`
- `benchmark_report_placeholder.svg`

Use a 90-120 second flow:

1. Search page and example queries.
2. Results page with why-matched evidence.
3. Dataset card with readiness, provenance, missing metadata, literature, and notebook generation.
4. Evaluation dashboard with benchmark queries.
5. Reports page with corpus coverage and demo readiness.

## Talk Track

Open with:

> Neural Search lets a researcher describe the experiment they want to analyze. It searches over task ontology, metadata, semantic text, and provenance to return reusable datasets, not generated prose.

Emphasize:

- The query is interpreted as experimental intent.
- Results are datasets with explanations, warnings, and reuse affordances.
- Dataset cards are generated artifacts that can be reviewed.
- Starter notebooks turn search into first analysis.
- Benchmarks evaluate whether ontology and ranking recover expected scientific labels.

Avoid calling it a chatbot or RAG demo. If asked, say:

> RAG retrieves passages for answer generation. Neural Search retrieves research objects and keeps the matching evidence, provenance, and reuse constraints visible.

## Script

### 1. Run the CLI Demo

```bash
make demo
```

The command should show ontology loading, demo dataset loading, dataset-card generation, benchmark evaluation, report generation, notebook creation, and a sample search.

Useful fallback commands:

```bash
make demo-search QUERY="Find reversal learning datasets with reward omission"
make benchmark
make reports
make notebook-generate DATASET_ID=DEMO_GONOGO_CALCIUM
```

### 2. Search by Experiment

Start with:

```text
Find reversal learning datasets with reward omission and trial outcomes
```

Point out:

- The match is not just keyword overlap.
- The result should surface `reversal_learning`, reward/omission behavior, neural modality, source archive, and readiness.
- Why-matched evidence explains the rank.
- Missing metadata warnings are a feature, not a failure.

Then try:

```text
Go/NoGo task with calcium imaging in mPFC and lick events
```

Use this to show ontology normalization across task, behavior, brain region, and modality.

### 3. Use Structured Search

Open Advanced Search and choose:

- Task: `go_nogo` or `reversal_learning`
- Modality: `calcium_imaging`, `neuropixels`, or `ieeg`
- Species: `mouse` or `human`
- Minimum Readiness: `70`

Point out the structured JSON and natural-language preview. The important message is that free text and explicit experimental constraints can coexist.

### 4. Open a Dataset Card

Open the top dataset card. Show:

- Scientific reuse summary.
- Experimental structure.
- Neural data modalities and brain regions.
- Scientific labels.
- Analysis readiness score.
- Strengths and limitations.
- Reuse instructions.
- Linked literature.
- Provenance and QA status.

Say:

> The dataset card is the handoff artifact. It tells the next researcher what was inferred, what evidence supports it, what is missing, and what the first analysis could be.

### 5. Generate a Starter Notebook

Click **Generate Notebook**. The notebook should download with:

- Loading template.
- Metadata inspection.
- Trial and event structure checks.
- Modality-specific analysis cells.
- TODOs for scientific follow-up.

Say:

> The search result is actionable. It does not stop at discovery; it creates a starting point for analysis.

### 6. Show Benchmark Report

Open `/evaluation` and click **Run Benchmark**.

Point out:

- Precision@5 and label recall.
- Per-query expected tasks and modalities.
- Top returned datasets.
- Warnings and recommendations.

Use this frame:

> We can evaluate the retrieval layer against scientific expectations: did it recover the right task labels, modalities, and behaviors?

### 7. Show Compilation Report

Open `/reports`.

Point out:

- Dataset coverage by source, task, modality, species, brain region, and data standard.
- Missing metadata distribution.
- Top demo-ready datasets.
- Top analysis-ready datasets.

Say:

> The system also helps curators understand corpus coverage and gaps.

## Example Queries

Use these during Q&A:

- `Find reversal learning datasets with reward omission and trial outcomes`
- `Go/NoGo task with calcium imaging in mPFC and lick events`
- `Visual decision-making with Neuropixels recordings`
- `Find datasets where I can decode choice from neural activity`
- `Human ECoG or iEEG reaching data for BCI classification`
- `Delay discounting with fiber photometry and reward choice`
- `Find behavior video datasets with pose tracking`
- `Find datasets with linked papers and reusable NWB files`

## Expected Local URLs

- Frontend: http://localhost:5173
- API: http://localhost:8000
- API health: http://localhost:8000/healthz

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Frontend says API request failed | Start `make api` and refresh the page |
| Empty evaluation dashboard | Click **Run Benchmark** or run `make benchmark` |
| Notebook button is disabled | Choose an available template or open a dataset with matching modality/standard |
| No search results | Try a demo query or clear strict structured filters |
| Port 5173 is busy | Run `cd apps/web && npm run dev -- --port 5174` |
| Port 8000 is busy | Run `uvicorn apps.api.main:app --reload --port 8001` and update Vite proxy if needed |

## Closing Message

End with:

> The next step is latent neural-state search: preserving ontology and provenance, while indexing learned representations of neural dynamics so researchers can search for comparable computational states across tasks, species, and recording technologies.
