# Neural Search Demo Walkthrough

This walkthrough is designed for a technical demo. The goal is to make the system's thesis obvious: Neural Search is experiment-aware dataset discovery over a real corpus and knowledge graph, not generic RAG over documents.

## Setup

```bash
python -m pip install -e ".[all]"
cd apps/web && npm install && cd ../..
```

Run the app against the real corpus (default) or the small fixture corpus (`NEURAL_SEARCH_DEMO_MODE=1`, useful for CI/quick local runs):

```bash
# Terminal 1
make api

# Terminal 2
make web
```

Open http://localhost:5173.

If the web app shows an API error, check that `make api` is still running on port 8000. The frontend dev server runs on port 5173 and proxies `/api` requests to the backend.

## Talk Track

Open with:

> Neural Search lets a researcher describe the experiment they want to analyze. It searches a real, ~150,000-edge knowledge graph over 7,171 dataset records, using hybrid sparse/dense/graph ranking, to return reusable datasets with evidence — not generated prose.

Emphasize:

- The query is interpreted as experimental intent.
- Results are datasets with explanations, warnings, and evidence-tiered reuse affordances.
- Dataset cards are generated artifacts that can be reviewed, including linked-paper retraction status.
- A search result can compile directly into an ExperimentGlancer scene — a synchronized, shareable timeline view.
- Starter notebooks turn search into first analysis.
- The NDCG@10 ablation ladder is re-run and gated on every knowledge-graph change — this is a real, automated discipline, not a one-time claim.

Avoid calling it a chatbot or RAG demo. If asked, say:

> RAG retrieves passages for answer generation. Neural Search retrieves research objects and keeps the matching evidence, provenance, evidence tier, and reuse constraints visible.

## Script

### 1. Search by Experiment

Start with:

```text
Find reversal learning datasets with reward omission and trial outcomes
```

Point out:

- The match is not just keyword overlap — it combines ontology matching, dense embeddings, and graph context.
- Why-matched evidence explains the rank.
- Missing metadata warnings are a feature, not a failure.

Then try:

```text
Go/NoGo task with calcium imaging in mPFC and lick events
```

Use this to show ontology normalization across task, behavior, brain region, and modality.

### 2. Use Structured Search

Open Advanced Search and choose a task, modality, species, and minimum readiness. Point out the structured JSON and natural-language preview coexisting.

### 3. Open a Dataset Card

Open the top dataset card. Show experimental structure, neural data modalities, readiness score, linked literature (including any retraction/correction warning), provenance, and QA status.

> The dataset card is the handoff artifact. It tells the next researcher what was inferred, what evidence supports it, what is missing, and what the first analysis could be.

### 4. Open an ExperimentGlancer Scene

Click **Open Scene** on a result card. Show the synchronized timeline: which layers are `available` (file-derived), which are `probable` (metadata-inferred, clearly marked), the anchor the query implied (e.g. a lick-onset event), and the shareable scene URL.

> This isn't a mockup — every layer's status is honest about what's actually been verified versus inferred from metadata.

### 5. Generate a Starter Notebook

Click **Generate Notebook**. Say:

> The search result is actionable. It does not stop at discovery; it creates a starting point for analysis.

### 6. Show Benchmark and Coverage Reports

Open `/evaluation` and `/reports`. Point out NDCG@10 on the 317-query canonical benchmark, and corpus coverage by source, task, modality, species, and brain region.

> We evaluate the retrieval layer against scientific expectations, and re-check it automatically every time the knowledge graph changes.

## Example Queries

- `Find reversal learning datasets with reward omission and trial outcomes`
- `Go/NoGo task with calcium imaging in mPFC and lick events`
- `Visual decision-making with Neuropixels recordings`
- `Find datasets where I can decode choice from neural activity`
- `Human ECoG or iEEG reaching data for BCI classification`
- `Delay discounting with fiber photometry and reward choice`

## Expected Local URLs

- Frontend: http://localhost:5173
- API: http://localhost:8000
- API health: http://localhost:8000/healthz

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| Frontend says API request failed | Start `make api` and refresh the page |
| No search results | Try a demo query or clear strict structured filters |
| ExperimentGlancer scene shows mostly "probable" layers | Expected for sources without a live file validator (currently DANDI/OpenNeuro only) — not a bug |
| Port 5173 is busy | Run `cd apps/web && npm run dev -- --port 5174` |
| Port 8000 is busy | Run `uvicorn apps.api.main:app --reload --port 8001` and update the Vite proxy if needed |

## Closing Message

End with:

> The next steps are closing the gold-qrels gap so ranking claims can be human-validated, and scaling live file validation from the top reanalysis candidates to the full corpus — turning more of what's currently "probable" into genuinely "available."
