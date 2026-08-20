# Neural Search

Experiment-aware discovery and reuse infrastructure for neuroscience datasets, literature, methods, and derived knowledge.

Neural Search is an alpha research platform for finding reusable neural and behavioral data in experimental terms rather than only by document keywords. It combines structured scientific metadata, behavioral and anatomical normalization, semantic retrieval, provenance-aware dataset cards, literature links, knowledge-graph signals, and evaluation tooling.

The repository contains both a small deterministic demo path and larger research pipelines. The demo is the easiest way to understand the product loop; larger corpus, embedding, graph, and literature artifacts are intentionally treated as reproducible research assets and may not all be committed to a fresh clone.

This is not generic RAG. The primary artifact is not a synthesized answer over retrieved chunks. Neural Search retrieves datasets and evidence, explains why they match, exposes missing metadata and evidence quality, links relevant literature and methods, and supports concrete reuse workflows.

![Search UI](docs/demo_media/search_ui.png)

## What you can do

1. **Search experiments, not filenames**: query by task, behavior, modality, species, brain region, data standard, intended analysis, and reuse constraints.
2. **Normalize scientific language**: map synonyms and related experimental concepts into searchable structured fields.
3. **Inspect provenance and uncertainty**: preserve source archive IDs, literature evidence, QA state, missing metadata, and evidence tiers instead of hiding them behind a score.
4. **Generate reuse artifacts**: produce dataset cards and starter notebooks for concrete follow-up work.
5. **Explore linked knowledge**: connect datasets, papers, findings, methods, regions, and other scientific concepts through graph-backed interfaces and derived relationships.
6. **Evaluate retrieval scientifically**: keep benchmarks, qrels, hard negatives, ablations, artifact manifests, and calibration machinery separate from product demos.

| Dataset card, provenance, and QA | Ablation-ladder benchmark (NDCG@10) |
|---|---|
| ![Dataset card](docs/demo_media/dataset_card.png) | ![Benchmark](docs/demo_media/benchmark_dashboard.svg) |

The longer-term direction is **latent neural-state search**: searching across learned representations of neural population state, task structure, behavior, and analysis affordances while retaining ontology and provenance layers as an interpretability scaffold.

## Repository map

```text
neural-search/
├── apps/
│   ├── api/                 FastAPI application and domain routers
│   └── web/                 React + Vite research interface
├── neural_search/           Core Python package
│   ├── ingestion/           Source adapters and normalization
│   ├── ontology/            Scientific vocabulary and matching
│   ├── search/              Retrieval, ranking, and query parsing
│   ├── graph/               Knowledge-graph schemas and features
│   ├── literature/          Findings, linking, and evidence processing
│   ├── evaluation/          Benchmarks and evaluation utilities
│   ├── cards/               Dataset-card generation
│   └── notebooks/           Starter notebook generation
├── scripts/                 Reproducible ingestion/evaluation/build jobs
├── tests/                   Unit, integration, and artifact safety tests
├── data/                    Small fixtures plus references to research assets
├── reports/                 Generated and frozen evaluation/report artifacts
└── docs/                    Architecture, evaluation, limitations, and whitepaper
```

## Quick start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Git
- Docker only if you want the optional service stack

### 1. Create an isolated environment

```bash
git clone https://github.com/sidhulyalkar/neural-search.git
cd neural-search
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2. Install the demo/development dependencies

```bash
python -m pip install -e ".[dev]"
cd apps/web
npm ci
cd ../..
```

Check that the checkout is usable:

```bash
neural-search doctor
```

The command reports the Python version, required imports, useful optional packages, and source-checkout assets without contacting external services.

### 3. Run the deterministic demo

```bash
make demo
```

### 4. Start the application

```bash
# Terminal 1
make api

# Terminal 2
make web
```

Open `http://localhost:5173`. The Vite development server proxies `/api` and `/healthz` to the backend on `http://localhost:8000`.

## Optional capabilities

Install only what you need. This keeps the first setup much smaller and makes missing research infrastructure explicit.

```bash
# Local sentence-transformer embeddings
python -m pip install -e ".[embeddings]"

# Dense/vector-index experiments
python -m pip install -e ".[dense]"

# PostgreSQL / pgvector support
python -m pip install -e ".[postgres]"

# Redis/RQ support
python -m pip install -e ".[redis]"

# Spectral/aperiodic analyses
python -m pip install -e ".[spectral]"

# Notebook environment
python -m pip install -e ".[notebooks]"
```

For example, to build an embedding index:

```bash
python -m neural_search.embeddings.build_index \
  --input data/corpus/normalized \
  --out data/indexes/embeddings \
  --provider hashing
```

Sentence-transformer models are optional and may require model downloads:

```bash
python -m neural_search.embeddings.build_index \
  --input data/corpus/normalized \
  --out data/indexes/embeddings \
  --provider sentence-transformer \
  --model sentence-transformers/all-MiniLM-L6-v2
```

## Example queries

- `Find reversal learning datasets with reward omission and trial outcomes`
- `Go/NoGo task with calcium imaging in mPFC and lick events`
- `Visual decision-making with Neuropixels recordings`
- `Find datasets where I can decode choice from neural activity`
- `Human ECoG or iEEG reaching data for BCI classification`
- `Delay discounting with fiber photometry and reward choice`

The web interface also supports structured task, behavior, modality, brain-region, source, data-standard, and readiness filters.

## Core commands

| Command | Purpose |
| --- | --- |
| `neural-search doctor` | Diagnose a fresh checkout and dependency state |
| `neural-search --version` | Show the installed package version |
| `make demo` | Run the deterministic demo pipeline |
| `make api` | Start the FastAPI backend on port 8000 |
| `make web` | Start the Vite frontend on port 5173 |
| `make demo-search QUERY="..."` | Run a single CLI search |
| `make benchmark` | Run the demo retrieval benchmark |
| `make reports` | Generate the dataset compilation report |
| `make notebook-generate DATASET_ID=DEMO_GONOGO_CALCIUM` | Generate a starter notebook |
| `make test-backend` | Run the quick backend test suite |
| `make build` | Type-check and build the frontend |

## Quality gate

Before opening or merging a PR, run:

```bash
bash scripts/quality_gate.sh
```

The gate runs backend tests, Ruff, the frontend TypeScript/Vite build, the retrieval benchmark, and the dataset compilation report. GitHub Actions runs corresponding backend, frontend, and demo-artifact jobs.

A green engineering gate does **not** automatically validate a scientific claim. Retrieval/ranking changes should be accompanied by the relevant benchmark or ablation evidence, and silver/heuristic labels must not be presented as expert gold judgments.

Common failures:

- Missing `ruff`: run `python -m pip install -e ".[dev]"`.
- Frontend dependency mismatch: run `cd apps/web && npm ci`.
- Missing optional large artifacts: read the failing test or command message before generating data; fresh-clone tests should explicitly distinguish required fixtures from optional research assets.
- Benchmark failures on specific queries: inspect the generated evaluation report for missed labels and hard-negative violations.

## API highlights

| Endpoint | Method | Description |
| --- | --- | --- |
| `/healthz` | GET | Backend health check |
| `/api/search` | POST | Experiment-aware dataset search |
| `/api/datasets` | GET | List indexed demo datasets |
| `/api/datasets/{id}/card` | GET | Dataset card with readiness, provenance, and QA |
| `/api/datasets/{id}/notebook` | POST | Generate a starter notebook |
| `/api/datasets/{id}/card/export/markdown` | GET | Export reuse card as Markdown |
| `/api/ontology/tasks` | GET | Ontology terms for the UI |
| `/api/evaluation/report` | GET | Latest in-process benchmark report |
| `/api/evaluation/run` | POST | Run benchmark evaluation |
| `/api/reports/compilation` | GET | Dataset compilation and QA report |

Additional routers expose graph, knowledge, coverage, atlas, methods, spectral, timeline, and ExperimentGlancer functionality. Treat the generated OpenAPI schema from a running backend as the authoritative endpoint contract.

## Documentation

- [Demo walkthrough](docs/demo_walkthrough.md)
- [Project vision](docs/project_vision.md)
- [Technical architecture](docs/technical_architecture.md)
- [Evaluation](docs/evaluation.md)
- [Ingestion](docs/ingestion.md)
- [Example queries](docs/example_queries.md)
- [Known limitations](docs/known_limitations.md)
- [Retrieval notes](docs/retrieval.md)
- [Dataset-card review checklist](docs/dataset_card_review_checklist.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Project status and scientific scope

Neural Search is alpha research software, not a validated clinical or diagnostic system. The repository deliberately mixes a portable demo surface with larger research pipelines and frozen evaluation artifacts. Some large corpora, graphs, embedding caches, and databases are not expected to exist in a clean clone.

When interpreting results, distinguish:

- demo fixtures from large-corpus results;
- heuristic/silver labels from expert/gold judgments;
- retrieval metrics from downstream scientific validity;
- a linked paper or method from evidence that a new analysis is scientifically appropriate;
- metadata availability from dataset quality or participant-level consent for a new use.

See `docs/known_limitations.md` and the evaluation documentation before using repository outputs in research claims.

## Contributing and citation

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), and preserve provenance plus evidence-tier distinctions when changing scientific logic.

Citation metadata is provided in `CITATION.cff`. If a corresponding paper becomes the canonical citation, that file should be updated rather than leaving users to infer how to cite the software.

## License

MIT. See [LICENSE](LICENSE).
