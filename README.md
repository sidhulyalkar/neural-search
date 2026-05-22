# Neural Search

Experiment-aware neural data discovery system. Describe the experiment you want - Neural Search finds reusable datasets, papers, and starter analyses.

## Overview

Neural Search is a semantic search engine for neuroscience datasets. It indexes data from DANDI, OpenNeuro, and OpenAlex, extracts scientific labels using a behavioral task ontology, and provides intelligent search with experiment-aware matching.

### Key Features

- **Ontology-driven search**: Match queries against a curated taxonomy of behavioral tasks, modalities, and brain regions
- **Dataset cards**: Auto-generated summaries with analysis readiness scores
- **Starter notebooks**: Generate Jupyter notebooks for NWB/BIDS datasets
- **Hybrid search**: Combines keyword, ontology, and vector search
- **Multi-source**: Integrates DANDI, OpenNeuro, and OpenAlex

## Architecture

```
neural-search/
├── apps/
│   ├── api/         # FastAPI backend
│   └── web/         # React frontend
├── packages/
│   ├── ontology/    # Behavioral task taxonomy
│   ├── ingestion/   # Data source connectors
│   ├── extraction/  # Label extraction
│   ├── indexing/    # Search engine
│   ├── cards/       # Dataset card generation
│   ├── notebooks/   # Notebook generation
│   └── evaluation/  # Benchmark evaluation
├── data/
│   ├── ontology/    # YAML ontology files
│   ├── seed/        # Sample data
│   └── eval/        # Benchmark queries
└── infra/           # Docker configs
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose (optional, for full stack)

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   cd apps/web && npm install
   ```

2. **Start services** (using Docker):
   ```bash
   docker-compose up -d postgres redis
   ```

3. **Run database migrations**:
   ```bash
   make db-migrate
   ```

4. **Start the API**:
   ```bash
   make api
   ```

5. **Start the frontend** (in another terminal):
   ```bash
   make web
   ```

6. Open http://localhost:3000

### Demo

Run the demo script to load sample data and test the system:

```bash
make demo
```

This will:
1. Load the behavioral task ontology
2. Ingest sample dataset records
3. Generate dataset cards
4. Index embeddings
5. Run example queries
6. Generate a starter notebook

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check |
| `/api/search` | POST | Search datasets |
| `/api/datasets` | GET | List datasets |
| `/api/datasets/{id}` | GET | Get dataset details |
| `/api/datasets/{id}/card` | GET | Get dataset card |
| `/api/datasets/{id}/notebook` | POST | Generate starter notebook |
| `/api/ontology/tasks` | GET | List ontology tasks |
| `/api/ingest/dandi` | POST | Ingest from DANDI |
| `/api/ingest/openneuro` | POST | Ingest from OpenNeuro |
| `/api/ingest/openalex` | POST | Link papers from OpenAlex |
| `/api/evaluation/run` | POST | Run benchmark evaluation |

## Ontology

The behavioral task ontology (`data/ontology/behavioral_task_ontology.yaml`) defines:

- **Tasks**: Go/NoGo, 2AFC, reversal learning, spatial navigation, etc.
- **Categories**: decision_making, cognitive_control, memory, sensory, motor
- **Synonyms**: Alternative names for tasks (e.g., "2AFC" → "two-alternative forced choice")
- **Events**: Common trial events (cue_onset, response, reward, etc.)
- **Modalities**: Relevant recording types (neuropixels, calcium_imaging, etc.)
- **Brain regions**: Typical regions studied (mPFC, hippocampus, etc.)
- **Analyses**: Suggested analysis approaches

## Development

### Testing

```bash
make test          # Run all tests
make test-unit     # Unit tests only
make lint          # Run linters
make format        # Auto-format code
```

### Adding a New Task

1. Edit `data/ontology/behavioral_task_ontology.yaml`
2. Add the task with all required fields
3. Reload the ontology: restart the API or call reload endpoint

### Adding a New Data Source

1. Create a connector in `packages/ingestion/`
2. Implement the `DataSourceConnector` interface
3. Add an ingestion endpoint in `apps/api/`
4. Register the source in the ingestion service

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://...` | PostgreSQL connection URL |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `LOG_LEVEL` | `INFO` | Logging level |

## Evaluation

Run benchmark queries to evaluate search quality:

```bash
make eval
```

Benchmarks are defined in `data/eval/benchmark_queries.yaml` and measure precision@k for predefined queries with known relevant datasets.

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request
