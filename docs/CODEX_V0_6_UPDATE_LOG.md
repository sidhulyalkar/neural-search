# Codex v0.6 Update Log

Date: 2026-05-24

## Summary

Codex unified the v0.5 retrieval substrate around a canonical normalized demo corpus and wired optional graph and field-embedding signals into the main dataset search path. The search pipeline now supports explicit hard-negative filtering instead of penalty-only handling.

## Implemented

- Added deterministic demo seed to normalized corpus conversion.
- Added artifact build targets for corpus, graph, reports, embeddings, and the full pipeline.
- Built normalized demo corpus artifacts from `data/seed/demo_datasets.yaml` and `data/seed/demo_papers.yaml`.
- Built graph artifact from normalized records.
- Built hashing field embedding cache from normalized records.
- Added optional graph loading and `graph_score` in result score breakdowns.
- Added optional field embedding loading and `field_semantic_score` in result score breakdowns.
- Added hard-negative parser and filter support for:
  - `exclude`
  - `excluding`
  - `without`
  - `not`
  - `no`
  - `but not`
  - `NOT`
- Added hard-negative filtering across modalities, tasks, sources, species, regions, behavior-only datasets, analysis affordances, and recording devices.
- Updated API tests to call endpoint functions directly because local `TestClient` transport hangs in this sandbox.
- Allowed the ontology model to tolerate a top-level `analysis_affordances` section.

## Artifacts

- `data/corpus/normalized/demo_v05.datasets.jsonl`: 26 dataset records.
- `data/corpus/normalized/demo_v05.papers.jsonl`: 26 paper records.
- `data/corpus/normalized/demo_v05.records.jsonl`: 52 total records.
- `data/graph/neural_search_graph.demo_v05.json`: 308 nodes and 957 edges.
- `data/embeddings/demo_v05.field_embeddings.jsonl`: 337 hashing field embeddings.
- `data/reports/graph/`: 4 graph reports.

## Validation

```bash
pytest -q
ruff check neural_search tests
make artifacts-build
python -m neural_search.evaluation.run_benchmark --suite demo_v02
python -m neural_search.evaluation.run_benchmark --suite adversarial
```

## Results

- Tests: 162 passed.
- Lint: passed.
- Artifact build: passed.
- `demo_v02` benchmark:
  - Precision@5: 0.787
  - Label recall@10: 0.878
  - Hard-negative violations: 0
- `adversarial` benchmark:
  - Precision@5: 0.800
  - Label recall@10: 0.766
  - Hard-negative violations: 0

## Notes

- Benchmark commands exit successfully, but a few individual queries remain below per-query targets in the generated reports.
- The default CI embedding provider remains hashing.
- Graph and field embedding search are opt-in through retrieval config and fall back safely when artifacts are absent.
