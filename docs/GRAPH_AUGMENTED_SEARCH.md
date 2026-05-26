# Graph-Augmented Search

The v0.5 graph layer augments retrieval without replacing the v0.4 scoring
pipeline. Search must still work when no graph file exists.

## Hooks

`neural_search.graph.search_features` provides:

- `load_graph_if_exists(path)`
- `compute_graph_features_for_result(graph, result_id, query_context=None)`
- `graph_context_score(graph, result_id, query_context=None, weights=None)`

The default graph score is capped at `0.25` and is intended as a small context
signal, not a dominant retrieval score.

## Features

For a dataset result, graph features include:

- graph degree
- linked paper labels
- analysis affordances
- task, modality, and brain-region concepts
- matched query-context labels

Missing graph inputs return empty feature lists and score `0.0`.

## CLI Inputs

Build a local graph:

```bash
python -m neural_search.graph.build_graph \
  --datasets data/corpus/enriched/datasets.jsonl \
  --papers data/corpus/enriched/papers.jsonl \
  --out data/graph/neural_search_graph.json
```

Generate reports:

```bash
python -m neural_search.graph.reports \
  --graph data/graph/neural_search_graph.json \
  --out data/reports/graph
```
