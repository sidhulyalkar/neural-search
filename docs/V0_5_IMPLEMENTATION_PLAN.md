# V0.5 Implementation Plan

## Completed Backend Substrate

- Graph schema and validation in `neural_search.graph.schema`
- Graph builder and build CLI in `neural_search.graph.builder` and
  `neural_search.graph.build_graph`
- Lightweight query engine in `neural_search.graph.query`
- Markdown graph reports in `neural_search.graph.reports`
- Optional graph search features in `neural_search.graph.search_features`
- Experimental design seed loading and matching in
  `neural_search.graph.experimental_design`
- Graph fixtures in `tests/fixtures/graph`

## Quality Commands

```bash
pytest -q tests/test_graph_schema.py tests/test_graph_builder.py tests/test_graph_query.py tests/test_graph_reports.py tests/test_graph_search_features.py tests/test_graph_experimental_design.py tests/test_graph_fixtures.py
ruff check neural_search/graph tests/test_graph_schema.py tests/test_graph_builder.py tests/test_graph_query.py tests/test_graph_reports.py tests/test_graph_search_features.py tests/test_graph_experimental_design.py tests/test_graph_fixtures.py
python -m neural_search.graph.build_graph --datasets tests/fixtures/graph/normalized_datasets.jsonl --papers tests/fixtures/graph/normalized_papers.jsonl --out /tmp/fixture_graph.json
python -m neural_search.graph.reports --graph /tmp/fixture_graph.json --out /tmp/graph_reports
```

## Next Integration Step

Keep graph scoring optional. Retrieval can call
`compute_graph_features_for_result` and `graph_context_score` only when a graph
file is configured and available. The v0.4 retrieval score breakdown should
remain the primary ranking explanation.
