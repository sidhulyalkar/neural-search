# Knowledge Graph Schema

Neural Search v0.5 adds a lightweight, file-backed scientific knowledge graph.
The first backend lives in `neural_search.graph` and does not require Neo4j,
NetworkX, RDF, DuckDB, or any external service.

## Core Models

- `GraphEvidence`: provenance for a node or edge, including source record,
  source field, evidence text, extractor name/version, and confidence.
- `KnowledgeGraphNode`: typed scientific record or concept node.
- `KnowledgeGraphEdge`: directed provenance-backed relationship.
- `KnowledgeGraph`: dictionaries of nodes and edges plus metadata.

## Supported Serialization

- JSON graph document: `write_graph_json`, `read_graph_json`
- JSONL records: `write_graph_jsonl`, `read_graph_jsonl`
- Dict roundtrip: `graph_to_dict`, `graph_from_dict`

## Stable IDs

Examples:

```text
node:dataset:dandi:000026
node:paper:openalex:W123456789
node:task:reversal_learning
edge:dataset:dandi:000026:has_task:reversal_learning
edge:paper:openalex:W123:uses_dataset:dandi:000026
```

Use `make_node_id` and `make_edge_id`; do not assemble graph IDs by hand in new
code.

## Validation

`KnowledgeGraph` validation enforces non-empty IDs, normalized node/edge types,
confidence ranges, matching identity-map keys, and edge references that resolve
to existing graph nodes.
