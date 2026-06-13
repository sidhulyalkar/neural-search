# Retrieval Loading Diagnostics

## Timings
- `corpus_load_s`: N/A
- `embedding_cache_load_s`: 0.6 ms
- `turbovec_index_load_s`: N/A
- `legacy_graph_load_s`: N/A
- `memory_graph_load_s`: 20.3 ms
- `query_parse_avg_s`: 851.5 ms

## Artifact Sizes
- `embedding_cache_rows`: 0
- `memory_graph_nodes`: 770
- `memory_graph_edges`: 180

## Warnings
- ⚠️ corpus load failed: [Errno 21] Is a directory: 'data/corpus/normalized/combined_corpus.jsonl'
- ⚠️ turbovec index not found at data/index/turbovec_dense_1024.index