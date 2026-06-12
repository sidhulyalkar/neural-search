# Field-State Memory Graph

The field-state memory graph is the primary relational substrate for neural-search dataset discovery. It stores every dataset, paper, concept, method, affordance, judgment, feedback signal, and snapshot in a single versioned, exportable graph that the retrieval pipeline can query directly.

## Schema

### Node Types

All nodes are instances of `KnowledgeGraphNode` (defined in `neural_search/graph/schema.py`).

| Node type | Role |
|---|---|
| `dataset` | A normalized neuroscience dataset. One node per `dataset_id`. |
| `source_archive` | A data archive (DANDI, OpenNeuro, IBL, etc.) |
| `paper` | A linked publication |
| `modality` | A recording modality (extracellular_ephys, calcium_imaging, fmri, …) |
| `species` | A species label (mouse, rat, human, macaque, …) |
| `brain_region` | A brain region (hippocampus_ca1, lateral_intraparietal, …) |
| `task` | A behavioral or cognitive task |
| `behavioral_event` | A task event type |
| `data_standard` | A data standard (NWB, BIDS, ALF, …) |
| `file_format` | A specific file format |
| `analysis_affordance` | An analysis that a dataset supports or lacks |
| `raw_data_signal` | Evidence that raw continuous data is present |
| `processed_data_signal` | Evidence that only processed/derived data is available |
| `file_artifact` | A specific file artifact within a dataset |
| `concept` | A concept-memory concept node |
| `pipeline` | An analysis pipeline |
| `query` | A stored retrieval query |
| `query_intent` | A query intent class |
| `retrieval_run` | A logged retrieval run |
| `neuro_judge_evidence_packet` | A neuro-judge evidence packet (silver, not gold) |
| `neuro_judge_judgment` | A neuro-judge judgment output (silver, not gold) |
| `feedback_signal` | A user feedback event (downstream signal, not gold) |
| `curation_issue` | A flagged data quality issue |
| `snapshot_manifest` | A versioned snapshot manifest |

Every node carries: `node_id`, `node_type`, `label`, `aliases`, `source_ids`, `properties`, `evidence`, `confidence`, `created_at`, `updated_at`.

### Edge Types

Key edge types added for the field-state graph (in addition to the existing 38 scientific edges):

| Edge type | Meaning |
|---|---|
| `dataset_from_source` | Dataset belongs to a source archive |
| `dataset_linked_to_paper` | Dataset is linked to a paper |
| `dataset_has_raw_signal` | Dataset has raw data evidence |
| `dataset_has_processed_signal` | Dataset has processed-only data evidence |
| `dataset_lacks_required_evidence` | Dataset is missing evidence for an affordance |
| `dataset_contraindicated_for` | Dataset should not be used for a query/task |
| `judgment_labels_query_dataset` | A judgment labels a (query, dataset) pair |
| `feedback_marks_result` | A feedback event marks a retrieval result |
| `snapshot_contains_node/edge` | A snapshot references a graph element |

### Provenance model

Every node carries a `properties` dict with provenance fields:
- `provenance`: string indicating how the node was created (`"neuro_judge_silver_not_human_gold"`, `"user_feedback_downstream_signal"`, etc.)
- `inferred`: `True` if the value was extracted from free text rather than structured metadata
- `confidence`: 0.0–1.0; structured metadata gets ≥0.7, text-inferred gets ≤0.6

## Storage

Artifacts live in `artifacts/field_state/`:
- `memory_graph_nodes.jsonl` — one `KnowledgeGraphNode` per line
- `memory_graph_edges.jsonl` — one `KnowledgeGraphEdge` per line
- `memory_graph_manifest.json` — build summary with node/edge counts by type
- `current_manifest.json` — pointer to the latest snapshot
- `snapshots/<timestamp>/` — versioned snapshot directory

## Graph store API

```python
from neural_search.field_state.graph_store import FieldStateGraphStore

# Load
store = FieldStateGraphStore.from_jsonl(
    Path("artifacts/field_state/memory_graph_nodes.jsonl"),
    Path("artifacts/field_state/memory_graph_edges.jsonl"),
)

# Query
datasets = store.query_datasets()
node = store.query_by_dataset_id("dataset:dandi:000026")
neighbors = store.get_neighbors(node.node_id, edge_types=["dataset_has_modality"])
missing = store.query_datasets_missing_evidence()

# Validate
errors = store.validate_invariants()

# Export
store.export_jsonl(nodes_path, edges_path)
store.write_manifest(manifest_path, build_id="build_001")
```
