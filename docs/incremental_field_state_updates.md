# Incremental Field-State Updates

## Daily update workflow

The field-state update pipeline detects changed/new/removed records, rebuilds only what changed, and versions the result as a snapshot.

```bash
# Full rebuild (force)
python scripts/field_state/update_field_state.py \
  --corpus data/corpus/normalized/combined_corpus.jsonl \
  --force

# Incremental update (only changed records)
python scripts/field_state/update_field_state.py \
  --corpus data/corpus/normalized/combined_corpus.jsonl \
  --changed-only

# Dry run — detect changes without writing
python scripts/field_state/update_field_state.py \
  --corpus data/corpus/normalized/combined_corpus.jsonl \
  --dry-run

# Skip expensive stages
python scripts/field_state/update_field_state.py \
  --corpus data/corpus/normalized/combined_corpus.jsonl \
  --skip-embeddings \
  --skip-neuro-judge
```

## Change detection

Changes are detected by computing a SHA-256 content hash over `{title, description, source_id, source}` for each record. If the hash differs from the previous manifest, the record is flagged as changed.

## Snapshot manifests

Each run writes a versioned snapshot to `artifacts/field_state/snapshots/<timestamp>/`:
- `corpus_manifest.json` — record hashes and change counts
- `memory_graph_manifest.json` — node/edge counts by type
- `index_manifest.json` — which stages were skipped
- `update_report.md` — human-readable summary

`artifacts/field_state/current_manifest.json` always points to the latest snapshot.

## Snapshot comparison

```bash
python scripts/field_state/compare_snapshots.py \
  --old artifacts/field_state/snapshots/20260611T000000Z \
  --new artifacts/field_state/snapshots/20260612T000000Z \
  --out-dir reports/field_state
```

Outputs `snapshot_diff.md` and `snapshot_diff.json` with:
- Datasets added/removed/changed
- Graph node/edge count changes by type
- Warnings for large unexpected changes (>10% removal)

## Learning signals update

```bash
python scripts/field_state/update_learning_signals.py \
  --feedback artifacts/frontend/retrieval_feedback.jsonl \
  --judgments artifacts/field_state/neuro_qrels_consensus.jsonl
```

All signals are provisional downstream signals, never gold labels. Output: `artifacts/field_state/learning_signals.jsonl`.
