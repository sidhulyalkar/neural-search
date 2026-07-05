# Real Corpus v0.7 Runbook

## CI-Safe Fixture Build

From the repo root:

```bash
make real-artifacts-build
python -m neural_search.evaluation.run_benchmark --suite real_v07
python -m neural_search.release.check --summary-only
```

This uses only local fixtures under `data/corpus/fixtures/real_v07/` and hashing
embeddings. It does not require network access or paid embedding providers.

## Dry-Run Ingestion

```bash
python -m neural_search.corpus.ingest_manifest \
  --manifest data/corpus/manifests/real_v07.yaml \
  --out data/corpus/normalized \
  --dry-run
```

Dry-run validates the manifest and reports what would be normalized.

## Generated Artifacts

- Normalized records: `data/corpus/normalized/real_v07.*.jsonl`
- File claims: `data/corpus/claims/real_v07.claims.jsonl`
- Graph: `data/graph/neural_search_graph.real_v07.json`
- Field embeddings: `data/embeddings/real_v07.field_embeddings.jsonl`
- Reports: `data/reports/real_v07/`
- Release summary: `data/reports/release/release_summary.*`

## Troubleshooting

- Missing fixture paths produce warning claims instead of false scientific labels.
- If graph loading is disabled or an artifact is absent, search remains usable without graph scores.
- If field embeddings are absent, hashing embeddings can be regenerated with `make real-embeddings-build`.
- `make release-check` fails if required artifacts are missing or older than their source manifest/fixture inputs.
- Search traces can be exported with `python -m neural_search.search.trace "mouse visual decision making" --out data/reports/release/example_trace.json`.
- Full public-source ingestion should be layered behind the manifest and raw payload writer so CI remains deterministic.
