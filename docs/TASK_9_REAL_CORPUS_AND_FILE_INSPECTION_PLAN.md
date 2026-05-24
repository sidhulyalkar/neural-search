# Task 9 Real Corpus and File Inspection Plan

## Purpose

Task 9 moves Neural Search from demo artifacts to a real public neuroscience corpus with evidence-backed file inspection. The work should preserve the canonical artifact path established in v0.6 while adding real DANDI, OpenNeuro, OpenAlex, and manually curated records.

## Target Outcomes

- Ingest and normalize a useful real corpus slice.
- Inspect available file metadata for NWB and BIDS datasets.
- Convert file observations into provenance-backed claims.
- Feed claims into normalized records, graph artifacts, field embeddings, reports, and search results.
- Keep CI deterministic by using sampled fixtures and hashing embeddings.

## Ownership

- Codex owns ingestion CLIs, file inspection parsers, schemas, artifact builds, tests, and reports.
- Claude owns source prioritization, scientific claim definitions, benchmark examples, and quality review rubrics.

## Workstream A: Real Corpus Manifest

Codex tasks:

- [ ] Add `data/corpus/manifests/real_v07.yaml`.
- [ ] Include source, source_id, expected type, priority, status, fetch metadata, and review notes.
- [ ] Support manifest tags such as `dandi`, `openneuro`, `openalex`, `landmark`, `behavior`, `ephys`, `imaging`, `clinical`, and `bci`.
- [ ] Add a manifest loader with validation tests.

Claude tasks:

- [ ] Select initial high-value DANDI, OpenNeuro, and landmark datasets.
- [ ] Annotate why each source belongs in the first real-corpus slice.
- [ ] Identify scientific domains that must be represented.

Acceptance criteria:

- [ ] Manifest validates deterministically.
- [ ] Manifest can drive dry-run ingestion.
- [ ] Each manifest entry has a source, source_id, priority, and scientific rationale.

## Workstream B: Source Ingestion Runs

Codex tasks:

- [ ] Add `python -m neural_search.corpus.ingest_manifest --manifest ... --out ... --dry-run`.
- [ ] Write raw payloads under `data/raw/<source>/<source_id>.json` when enabled.
- [ ] Normalize source records into `data/corpus/normalized/real_v07.*.jsonl`.
- [ ] Track fetched, normalized, skipped, failed, and warning counts.
- [ ] Add source-specific failure fixtures for malformed payloads.

Claude tasks:

- [ ] Define minimum metadata expectations by source.
- [ ] Review skipped/failure categories for scientific importance.

Acceptance criteria:

- [ ] Ingestion can run in dry-run mode without network access.
- [ ] Tests use local fixtures only.
- [ ] Source failures are reported without crashing the whole run.

## Workstream C: File Inspection Claims

Codex tasks:

- [ ] Add file-inspection claim schema with claim_id, dataset_id, claim_type, field, value, confidence, evidence, source_path, extractor, and timestamp.
- [ ] Add `neural_search/file_inspection/` package.
- [ ] Inspect NWB metadata for trials, units, electrodes, devices, processing modules, intervals, subject, sessions, and acquisition groups.
- [ ] Inspect BIDS metadata for `dataset_description.json`, `participants.tsv`, `events.tsv`, channels/electrodes TSV files, sidecars, and derivatives.
- [ ] Persist claims to `data/corpus/claims/real_v07.claims.jsonl`.

Claude tasks:

- [ ] Define claim categories and disallowed overclaims.
- [ ] Provide examples of high-confidence versus low-confidence inspection evidence.

Acceptance criteria:

- [ ] Claims are evidence-backed and machine-validated.
- [ ] Claims include confidence and source path.
- [ ] Missing files produce warnings, not false claims.

## Workstream D: Claim-Aware Normalization

Codex tasks:

- [ ] Link file-inspection claims to normalized dataset records.
- [ ] Use claims to update usability flags, missing fields, and analysis affordances conservatively.
- [ ] Add graph edges from datasets to claim-derived concepts when confidence passes threshold.
- [ ] Include claim summaries in graph reports.

Claude tasks:

- [ ] Review whether each claim should affect analysis readiness, graph edges, or explanation only.

Acceptance criteria:

- [ ] Claim-derived labels remain distinguishable from metadata-derived labels.
- [ ] Graph evidence points back to claim IDs or source fields.
- [ ] Search explanations can show file-inspection evidence when available.

## Workstream E: Real Corpus Artifacts

Codex tasks:

- [ ] Add `make real-corpus-build`.
- [ ] Add `make real-claims-build`.
- [ ] Add `make real-graph-build`.
- [ ] Add `make real-embeddings-build`.
- [ ] Add `make real-artifacts-build`.
- [ ] Add reports under `data/reports/real_v07/`.

Claude tasks:

- [ ] Review real-corpus reports for obvious scientific gaps.
- [ ] Identify next ingestion priorities based on reports.

Acceptance criteria:

- [ ] Real artifacts can be built from a local fixture subset in CI.
- [ ] Full real-corpus build can be run locally with network/data access.
- [ ] Hashing embeddings remain the default build provider.

## Workstream F: Real Corpus Benchmarks

Codex tasks:

- [ ] Add `real_v07` benchmark suite registration.
- [ ] Add metrics for file-claim coverage and claim-backed explanation coverage.
- [ ] Ensure benchmark reports include corpus tag and artifact paths.

Claude tasks:

- [ ] Create real-corpus benchmark queries with expected labels, expected dataset IDs where known, and hard negatives.
- [ ] Add human relevance review instructions for real-corpus top-k outputs.

Acceptance criteria:

- [ ] `python -m neural_search.evaluation.run_benchmark --suite real_v07` runs.
- [ ] Hard-negative violation rate remains 0 for explicit constraints.
- [ ] Failed real-corpus queries produce actionable audit output.

## Quality Gates

```bash
pytest -q
ruff check neural_search tests
make artifacts-build
make real-artifacts-build
python -m neural_search.evaluation.run_benchmark --suite demo_v02
python -m neural_search.evaluation.run_benchmark --suite adversarial
python -m neural_search.evaluation.run_benchmark --suite real_v07
```

## Risks

- Public source APIs may change or rate-limit requests.
- File downloads can be too large for routine development.
- Metadata may be sparse, inconsistent, or scientifically ambiguous.
- File inspection must avoid turning weak evidence into strong claims.

## First Implementation Slice

1. Add manifest schema and local fixture manifest.
2. Add claim schema and tiny NWB/BIDS fixture inspectors.
3. Add `real_v07` artifact commands that run on local fixtures.
4. Add reports showing source coverage, claim coverage, and missing metadata.
