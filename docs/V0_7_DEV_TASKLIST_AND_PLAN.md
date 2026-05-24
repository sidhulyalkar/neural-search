# v0.7 Developer Tasklist and Plan

## Goal

Move Neural Search from a canonical demo corpus into a real public corpus with file inspection, evidence-backed claims, and artifact builds that stay reproducible without network or paid model dependencies in CI.

## Phase 1: Real Corpus Intake

- [ ] Define a versioned real-corpus manifest for DANDI, OpenNeuro, OpenAlex, and manually curated landmark datasets.
- [ ] Add source-specific ingestion runs that write raw payloads under `data/raw/<source>/`.
- [ ] Normalize 50 to 100 DANDI datasets into JSONL.
- [ ] Normalize 50 to 100 OpenNeuro datasets into JSONL.
- [ ] Normalize 500 to 2000 paper records into JSONL.
- [ ] Add source-level ingestion reports for fetched, normalized, skipped, failed, and missing metadata counts.

## Phase 2: File Inspection Layer

- [ ] Add a file-inspection claim schema for evidence-backed dataset claims.
- [ ] Inspect NWB metadata for trials, units, electrodes, devices, processing modules, and intervals.
- [ ] Inspect BIDS metadata for `dataset_description.json`, `participants.tsv`, `events.tsv`, channel/electrode tables, and sidecars.
- [ ] Generate claims for trial structure, event timestamps, neural modality coverage, behavior coverage, data standard compliance, and analysis readiness.
- [ ] Persist inspection claims as JSONL and include claim IDs in normalized records.

## Phase 3: Retrieval Integration

- [ ] Extend corpus converter/build commands to support `demo_v05`, `real_v07`, and custom corpus tags.
- [ ] Build graph artifacts from real normalized records and file-inspection claims.
- [ ] Build field embedding caches for real corpus records using hashing by default.
- [ ] Add retrieval config presets for demo, real-corpus local, and CI.
- [ ] Add result fields for linked papers, graph context, missing metadata, and file-inspection evidence where available.

## Phase 4: Evaluation

- [ ] Add a `real_v07` benchmark suite with expected dataset IDs where known.
- [ ] Add benchmark coverage for paper-to-dataset linking, analysis affordance search, and graph reasoning.
- [ ] Report hard-negative violation rate, label recall, graph-link precision, explanation coverage, MRR, and NDCG.
- [ ] Add regression tests that ensure explicit negative constraints return zero violations.

## Phase 5: Developer Ergonomics

- [ ] Add `make real-corpus-build`.
- [ ] Add `make real-graph-build`.
- [ ] Add `make real-embeddings-build`.
- [ ] Add `make real-artifacts-build`.
- [ ] Add a dry-run mode for ingestion and artifact builds.
- [ ] Add a short troubleshooting guide for missing credentials, rate limits, malformed records, and skipped files.

## Phase 6: Task 7 AI Research Workflows

- [ ] Create agent-facing workflow schemas for dataset discovery, paper-to-dataset linking, dataset comparison, benchmark audit, experimental design, notebook generation, and gap analysis.
- [ ] Add graph, provenance, missing-metadata, and hard-negative summaries to workflow outputs.
- [ ] Add deterministic tests for Codex-owned workflow APIs and reports.
- [ ] Coordinate with Claude on workflow language, scientific examples, and relevance rubrics.
- [ ] Track kickoff work in `docs/TASK_7_AI_WORKFLOW_KICKOFF.md`.
- [ ] Preserve the broader architecture in `docs/AI_RESEARCH_WORKFLOW_ROADMAP.md`.

## Phase 7: Task 8 Search Improvement

- [ ] Add a query intent router with intent-specific retrieval profiles.
- [ ] Improve label and synonym recall using benchmark failure reports.
- [ ] Tune field embedding weights by intent while keeping hashing as the CI default.
- [ ] Make graph reranking explanations more explicit.
- [ ] Preserve 0 hard-negative violations while improving Precision@5 and label recall.
- [ ] Track implementation in `docs/TASK_8_SEARCH_IMPROVEMENT_PLAN.md`.

## Quality Gates

```bash
pytest -q
ruff check neural_search tests
make artifacts-build
python -m neural_search.evaluation.run_benchmark --suite demo_v02
python -m neural_search.evaluation.run_benchmark --suite adversarial
```

For v0.7 real corpus work, add:

```bash
make real-artifacts-build
python -m neural_search.evaluation.run_benchmark --suite real_v07
```

## Open Engineering Decisions

- Whether real corpus raw payloads should be committed, sampled, or generated only locally.
- Whether file-inspection claims should live inside normalized records or as a linked sidecar artifact.
- Whether graph reports should summarize by corpus tag and timestamp.
- How much real-corpus benchmark labeling should be automated versus manually reviewed.
- Whether Task 8 search improvement should land before, during, or immediately after real-corpus ingestion.
