# v0.7 Developer Tasklist and Plan

## Goal

Move Neural Search from a canonical demo corpus into a real public corpus with file inspection, evidence-backed claims, and artifact builds that stay reproducible without network or paid model dependencies in CI.

## Phase 1: Real Corpus Intake

- [x] Define a versioned real-corpus manifest for DANDI, OpenNeuro, OpenAlex, and manually curated landmark datasets.
- [x] Add source-specific ingestion runs that write raw payloads under `data/raw/<source>/`.
- [ ] Normalize 50 to 100 DANDI datasets into JSONL.
- [ ] Normalize 50 to 100 OpenNeuro datasets into JSONL.
- [ ] Normalize 500 to 2000 paper records into JSONL.
- [x] Add source-level ingestion reports for fetched, normalized, skipped, failed, and missing metadata counts.

## Phase 2: File Inspection Layer

- [x] Add a file-inspection claim schema for evidence-backed dataset claims.
- [x] Inspect NWB metadata for trials, units, electrodes, devices, processing modules, and intervals.
- [x] Inspect BIDS metadata for `dataset_description.json`, `participants.tsv`, `events.tsv`, channel/electrode tables, and sidecars.
- [x] Generate claims for trial structure, event timestamps, neural modality coverage, behavior coverage, data standard compliance, and analysis readiness.
- [x] Persist inspection claims as JSONL and include claim IDs in normalized records.

## Phase 3: Retrieval Integration

- [x] Extend corpus converter/build commands to support `demo_v05`, `real_v07`, and custom corpus tags.
- [x] Build graph artifacts from real normalized records and file-inspection claims.
- [x] Build field embedding caches for real corpus records using hashing by default.
- [ ] Add retrieval config presets for demo, real-corpus local, and CI.
- [x] Add result fields for linked papers, graph context, missing metadata, and file-inspection evidence where available.

## Phase 4: Evaluation

- [x] Add a `real_v07` benchmark suite with expected dataset IDs where known.
- [ ] Add benchmark coverage for paper-to-dataset linking, analysis affordance search, and graph reasoning.
- [x] Report hard-negative violation rate, label recall, explanation coverage, MRR, and NDCG.
- [x] Add regression tests that ensure explicit negative constraints return zero violations.

## Phase 5: Developer Ergonomics

- [x] Add `make real-corpus-build`.
- [x] Add `make real-graph-build`.
- [x] Add `make real-embeddings-build`.
- [x] Add `make real-artifacts-build`.
- [x] Add a dry-run mode for ingestion and artifact builds.
- [x] Add a short troubleshooting guide for missing credentials, rate limits, malformed records, and skipped files.

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

## Phase 8: Task 9 Real Corpus and File Inspection

- [x] Add a versioned real-corpus manifest for DANDI, OpenNeuro, OpenAlex, and landmark datasets.
- [x] Add file-inspection claim schemas for NWB and BIDS metadata evidence.
- [x] Build claim-aware normalized records, graph artifacts, reports, and field embeddings.
- [x] Add real-corpus fixture builds that remain CI-safe.
- [x] Add `real_v07` benchmarks and claim-backed explanation metrics.
- [x] Track implementation in `docs/TASK_9_REAL_CORPUS_AND_FILE_INSPECTION_PLAN.md`.

## Phase 9: Task 10 Release and Product Hardening

- [x] Stabilize API contracts for search, graph context, workflow outputs, paper links, and benchmark audits.
- [x] Update frontend-ready payloads to expose score breakdowns, graph context, missing metadata, linked papers, and filtered constraints.
- [x] Add release quality gates, release summaries, and artifact checks.
- [x] Add search traces and benchmark debugging outputs.
- [x] Improve onboarding docs, release notes, and product walkthroughs.
- [x] Track implementation in `docs/TASK_10_RELEASE_AND_PRODUCT_HARDENING_PLAN.md`.

## Phase 10: Task 13 General Neuroscience Search Intelligence

- [x] Add a deterministic neuroscience data-form taxonomy.
- [x] Infer query awareness for data forms, analysis families, scales, species, and exclusions.
- [x] Score dataset awareness separately from the main retrieval score.
- [x] Add corpus awareness reports for coverage and underrepresented data forms.
- [x] Add an awareness-aware search wrapper with optional reranking while concurrent Task 12 search-core work settles.
- [ ] Wire awareness scoring into the main retrieval path after concurrent Task 12 search-core work settles.
- [ ] Add real-corpus benchmarks for fMRI, MEG, connectomics, molecular, clinical, and computational-model searches.
- [x] Track implementation in `docs/TASK_13_GENERAL_NEUROSCIENCE_SEARCH_INTELLIGENCE.md`.

## Phase 11: Task 14 Awareness-Integrated Retrieval

- [x] Preserve the existing `search_datasets` API and add a reversible awareness wrapper.
- [x] Expose `query_awareness`, `awareness_score`, and data-form evidence in search responses.
- [x] Support opt-in awareness reranking with bounded weights.
- [x] Add tests for annotation, reranking, and warning propagation.
- [ ] Promote awareness scoring into the primary retrieval core once Task 12 semantic changes are stable.
- [x] Track implementation in `docs/TASK_14_AWARENESS_INTEGRATED_RETRIEVAL.md`.

## Phase 12: Task 17 Search Intelligence Planner

- [x] Convert query awareness into deterministic retrieval plans.
- [x] Recommend mode-specific weights for hard negatives, cross-modal fit, analysis readiness, graph similarity, exact lookup, and exploratory search.
- [x] Emit required signals, complementary data forms, quality checks, warnings, and benchmark tags.
- [x] Add CLI and Make target for inspecting planner output.
- [ ] Integrate planner-selected profiles into the main retrieval core after Task 12 semantic work settles.
- [x] Track implementation in `docs/TASK_17_SEARCH_INTELLIGENCE_PLANNER.md`.

## Phase 13: Task 18 Coverage-Driven Search Expansion

- [x] Compare normalized corpus data-form coverage against benchmark-query coverage.
- [x] Produce prioritized ingestion and benchmark gaps with query seeds and source recommendations.
- [x] Add JSON and Markdown report writing plus CLI support.
- [x] Generate human-reviewable benchmark seed YAML from coverage gaps.
- [ ] Add this report to the standard artifact pipeline after benchmark labels expand.
- [x] Track implementation in `docs/TASK_18_COVERAGE_DRIVEN_SEARCH_EXPANSION.md`.

## Phase 14: Task 19 Retrieval Planner Integration

- [x] Add bridge-level retrieval config plumbing for planner integration.
- [x] Expose planner metadata in parsed query through `search_datasets_with_intelligence`.
- [x] Blend planner weights with existing retrieval config conservatively.
- [ ] Validate demo, adversarial, and real_v07 benchmarks before enabling by default.
- [x] Track preparation in `docs/TASK_19_RETRIEVAL_PLANNER_INTEGRATION.md`.

## Phase 15: Task 20 Human Relevance and Active Learning Loop

- [x] Define/load human relevance judgments for query-result pairs.
- [x] Generate review queues from coverage gaps and benchmark seed queries.
- [x] Report human-labeled metrics separately from synthetic benchmark expectations.
- [ ] Use labels to prioritize corpus and benchmark expansion before ranking changes.
- [x] Track preparation in `docs/TASK_20_HUMAN_RELEVANCE_AND_ACTIVE_LEARNING.md`.

## Phase 16: Task 21 Query Plan Evaluation Harness

- [x] Compare baseline, awareness, and intelligence retrieval for benchmark query files.
- [x] Evaluate against normalized real_v07 records instead of demo fallback when records are supplied.
- [x] Group metric deltas by planner intent.
- [x] Block promotion when hard-negative violations increase.
- [x] Add JSON and Markdown query-plan evaluation reports.
- [x] Track implementation in `docs/TASK_21_QUERY_PLAN_EVALUATION_HARNESS.md`.

## Phase 17: Task 22 Retrieval Default Promotion

- [x] Add an intent-specific promotion manifest.
- [x] Add deterministic promotion gate reports from query-plan evaluation output.
- [x] Add global human-label gates for promotion readiness.
- [ ] Define CI/local/exploratory config presets.
- [ ] Promote planner metadata into standard search traces after Task 21 review.
- [x] Track implementation in `docs/TASK_22_RETRIEVAL_DEFAULT_PROMOTION.md`.

## Phase 18: Task 23 Adaptive Realistic Search Optimization

- [x] Add per-intent human-label gates once reviewed judgment files exist.
- [x] Add data-form grouped query-plan evaluation.
- [x] Add score calibration reports against human relevance judgments.
- [x] Add realistic fixtures for underrepresented neuroscience data forms.
- [ ] Track planning in `docs/TASK_23_ADAPTIVE_REALISTIC_SEARCH_OPTIMIZATION.md`.

## Phase 19: Task 24 Corpus and Knowledge Base Expansion Plan

- [x] Add a regeneratable corpus and knowledge-base expansion report.
- [x] Convert data-form coverage gaps into source, concept, benchmark, and graph-edge tasks.
- [x] Add `make corpus-knowledge-plan`.
- [ ] Review the generated plan before promoting any new ranking defaults.
- [ ] Track planning in `docs/TASK_24_CORPUS_AND_KNOWLEDGE_BASE_EXPANSION.md`.

## Phase 20: Task 25 Knowledge Graph Enrichment

- [ ] Add graph coverage gates for required scientific node and edge types.
- [ ] Expand dataset-paper linking beyond weak linked ID fields.
- [x] Add analysis requirement edges from data forms to modalities, events, standards, and signals.
- [x] Add a graph requirement report for analysis-to-requirement edges.
- [x] Add requirement-aware graph context to retrieval explanations when graph config is enabled.
- [ ] Add source-specific graph provenance summaries for DANDI, OpenNeuro, OpenAlex, ModelDB, cellxgene, MICrONS, and curated records.
- [ ] Track planning in `docs/TASK_25_KNOWLEDGE_GRAPH_ENRICHMENT.md`.

## Phase 21: Task 26 Source Intake Scaling

- [x] Add fixture-backed source families for ModelDB, cellxgene, and MICrONS.
- [ ] Add fixture-backed curated landmark records for remaining benchmark-critical gaps.
- [ ] Expand real-corpus local intake targets by data form before tuning ranking weights.
- [ ] Keep network-backed source expansion optional outside CI.
- [ ] Add source-balance checks to release and search-intelligence reports.
- [ ] Track planning in `docs/TASK_26_SOURCE_INTAKE_SCALING.md`.

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
make release-check
make awareness-report
```

## Open Engineering Decisions

- Whether real corpus raw payloads should be committed, sampled, or generated only locally.
- Whether file-inspection claims should live inside normalized records or as a linked sidecar artifact.
- Whether graph reports should summarize by corpus tag and timestamp.
- How much real-corpus benchmark labeling should be automated versus manually reviewed.
- Whether Task 8 search improvement should land before, during, or immediately after real-corpus ingestion.
- Which real-corpus artifacts are small and stable enough to commit.
- Whether release checks should block on per-query benchmark failures or only aggregate thresholds.
