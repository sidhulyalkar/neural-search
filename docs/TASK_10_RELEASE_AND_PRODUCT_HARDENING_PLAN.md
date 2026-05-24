# Task 10 Release and Product Hardening Plan

## Purpose

Task 10 turns the improved research engine into a release-ready system for repeated use by humans and AI agents. The focus is API stability, frontend clarity, reproducible builds, observability, documentation, and quality gates.

## Target Outcomes

- Stable agent-facing and frontend-facing APIs.
- Search results that clearly expose scores, evidence, graph context, missing metadata, and filtered constraints.
- Reproducible artifact builds for demo and real-corpus modes.
- A product UI that supports scientific workflows without hiding uncertainty.
- Release docs that make onboarding and validation straightforward.

## Ownership

- Codex owns API contracts, frontend integration, build quality, release checks, tests, and CI-friendly scripts.
- Claude owns product language, scientific UX review, onboarding docs, demo narratives, and evaluation interpretation.

## Workstream A: API Contract Stabilization

Codex tasks:

- [ ] Define versioned response schemas for search, dataset cards, graph context, paper links, workflow outputs, and benchmark audits.
- [ ] Add compatibility tests for existing public API fields.
- [ ] Add explicit optional fields for `graph_context`, `linked_papers`, `field_semantic_score`, `graph_score`, `filtered_constraints`, and `missing_metadata`.
- [ ] Add OpenAPI examples for major workflows.

Claude tasks:

- [ ] Review API field names for scientific clarity.
- [ ] Define user-facing descriptions for uncertainty and missing metadata fields.

Acceptance criteria:

- [ ] Existing API behavior remains backward compatible.
- [ ] Agent-facing payloads are documented and tested.
- [ ] Missing optional artifacts do not produce API errors.

## Workstream B: Frontend Workflow Integration

Codex tasks:

- [ ] Update search UI to show score breakdowns without overwhelming users.
- [ ] Show hard-negative filtered summaries when constraints are applied.
- [ ] Add graph context and linked papers to dataset result cards.
- [ ] Add missing metadata and analysis affordance panels to dataset pages.
- [ ] Add benchmark/audit report views for search quality debugging.

Claude tasks:

- [ ] Review result-card language for scientific care.
- [ ] Define UI copy for provenance, uncertainty, and analysis readiness.
- [ ] Create demo walkthrough scripts for major workflows.

Acceptance criteria:

- [ ] UI exposes why a result matched and what may be missing.
- [ ] UI does not imply unsupported scientific claims.
- [ ] Core workflows work on desktop and mobile.

## Workstream C: Release Quality Gates

Codex tasks:

- [ ] Add `make release-check`.
- [ ] Run tests, lint, artifact builds, benchmark suites, and report generation.
- [ ] Add a release summary generator that records commit, artifact counts, benchmark metrics, and known failures.
- [ ] Add checks for stale artifacts and missing generated files.

Claude tasks:

- [ ] Review release summaries for scientific regressions.
- [ ] Define acceptable metric thresholds for release candidates.

Acceptance criteria:

- [ ] One command produces a release-readiness report.
- [ ] Release checks fail on hard-negative violations.
- [ ] Release checks record benchmark metrics in machine-readable JSON.

## Workstream D: Observability and Debugging

Codex tasks:

- [ ] Add structured debug output for parsed query, intent, filters, score heads, graph features, and field matches.
- [ ] Add a search trace object that can be stored or exported.
- [ ] Add benchmark audit links from failed queries to search traces.
- [ ] Add performance timing for retrieval heads and artifact loading.

Claude tasks:

- [ ] Define a failure taxonomy that maps traces to likely scientific causes.
- [ ] Review trace outputs for user-safe language.

Acceptance criteria:

- [ ] Developers can explain a bad result from a saved trace.
- [ ] Benchmark reports identify which retrieval head likely failed.
- [ ] Debug output does not require external services.

## Workstream E: Documentation and Onboarding

Codex tasks:

- [ ] Add setup instructions for demo artifacts and real-corpus fixture artifacts.
- [ ] Add API examples for agent-facing workflows.
- [ ] Add troubleshooting docs for artifact builds, ingestion, graph loading, and embeddings.
- [ ] Keep command examples copy-pasteable from repo root.

Claude tasks:

- [ ] Write a product-level walkthrough for dataset discovery, paper linking, experimental design, and notebook planning.
- [ ] Explain the scientific value of provenance, graph reasoning, and analysis affordances.

Acceptance criteria:

- [ ] A new developer can run the demo workflow from a clean checkout.
- [ ] A scientific user can understand result explanations and limitations.
- [ ] Docs identify which commands are CI-safe and which need network access.

## Workstream F: Release Candidate Packaging

Codex tasks:

- [ ] Tag artifact versions with corpus tag, graph version, embedding provider, and build timestamp.
- [ ] Add release notes template.
- [ ] Add changelog entries for retrieval, graph, corpus, workflows, UI, and benchmarks.
- [ ] Ensure generated artifacts that should be committed are listed explicitly.

Claude tasks:

- [ ] Write release narrative and known limitations.
- [ ] Review benchmark changes for scientific interpretability.

Acceptance criteria:

- [ ] Release notes include features, metrics, artifacts, known issues, and next tasks.
- [ ] Artifact versions are traceable back to a commit.
- [ ] Release candidate can be validated without paid providers.

## Quality Gates

```bash
pytest -q
ruff check neural_search tests
make artifacts-build
python -m neural_search.evaluation.run_benchmark --suite demo_v02
python -m neural_search.evaluation.run_benchmark --suite adversarial
make release-check
```

## First Implementation Slice

1. Add `make release-check`.
2. Add release summary JSON and Markdown output under `data/reports/release/`.
3. Add search trace object for parsed query, filters, score heads, graph score, field score, and filtered constraints.
4. Add frontend rendering for `graph_score`, `field_semantic_score`, and filtered hard negatives.
