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

- [x] Define versioned response schemas for search, dataset cards, graph context, paper links, workflow outputs, and benchmark audits.
- [x] Add compatibility tests for existing public API fields.
- [x] Add explicit optional fields for `graph_context`, `linked_papers`, `field_semantic_score`, `graph_score`, `filtered_constraints`, and `missing_metadata`.
- [x] Add OpenAPI examples for major workflows.

Claude tasks:

- [ ] Review API field names for scientific clarity.
- [ ] Define user-facing descriptions for uncertainty and missing metadata fields.

Acceptance criteria:

- [x] Existing API behavior remains backward compatible.
- [x] Agent-facing payloads are documented and tested.
- [x] Missing optional artifacts do not produce API errors.

## Workstream B: Frontend Workflow Integration

Codex tasks:

- [x] Update response contracts so the search UI can show score breakdowns without overwhelming users.
- [x] Expose hard-negative filtered summaries when constraints are applied.
- [x] Expose graph context and linked papers to dataset result cards.
- [x] Expose missing metadata and analysis affordances for dataset pages.
- [x] Generate benchmark/release reports for search quality debugging.

Claude tasks:

- [ ] Review result-card language for scientific care.
- [ ] Define UI copy for provenance, uncertainty, and analysis readiness.
- [ ] Create demo walkthrough scripts for major workflows.

Acceptance criteria:

- [x] API payloads expose why a result matched and what may be missing.
- [x] Evidence-backed claims remain labeled so UI language can avoid unsupported claims.
- [x] Core payloads remain compatible with desktop/mobile frontend clients.

## Workstream C: Release Quality Gates

Codex tasks:

- [x] Add `make release-check`.
- [x] Run tests, lint, artifact builds, benchmark suites, and report generation.
- [x] Add a release summary generator that records commit, artifact counts, benchmark metrics, and known failures.
- [x] Add checks for missing generated files.

Claude tasks:

- [ ] Review release summaries for scientific regressions.
- [ ] Define acceptable metric thresholds for release candidates.

Acceptance criteria:

- [x] One command produces a release-readiness report.
- [x] Release checks fail on hard-negative violations.
- [x] Release checks record benchmark metrics in machine-readable JSON.

## Workstream D: Observability and Debugging

Codex tasks:

- [x] Add structured debug output for parsed query, intent, filters, score heads, graph features, and field matches.
- [x] Add a search trace object that can be stored or exported.
- [x] Benchmark audit outputs are compatible with saved trace payloads.
- [x] Add performance timing for parse and search execution.

Claude tasks:

- [ ] Define a failure taxonomy that maps traces to likely scientific causes.
- [ ] Review trace outputs for user-safe language.

Acceptance criteria:

- [x] Developers can explain a bad result from a saved trace.
- [x] Benchmark and release reports identify failures and affected suites.
- [x] Debug output does not require external services.

## Workstream E: Documentation and Onboarding

Codex tasks:

- [x] Add setup instructions for demo artifacts and real-corpus fixture artifacts.
- [x] Add API examples for agent-facing workflows.
- [x] Add troubleshooting/release docs for artifact builds, ingestion, graph loading, and embeddings.
- [x] Keep command examples copy-pasteable from repo root.

Claude tasks:

- [ ] Write a product-level walkthrough for dataset discovery, paper linking, experimental design, and notebook planning.
- [ ] Explain the scientific value of provenance, graph reasoning, and analysis affordances.

Acceptance criteria:

- [x] A new developer can run the demo and real fixture workflows from a clean checkout.
- [x] A scientific user can understand result explanations and limitations.
- [x] Docs identify CI-safe fixture commands.

## Workstream F: Release Candidate Packaging

Codex tasks:

- [x] Tag artifact versions with corpus tag, graph version, embedding provider, and build timestamp.
- [x] Add release notes template.
- [x] Add changelog entries for retrieval, graph, corpus, workflows, API/UI payloads, and benchmarks.
- [x] Ensure generated artifacts that should be committed are listed explicitly.

Claude tasks:

- [ ] Write release narrative and known limitations.
- [ ] Review benchmark changes for scientific interpretability.

Acceptance criteria:

- [x] Release notes include features, metrics, artifacts, known issues, and next tasks.
- [x] Artifact versions are traceable back to a commit.
- [x] Release candidate can be validated without paid providers.

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

1. [x] Add `make release-check`.
2. [x] Add release summary JSON and Markdown output under `data/reports/release/`.
3. [x] Add search trace object for parsed query, filters, score heads, graph score, field score, and filtered constraints.
4. [x] Expose frontend-ready payload fields for `graph_score`, `field_semantic_score`, and filtered hard negatives.
