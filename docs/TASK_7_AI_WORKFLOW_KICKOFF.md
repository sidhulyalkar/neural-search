# Task 7 AI Workflow Kickoff

## Purpose

Task 7 starts the AI research workflow track. The existing roadmap in `docs/AI_RESEARCH_WORKFLOW_ROADMAP.md` gives the broad agent architecture; this kickoff checklist turns that roadmap into near-term Claude and Codex work.

## Shared Goal

Make Neural Search usable by research agents that need audit-ready dataset discovery, paper linking, dataset comparison, benchmark audit, experimental design, notebook planning, and corpus gap analysis.

## Codex Start List

- [ ] Define stable agent-facing response schemas for the workflows below.
- [ ] Add deterministic tests for each Codex-owned workflow endpoint or report.
- [ ] Reuse existing search, graph, comparison, notebook, benchmark, and report modules before adding new abstractions.
- [ ] Include provenance, missing metadata, hard-negative summaries, graph context, and score breakdowns where available.
- [ ] Keep workflows functional when graph or embedding artifacts are absent.

## Claude Start List

- [ ] Provide scientific example prompts for each workflow.
- [ ] Define relevance rubrics for exact, strong, partial, wrong task, wrong modality, wrong species, missing required data, violates negative constraint, and unclear.
- [ ] Review workflow language so generated explanations stay scientifically careful.
- [ ] Identify benchmark cases that should exercise each workflow.

## Workflow Checklist

### Dataset Discovery

- [ ] Return ranked datasets with score breakdowns, matched labels, missing metadata, graph context, and linked papers.
- [ ] Expose filtered hard-negative constraints in response metadata.
- [ ] Add tests for graph-enabled and graph-missing behavior.

### Paper-to-Dataset Linking

- [ ] Accept paper title, DOI, OpenAlex ID, abstract, or methods text.
- [ ] Return candidate datasets with link type, confidence, evidence, and graph paths.
- [ ] Add paper-dataset link precision metrics.

### Dataset Comparison

- [ ] Extend comparison output with graph overlap and file-inspection claims when available.
- [ ] Add export tests for agent-facing JSON.

### Benchmark Audit

- [ ] Generate failed-query summaries.
- [ ] Classify likely failure causes: ontology, corpus, ranking, constraints, graph, or benchmark expectation.
- [ ] Surface hard-negative violation rate and filtered-result counts.

### Experimental Design

- [ ] Expose experimental design graph seed matching through an API or report.
- [ ] Return prior datasets, prior papers, missing requirements, candidate controls, and graph-grounded rationale.

### Notebook Generation

- [ ] Recommend notebook templates in search and dataset workflow outputs.
- [ ] Preserve deterministic notebook generation tests.

### Gap Analysis

- [ ] Extend graph/corpus reports with underrepresented labels, missing metadata, paper-linked gaps, and ingestion priorities.

## First Codex Implementation Slice

1. Add workflow response schema models under `neural_search/schemas.py` or a focused workflow module.
2. Add a dataset discovery workflow wrapper around `search_datasets`.
3. Add benchmark audit report generation from existing benchmark JSON output.
4. Add tests for deterministic dataset discovery and benchmark audit behavior.

## Acceptance Criteria

- [ ] Task 7 docs are linked from the v0.7 plan.
- [ ] At least one workflow is implemented end to end.
- [ ] Every workflow response includes enough provenance or missing-data context to be auditable.
- [ ] `pytest -q` and `ruff check neural_search tests` pass.
