# Task 8 Search Improvement Plan

## Purpose

Task 8 is the next shared Claude and Codex track after Task 7. Its goal is to make Neural Search noticeably better at scientific retrieval quality, not just broader in corpus size.

## Target Outcomes

- Improve precision and recall on demo and adversarial benchmark suites.
- Reduce ambiguity failures around tasks, modalities, analysis affordances, species, regions, and sources.
- Make graph and field-semantic scores useful but explainable.
- Preserve hard-negative violation rate at 0 for explicit constraints.
- Keep default CI behavior deterministic and hashing-based.

## Workstream A: Query Intent and Routing

Codex tasks:

- [ ] Add `neural_search/search/intent.py`.
- [ ] Return structured intent, confidence, positive constraints, negative constraints, and retrieval profile.
- [ ] Route between dataset lookup, paper lookup, analysis affordance search, graph reasoning, and exploratory search.
- [ ] Add retrieval config profiles per intent.

Claude tasks:

- [ ] Define examples for each intent class.
- [ ] Identify confusing query patterns and intended behavior.
- [ ] Add benchmark cases for ambiguous and multi-intent queries.

## Workstream B: Label and Synonym Recall

Codex tasks:

- [ ] Add tests for ontology misses surfaced by benchmark reports.
- [ ] Add safe alias plumbing without broad ontology rewrites.
- [ ] Ensure labels are normalized consistently across seed, normalized corpus, graph, and search.

Claude tasks:

- [ ] Expand scientific synonyms for missed benchmark concepts.
- [ ] Separate vague natural-language labels from evidence-backed canonical labels.
- [ ] Review modality aliases such as EEG versus iEEG and ECoG.

## Workstream C: Field Embedding Quality

Codex tasks:

- [ ] Add field-specific retrieval debug output.
- [ ] Add per-field weights by intent.
- [ ] Add ablation report comparing lexical, ontology, field semantic, graph, and combined retrieval.
- [ ] Keep hashing provider as the default for CI.

Claude tasks:

- [ ] Define which fields matter most for each scientific intent.
- [ ] Identify cases where semantic embeddings should not override explicit metadata.

## Workstream D: Graph-Augmented Ranking

Codex tasks:

- [ ] Add `neural_search/search/graph_rerank.py`.
- [ ] Add graph feature explanations for linked papers, shared concepts, and analysis affordances.
- [ ] Add graph-link precision and graph explanation coverage metrics.
- [ ] Ensure missing graph artifacts never break search.

Claude tasks:

- [ ] Define graph-path relevance examples.
- [ ] Review which graph links should affect ranking versus only explanation.

## Workstream E: Hard-Negative Safety

Codex tasks:

- [ ] Add regression tests for every explicit negative pattern in benchmark suites.
- [ ] Add parser coverage for quoted negatives and comma-separated negatives.
- [ ] Track filtered-result counts in benchmark reports.

Claude tasks:

- [ ] Add adversarial queries for clinically risky and modality-confounded searches.
- [ ] Define disallowed false-positive categories for scientific use cases.

## Workstream F: Evaluation and Reports

Codex tasks:

- [ ] Add score-head summaries to benchmark reports.
- [ ] Add per-query error classification.
- [ ] Add constraint violation rate and hard-negative violation rate to top-level reports.
- [ ] Add `make search-quality-report`.

Claude tasks:

- [ ] Review failed queries and classify whether the issue is benchmark, ontology, corpus, or ranking.
- [ ] Define human relevance labeling examples for top-k search outputs.

## Search Quality Acceptance Criteria

- [ ] `pytest -q` passes.
- [ ] `ruff check neural_search tests` passes.
- [ ] `python -m neural_search.evaluation.run_benchmark --suite demo_v02` passes.
- [ ] `python -m neural_search.evaluation.run_benchmark --suite adversarial` passes.
- [ ] Hard-negative violation rate remains 0.
- [ ] `demo_v02` and `adversarial` mean Precision@5 improve from the v0.6 baseline or each degraded query has a documented reason.
- [ ] Search result explanations identify graph, field-semantic, ontology, and hard-negative behavior clearly when those heads are active.

## v0.6 Baseline

- `demo_v02` Precision@5: 0.787.
- `demo_v02` Label recall@10: 0.878.
- `adversarial` Precision@5: 0.800.
- `adversarial` Label recall@10: 0.766.
- Hard-negative violations: 0 in both suites.
