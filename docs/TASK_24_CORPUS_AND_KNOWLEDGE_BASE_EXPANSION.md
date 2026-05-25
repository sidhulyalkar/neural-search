# Task 24: Corpus and Knowledge Base Expansion

Status: in progress

## Goal

Turn corpus coverage gaps into concrete expansion tasks that improve general neuroscience search across datasets, papers, graph concepts, analysis affordances, and benchmark labels.

## Why This Matters

Search quality cannot be fixed only by reranking. The engine needs representative normalized records and graph relationships for the forms of neuroscience people actually ask about: physiology, imaging, clinical recordings, connectomics, molecular assays, computational models, behavior, and paper evidence.

## Implementation Slices

- [x] Add `neural_search.intelligence.expansion` for deterministic expansion planning.
- [x] Generate tasks from data-form coverage gaps, source balance, and optional graph artifacts.
- [x] Write `corpus_knowledge_expansion_plan.json` and `.md` reports.
- [x] Add `make corpus-knowledge-plan`.
- [ ] Review generated tasks and turn the highest-priority gaps into fixture-backed normalized records.
- [ ] Add expected dataset IDs and hard negatives for new benchmark seeds after record review.

## Expansion Targets

- Dataset coverage: at least 5 reviewed records per major neuroscience data form before ranking default promotion.
- Benchmark coverage: at least 3 reviewed queries per major data form, with explicit hard negatives for exclusion-heavy searches.
- Source coverage: DANDI, OpenNeuro, OpenAlex, ModelDB, cellxgene, MICrONS, and curated landmark records.
- Knowledge coverage: modality, task, species, analysis affordance, data standard, dataset-paper links, and analysis requirement edges.

## Acceptance Criteria

- The expansion plan can be regenerated from local normalized records and optional graph artifacts.
- Each generated task includes target sources, graph/concept targets, and acceptance checks.
- The plan separates corpus gaps from ranking problems.
- CI remains deterministic and uses local fixtures plus hashing embeddings by default.
