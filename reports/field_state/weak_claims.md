# Weak Claims

Generated: 2026-06-11T01:53:34.656009+00:00

Claims are listed when they have low evidence, low confidence, or unresolved tests.

## Graph proximity may improve dataset-method matching.

- ID: `claim_graph_proximity`
- Evidence level: `hypothesis`
- Confidence: 0.55
- Status: `needs_validation`
- Human review: `unreviewed`
- Human status: `needs_validation`
- Source note: `Field-State/20_Claims/weak/Graph proximity may improve dataset-method matching.md`
- Related artifacts:
- docs/WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md
- docs/whitepaper/neural_search_whitepaper.tex
- Missing tests:
- Run graph proximity ablation on dataset-method compatibility queries.
- Check for spurious gains from popularity or metadata density.

## Dense semantic retrieval improves dataset matching.

- ID: `claim_dense_semantic_retrieval`
- Evidence level: `plausible`
- Confidence: 0.62
- Status: `needs_validation`
- Human review: `unreviewed`
- Human status: `needs_validation`
- Source note: `Field-State/20_Claims/weak/Dense semantic retrieval improves dataset matching.md`
- Related artifacts:
- data/corpus/normalized/combined_corpus.jsonl
- reports/eval
- docs/WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md
- docs/whitepaper/neural_search_whitepaper.tex
- Missing tests:
- Compare dense retrieval against lexical and metadata-only baselines on human qrels.
- Measure whether gains hold for dataset-method compatibility queries.

## Analysis affordance extraction improves dataset recommendation.

- ID: `claim_affordance_extraction`
- Evidence level: `plausible`
- Confidence: 0.64
- Status: `needs_validation`
- Human review: `unreviewed`
- Human status: `needs_validation`
- Source note: `Field-State/20_Claims/weak/Analysis affordance extraction improves dataset recommendation.md`
- Related artifacts:
- data/corpus/normalized/combined_corpus.jsonl
- reports/eval
- docs/WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md
- docs/whitepaper/neural_search_whitepaper.tex
- Missing tests:
- Build a small gold benchmark for analysis affordances.
- Evaluate recommendation quality with and without affordance features.

## Metadata richness affects dataset reuse potential.

- ID: `claim_metadata_richness`
- Evidence level: `plausible`
- Confidence: 0.66
- Status: `needs_validation`
- Human review: `unreviewed`
- Human status: `needs_validation`
- Source note: `Field-State/20_Claims/weak/Metadata richness affects dataset reuse potential.md`
- Related artifacts:
- data/corpus/normalized/combined_corpus.jsonl
- reports/eval
- Missing tests:
- Define metadata richness features.
- Estimate relationship between metadata richness and reuse/readiness labels.

## Hard negatives are necessary for scientific retrieval evaluation.

- ID: `claim_hard_negatives`
- Evidence level: `supported`
- Confidence: 0.72
- Status: `partially_tested`
- Human review: `unreviewed`
- Human status: `partially_tested`
- Source note: `Field-State/20_Claims/weak/Hard negatives are necessary for scientific retrieval evaluation.md`
- Related artifacts:
- reports/eval
- Missing tests:
- Track top-k hard-negative violations by query type.
- Separate near-miss datasets from clearly irrelevant datasets.

## Human qrels are required for credible nDCG/MRR metrics.

- ID: `claim_human_qrels`
- Evidence level: `plausible`
- Confidence: 0.78
- Status: `needs_validation`
- Human review: `unreviewed`
- Human status: `needs_validation`
- Source note: `Field-State/20_Claims/weak/Human qrels are required for credible nDCGMRR metrics.md`
- Related artifacts:
- reports/eval
- Missing tests:
- Create expert-labeled qrels for dataset-method compatibility.
- Report inter-annotator agreement or adjudication notes.
