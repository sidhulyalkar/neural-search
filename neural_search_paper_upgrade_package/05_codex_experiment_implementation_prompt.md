# Codex Prompt: Experiment Implementation

Paste this into Codex.

```text
Inspect the repo and implement support for the next version of the paper’s experiments without changing the paper prose.

Assume tests currently pass.

Build feature-by-feature, with tests after each feature. Avoid broad rewrites. Keep changes modular and reviewable.

Priority 1: Benchmark baseline ladder

Add evaluation configs for:

1. keyword search
2. BM25
3. field-weighted BM25
4. dense-only retrieval
5. BM25 + dense reciprocal rank fusion
6. BM25 + dense + ontology expansion
7. BM25 + dense + ontology + graph features
8. full system with affordance scoring and hard-negative constraints

Output metrics:

- Precision@5
- Recall@10
- MRR
- NDCG@10
- hard-negative violations
- latency p50 / p95
- result diversity if feasible
- explanation completeness if feasible

Save outputs as:

- JSON
- Markdown
- machine-readable CSV or Parquet if the repo already uses tabular outputs

Priority 2: Hard-negative adversarial benchmark

Add a query set focused on NOT/exclusion constraints.

Include examples such as:

- mouse decision-making NOT visual cortex
- human EEG excluding fMRI
- Neuropixels motor cortex but not visual coding
- calcium imaging with behavior labels, excluding spontaneous-only data
- reward learning data but not Pavlovian conditioning

Add tests ensuring excluded species, modalities, regions, and task families do not appear in top-k results.

Report:

- violation type
- violating field
- result rank
- whether violation came from text, extracted metadata, ontology expansion, or graph edges

Priority 3: Affordance validation

Add a structured affordance rubric file.

Implement deterministic affordance checks for:

- choice decoding
- Q-learning
- state-space modeling
- neural-behavior alignment
- cross-session analysis
- causal perturbation analysis
- representational similarity analysis

Each affordance should have:

- required fields
- optional fields
- confidence scoring
- missing metadata explanation
- test cases for true positive, false positive, and incomplete metadata

Priority 4: Cross-dataset pairing

Implement a function that ranks dataset pairs by:

- task similarity
- modality compatibility
- species relationship
- brain-region relationship
- shared analysis affordance
- graph path confidence
- provenance confidence
- novelty/diversity score

Add tests for expected pair types:

- same task, different region
- same task, different species
- same modality, different task
- shared analysis affordance
- complementary modalities

Priority 5: Metadata robustness

Add perturbation utilities:

- drop fields
- corrupt synonyms
- remove paper links
- delete task labels
- delete modality labels
- delete species labels
- delete event labels
- remove provenance confidence

Measure retrieval degradation:

- P@5 delta
- NDCG@10 delta
- hard-negative violation increase
- graph rescue rate
- field importance ranking

Priority 6: Embedding model bakeoff scaffold

Add a clean interface for comparing embedding providers or embedding indexes.

Support config-driven comparison of:

- existing local embeddings
- dense embedding providers already supported by the repo
- scientific or biomedical embedding models if dependencies are already available

Do not add heavyweight dependencies without isolating them behind optional extras.

Priority 7: Graph link-prediction benchmark scaffold

Add a benchmark that can hold out known edges:

- dataset -> task
- dataset -> modality
- dataset -> brain region
- dataset -> paper
- dataset -> supports analysis

Start with heuristic baselines:

- node-degree baseline
- ontology-distance baseline
- metapath-count baseline
- path-confidence baseline

Only add learned graph embeddings if the repo already has a suitable dependency structure.

Priority 8: Report artifacts

Generate a single experiment report:

`reports/neural_search_experiment_report.md`

Include:

- run metadata
- corpus summary
- query summary
- baseline ladder table
- hard-negative violation table
- affordance validation table
- cross-dataset pairing examples
- robustness table
- paper-ready LaTeX table snippets

Do not fabricate results.

Implementation rules:

- Assume tests pass at the start.
- Add focused tests after each feature.
- Avoid large rewrites of the search stack.
- Prefer configuration-driven evaluation.
- Preserve existing public APIs unless a change is necessary.
- Put experimental scripts under the repo’s existing benchmark/evaluation layout.
- If no layout exists, create a minimal, clearly named structure:
  - `benchmarks/`
  - `benchmarks/configs/`
  - `benchmarks/queries/`
  - `benchmarks/rubrics/`
  - `benchmarks/reports/`
  - `tests/`

Final deliverables:

- Code changes for the evaluation pipeline
- Tests for each new feature
- A generated report with real results if runnable locally
- A summary of changes
- Any TODOs that require human labels or larger corpora
```
