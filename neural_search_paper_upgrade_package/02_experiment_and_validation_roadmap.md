# Experiment and Validation Roadmap

This roadmap is designed to turn Neural Search from a promising prototype into a credible research artifact. The goal is not only to improve search quality, but to demonstrate why each layer of the system matters.

## Experiment 1: Baseline ladder

### Goal

Measure whether each retrieval layer improves scientific relevance, constraint satisfaction, and explainability.

### Systems to compare

1. Keyword search
2. BM25
3. Field-weighted BM25
4. Dense-only retrieval
5. BM25 + dense reciprocal rank fusion
6. BM25 + dense + ontology expansion
7. BM25 + dense + ontology + graph features
8. Full system with affordance scoring and hard-negative constraints

### Metrics

- Precision@5
- Recall@10
- MRR
- NDCG@10
- hard-negative violation rate
- latency p50 / p95
- explanation completeness
- result diversity

### Expected paper contribution

This becomes the core empirical table. It shows which layer earns its keep.

## Experiment 2: Hard-negative adversarial benchmark

### Goal

Test whether Neural Search avoids scientifically invalid matches.

### Query examples

```text
mouse decision-making NOT visual cortex
human EEG excluding fMRI
Neuropixels motor cortex but not visual coding
calcium imaging with behavior labels, excluding spontaneous-only data
reward learning data but not Pavlovian conditioning
macaque electrophysiology with natural images, not rodent behavior
hippocampal replay datasets excluding calcium imaging
```

### Violation categories

- species violation
- modality violation
- brain-region violation
- task-family violation
- format violation
- unsupported analysis affordance
- missing required metadata
- ambiguous or insufficient provenance

### Metrics

- violation rate at top 1, top 5, top 10
- false-positive source field
- whether graph edges amplified or corrected the error
- whether ontology expansion caused overgeneralization

### Expected paper contribution

This is a high-value experiment because scientific search is often less about finding something vaguely relevant and more about avoiding invalid comparisons.

## Experiment 3: Analysis affordance validation

### Goal

Validate that predicted analysis affordances correspond to real analyzability.

### Initial affordances

1. Choice decoding
2. Q-learning
3. State-space modeling
4. Neural-behavior alignment
5. Cross-session analysis
6. Causal perturbation analysis
7. Representational similarity analysis

### Validation protocol

For each dataset, label whether the dataset truly supports each affordance.

Labels:

- 0: unsupported
- 1: weakly supported or requires major manual repair
- 2: supported with caveats
- 3: directly supported

### Metrics

- affordance precision
- affordance recall
- F1
- calibration error
- false affordance rate
- missing-affordance rate

### Expected paper contribution

This validates the most distinctive idea in Neural Search: retrieval by what a dataset can scientifically support, not only what it mentions.

## Experiment 4: Cross-dataset pairing benchmark

### Goal

Evaluate whether the system can find scientifically useful dataset pairs.

### Pairing categories

| Pair type | Example |
|---|---|
| Same task, different region | decision-making PFC versus striatum |
| Same modality, different task | Neuropixels visual coding versus decision-making |
| Same task, different species | mouse reward learning versus primate reward learning |
| Same analysis affordance | two datasets supporting Q-learning |
| Same latent construct | inhibition, reward prediction, working memory |
| Same model benchmark potential | datasets suitable for encoding model comparison |
| Complementary modalities | calcium imaging plus ephys for similar task context |

### Scoring features

- task similarity
- modality compatibility
- species relationship
- brain-region relationship
- shared affordance
- graph path confidence
- provenance confidence
- novelty or diversity score

### Metrics

- pair relevance
- reason correctness
- graph path validity
- novelty score
- human usefulness rating

### Expected paper contribution

This experiment supports the claim that Neural Search can connect datasets into reusable scientific context graphs.

## Experiment 5: Metadata robustness perturbation

### Goal

Measure how search quality degrades when metadata is incomplete or noisy.

### Perturbations

- remove descriptions
- remove task labels
- remove modality labels
- remove species labels
- remove paper links
- corrupt region fields
- replace terms with synonyms
- insert ambiguous abbreviations
- delete event labels
- remove provenance confidence

### Metrics

- retrieval degradation
- P@5 delta
- NDCG@10 delta
- hard-negative violation increase
- graph rescue rate
- field importance ranking

### Expected paper contribution

This demonstrates robustness and reveals which metadata fields are most valuable.

## Experiment 6: Embedding model bakeoff

### Goal

Evaluate which embedding families best capture neuroscience dataset meaning.

### Candidates

- general sentence-transformer
- SciBERT-style scientific text encoder
- BioBERT-style biomedical encoder
- SPECTER2-style scientific document encoder
- OpenAI embedding model
- local field-specific embeddings
- late-interaction retrieval model, if implemented

### Query categories

- modality synonym queries
- brain-region synonym queries
- task synonym queries
- abbreviation-heavy queries
- cross-species queries
- analysis-affordance queries
- hard-negative exclusion queries

### Metrics

- P@5
- Recall@10
- NDCG@10
- synonym robustness
- abbreviation handling
- hard-negative violation rate
- embedding latency

### Expected paper contribution

This tells readers whether general semantic embeddings are enough or whether neuroscience-specific retrieval requires field-aware structure.

## Experiment 7: Graph link-prediction benchmark

### Goal

Test whether graph structure can infer missing scientific relationships.

### Held-out edges

- dataset -> task
- dataset -> modality
- dataset -> brain region
- dataset -> paper
- dataset -> supports analysis
- task -> cognitive construct
- modality -> compatible analysis method

### Baselines

- random edge baseline
- node-degree heuristic
- ontology distance
- metapath count
- TransE-style graph embedding
- metapath2vec-style heterogeneous embedding
- heterogeneous graph attention, if implemented

### Metrics

- Hits@K
- MRR
- AUC
- precision at high confidence
- calibration error

### Expected paper contribution

This supports the transition from search over records to search over a scientific knowledge graph.

## Experiment 8: Latent neural signature search prototype

### Goal

Move beyond metadata search into search over measured neural dynamics.

### Minimal prototype

Select a small set of NWB datasets and extract standardized summary signatures:

- firing rate statistics
- inter-spike interval statistics
- trial-aligned PSTHs
- population covariance
- neural-behavior mutual information
- participation ratio or effective dimensionality
- event-aligned modulation indices
- spectral features for LFP/EEG
- calcium transient statistics for imaging data

### Signature object

```python
latent_signature = {
    "dataset_id": str,
    "modality": str,
    "species": str,
    "brain_region": str,
    "task_context": list[str],
    "population_dynamics": np.ndarray,
    "event_modulation": dict,
    "behavior_alignment": dict,
    "quality_metrics": dict,
}
```

### Search examples

```text
find datasets with event-aligned reward modulation
find datasets with low-dimensional population trajectories during choice
find datasets with neural dynamics similar to reward prediction error signals
find datasets with strong neural-behavior alignment during decision-making
```

### Metrics

- biological plausibility of nearest neighbors
- agreement with metadata-based similarity
- ability to recover known task/modality groupings
- novelty of discovered pairings

### Expected paper contribution

This is the future jewel: search by what the brain activity does, not just what the dataset says.

## Experiment 9: Causal claim graph prototype

### Goal

Represent causal claims and perturbation evidence as searchable graph objects.

### Causal claim schema

Fields:

- intervention
- target
- outcome
- population
- time scale
- control condition
- randomization
- confounders
- estimand
- evidence source
- confidence

### Query examples

```text
find datasets where perturbation of PFC affects reward-guided choice
find experiments with optogenetic intervention and behavioral outcome controls
find causal evidence for hippocampal involvement in replay-driven planning
```

### Metrics

- claim extraction precision
- claim graph edge correctness
- intervention/outcome matching
- false causal implication rate

### Expected paper contribution

This extends Neural Search from dataset retrieval toward evidence-aware experimental design.

## Experiment 10: Human-in-the-loop dataset recommendation study

### Goal

Measure whether the system helps researchers discover useful datasets faster.

### Protocol

Ask domain experts to solve dataset discovery tasks using:

1. repository search only
2. keyword search only
3. Neural Search without graph explanations
4. Neural Search with graph explanations

### Metrics

- time to useful dataset
- number of valid datasets found
- number of invalid datasets selected
- confidence in result
- explanation usefulness
- novelty of discovered connection

### Expected paper contribution

This turns the system from metric-improving to user-validated.

## Recommended experiment order

1. Baseline ladder
2. Hard-negative adversarial benchmark
3. Affordance validation
4. Cross-dataset pairing benchmark
5. Metadata robustness
6. Embedding bakeoff
7. Graph link prediction
8. Latent neural signature prototype
9. Causal claim graph
10. Human-in-the-loop study

## Minimum publishable experimental package

For a strong first paper, prioritize:

- baseline ladder
- hard-negative adversarial benchmark
- affordance validation
- cross-dataset pairing
- metadata robustness

The latent neural signature and causal claim graph can be framed as future work unless implemented on a small prototype corpus.
