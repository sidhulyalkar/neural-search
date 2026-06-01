# Retrieval and Embeddings Plan

## Retrieval philosophy

Neural Search should rank datasets by experiment compatibility, not only semantic similarity. Embeddings are useful, but they should be a kite string, not the whole kite.

Final score should combine:

- Ontology task match
- Behavior/event match
- Modality match
- Brain region match
- Species match
- Data standard/source match
- Analysis intent match
- Text/semantic similarity
- Readiness/provenance score
- QA trust boost
- Mismatch penalties

## Proposed score decomposition

```text
score =
  ontology_task_score      * w_task +
  behavior_event_score     * w_behavior +
  modality_score           * w_modality +
  brain_region_score       * w_region +
  species_score            * w_species +
  data_standard_score      * w_standard +
  analysis_intent_score    * w_analysis +
  semantic_score           * w_semantic +
  readiness_score          * w_readiness +
  paper_confidence_score   * w_paper +
  qa_score                 * w_qa -
  penalties
```

Every result should expose this decomposition in `why_matched` and optionally a structured `score_breakdown` object.

## Embedding provider design

Add:

```text
neural_search/embeddings/
  __init__.py
  base.py
  hashing.py
  sentence_transformers.py
  index.py
```

Interface:

```python
class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
```

Providers:

- `HashingEmbeddingProvider`: deterministic, no external downloads, CI-safe.
- `SentenceTransformerProvider`: optional, used only when `neural-search[embeddings]` is installed.

## Texts to embed

Generate separate fields:

- `dataset_title_text`
- `dataset_description_text`
- `ontology_label_text`
- `analysis_intent_text`
- `paper_evidence_text`
- `combined_search_text`

This allows ablations: title-only vs ontology-only vs combined.

## Benchmark expansion

Add hard negatives. A search system without hard negatives becomes a golden retriever in a costume: happy, confident, and not very discriminating.

Example:

```yaml
- id: human_ecog_bci_reaching
  query: Human ECoG or iEEG reaching data for BCI classification
  expected_modalities: [ecog, ieeg]
  expected_species: [human]
  expected_behaviors: [reaching, motor_control]
  hard_negative_modalities: [calcium_imaging, fiber_photometry]
  hard_negative_species: [mouse, rat]
  minimum_precision_at_5: 0.6
```

## Metrics

Report:

- Precision@1/3/5/10
- Recall@5/10 when expected IDs are known
- Label recall@10
- MRR
- NDCG@10
- False-positive count
- Hard-negative violations
- Missing metadata warnings among top results

## Ablations

For each benchmark run, compare:

- keyword only
- ontology only
- embeddings only
- ontology + metadata
- full hybrid

This will make the system scientifically defensible and gives you a great demo slide later.
