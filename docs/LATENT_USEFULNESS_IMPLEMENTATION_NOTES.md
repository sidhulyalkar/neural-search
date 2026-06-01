# Latent Usefulness v0.8 Implementation Notes

## What Was Built

### New Modules

| Module | Path | Purpose |
|--------|------|---------|
| UsefulnessIntent classifier | `neural_search/retrieval/query_intent.py` | Rule-based intent classification (7 intents, no LLM) |
| Usefulness scorer | `neural_search/retrieval/usefulness_scorer.py` | 10-dimension intent-weighted scoring |
| Graph usefulness signals | `neural_search/retrieval/graph_usefulness.py` | PathSim hub-normalized + complementarity |
| Usefulness benchmark | `neural_search/evaluation/usefulness_benchmark.py` | Graded NDCG/MRR/P@k/hard-neg violation |
| Affordance validation v2 | `neural_search/evaluation/affordance_validation_v2.py` | Precision/recall vs ground truth |
| Ablation runner | `neural_search/evaluation/ablation_runner.py` | 8 retrieval variant comparison |

### Data and Config
- `data/eval/usefulness_seed_pairs.jsonl` — 30 seed pairs across 6 usefulness categories
- `config/eval/usefulness_v08.yaml` — evaluation configuration

### Reports Generated
- `reports/usefulness_benchmark_v08.md` — baseline benchmark on seed pairs
- `reports/ablation_v08.md` — 8-variant ablation comparison

## Key Design Decisions

### 1. `UsefulnessIntent` vs existing `QueryIntent`
The new `UsefulnessIntent` enum at `neural_search/retrieval/query_intent.py` targets *latent usefulness relationships* (replication, pipeline_reuse, method_transfer, etc.), while the existing `neural_search/search/intent.py` `QueryIntent` handles retrieval-head weight overrides (modality_search, dataset_lookup, etc.). Both coexist without conflict.

### 2. `DatasetContext` as neutral carrier
The scorer accepts plain `DatasetContext` dataclasses instead of requiring `DatasetCardV1` or `NormalizedDatasetRecord`. This decouples the scorer from the schema layer. Callers convert their schema to `DatasetContext` before scoring.

### 3. `neural_signature_similarity` fixed at 0.0
Placeholder pending Phase 3 neural signature search. Scores a warning into `UsefulnessScore.warnings`. For `EXPLORATION` intent, this dimension has 0.16 weight — so exploration scores are systematically lower until this is implemented.

### 4. `graph_proximity` uses neutral prior 0.3
Without a live graph, returns 0.3 + warning. `graph_usefulness.py` provides the full PathSim implementation for use when a graph is available. To integrate: pass the graph dict to `graph_usefulness_features` and replace the 0.3 prior with the computed score.

### 5. Ablation uses scoring proxies
The 8 ablation variants are implemented as different scoring functions over `DatasetContext` objects — no external BM25/dense retrieval infrastructure required. This allows deterministic testing on seed data without downloads.

## Limitations

1. **Seed-only benchmark**: All results are on 30 synthetic seed pairs. No validated real corpus labels exist yet. Whitepaper claims from this phase must be marked "preliminary."
2. **Proxy ablation variants**: The `bm25_only`, `dense_only`, `graph_only` variants are proxies (e.g., bm25_only = task+modality Jaccard), not actual retrieval systems.
3. **`neural_signature_similarity` unimplemented**: Contributes 0 to all scores. This is a significant gap for `EXPLORATION` intent where it has 16% weight.
4. **No claim of generalization**: Results should not be claimed to generalize across neuroscience until corpus and labels are expanded beyond DANDI seed pairs.

## Integration Guide

To use the usefulness scorer in the existing retrieval pipeline:

```python
from neural_search.retrieval import (
    classify_query_intent,
    DatasetContext,
    score_usefulness,
)

# 1. Classify user intent
intent_cls = classify_query_intent(user_query)

# 2. Convert candidate to DatasetContext
ctx = DatasetContext(
    dataset_id=candidate.id,
    modalities=[e.label for e in candidate.modalities],
    tasks=[e.label for e in candidate.tasks],
    species=[e.label for e in candidate.species],
    brain_regions=[e.label for e in candidate.brain_regions],
    affordances=[a.affordance_id for a in candidate.affordances],
    data_standards=[str(s) for s in candidate.data_standards],
    quality_score=candidate.quality_score or 0.0,
)

# 3. Score
score = score_usefulness(query_ctx, ctx, intent_cls.intent)
```

## Next Steps

1. **Phase 3**: Implement `neural_signature_similarity` using `NeuralSignatureV1` schema
2. **Graph integration**: Pass real `KnowledgeGraph.model_dump()` to `graph_usefulness_features` and feed result back into `score_usefulness`
3. **Label expansion**: Expand seed pairs to 200+ with real DANDI dataset pairs
4. **Feedback learning**: Implement Phase 4 feedback-driven weight learning
5. **DatasetCardV1 converter**: Add `DatasetContext.from_card(card: DatasetCardV1)` classmethod
