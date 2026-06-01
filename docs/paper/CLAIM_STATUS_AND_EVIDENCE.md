# Claim Status and Evidence

This document tracks the implementation and validation status of claims made in the Neural Search manuscript.

## Status Legend

- **Implemented**: Code exists and runs successfully
- **Validated**: Measured on benchmark with quantitative results
- **Partial**: Partially implemented, some components missing
- **Proposed**: Future work, not yet implemented

## Core Claims

| Claim | Status | Evidence | Remaining Work |
|-------|--------|----------|----------------|
| Hybrid metadata retrieval improves search quality | Validated | 30-query benchmark, ablation study showing +11.6% NDCG from ontology matching | Expand to 100+ queries, independent annotators |
| Hard-negative filtering reduces invalid matches | Implemented | Zero violations on benchmark queries with exclusions | Expand adversarial benchmark, test edge cases |
| Typed scientific labels improve interpretability | Implemented | Manual inspection, labels visible in search results | Quantitative user study |
| Graph metapaths improve cross-dataset relatedness | Partial | Graph infrastructure exists, PathSim implemented | Cross-dataset pairing benchmark |
| Analysis affordance search | Implemented | 16 affordance detectors with rule-based detection | Affordance validation rubric, false positive analysis |
| Reciprocal Rank Fusion improves over single retriever | Validated | Ablation shows RRF beats BM25-only and dense-only | Statistical significance testing |
| Field-weighted BM25 outperforms standard BM25 | Implemented | Weight configuration exists | Head-to-head comparison |
| Provenance-weighted confidence improves precision | Implemented | Confidence model defined, source hierarchy implemented | Ablation on confidence vs uniform weighting |

## Expansion Claims

| Claim | Status | Evidence | Remaining Work |
|-------|--------|----------|----------------|
| Latent neural signature search | Proposed | Schema defined in expansion roadmap | NWB feature extraction, embedding infrastructure |
| Cross-species alignment | Proposed | Ontology structure includes taxon hierarchy | Alignment benchmark, metapath scoring |
| Causal claim graph | Proposed | Not implemented | Full schema, paper extraction pipeline |
| Computational model integration | Proposed | Schema outlined | ModelDB/GitHub connector, metadata alignment |
| Dataset-to-paper linking | Implemented | DOI matching, OpenAlex integration | Expand coverage, citation graph traversal |

## Benchmark Claims

| Metric | Claimed Value | Evidence Source | Replication Info |
|--------|---------------|-----------------|------------------|
| Precision@5 | 76.7% | `data/eval/results/demo_v02/benchmark_report.json` | Run `python -m neural_search.evaluation.run_benchmark --suite demo_v02` |
| Recall@10 | 87.8% | Same as above | Same as above |
| MRR | 0.950 | Same as above | Same as above |
| NDCG@10 | 0.937 | Same as above | Same as above |
| Hard-Negative Violations | 0 | Same as above | Same as above |

## Ablation Claims

| Configuration | NDCG Delta | Evidence Source |
|--------------|------------|-----------------|
| - Ontology matching | -0.116 | `data/eval/results/demo_v02/ablation_report.json` |
| - Semantic embeddings | -0.075 | Same as above |
| - Graph features | -0.046 | Same as above |
| - BM25 lexical | -0.034 | Same as above |
| - Affordance scoring | -0.025 | Same as above |

## Evidence Quality Assessment

### High Confidence
- Benchmark query results (reproducible, versioned)
- Ablation study results (deterministic, configurable)
- Hard-negative constraint enforcement (zero violations observed)

### Medium Confidence
- Label quality (automated extraction with heuristics)
- Graph edge accuracy (derived from extraction, not curated)
- Affordance predictions (rule-based, not validated against ground truth)

### Low Confidence
- Cross-dataset pairing quality (infrastructure exists, no benchmark)
- Embedding generalization (not tested on novel terminology)
- Scalability claims (tested only on small corpus)

## Next Steps for Validation

1. **Independent annotation**: Recruit domain experts for blind relevance labeling
2. **Inter-annotator agreement**: Measure Krippendorff's alpha on overlapping queries
3. **Corpus expansion**: Increase to 500+ datasets from additional sources
4. **Adversarial benchmark**: Create 50+ hard-negative queries with complex exclusions
5. **Affordance validation**: Manual verification of predicted analysis support
6. **User study**: Measure task completion time with/without Neural Search

---

*Last updated: 2026-05-26*
*Manuscript version: v1.0*
