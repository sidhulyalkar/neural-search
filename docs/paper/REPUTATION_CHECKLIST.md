# Reputation and Submission Readiness Checklist

This document tracks readiness for submission to a peer-reviewed venue.

## Claim Integrity

| Requirement | Status | Notes |
|-------------|--------|-------|
| No unsupported "state-of-the-art" claims | DONE | Removed/softened in revision |
| Benchmark numbers verified reproducible | DONE | Scripts in data/eval/ |
| Ablation results traceable to code | DONE | run_ablation.py |
| Limitations section present | DONE | Expanded in revision |
| Future work distinguished from implemented | DONE | Claim status table added |
| Confidence intervals reported | DONE | Bootstrap 95% CI in results |
| Hard-negative violations explicitly tracked | DONE | Zero violations documented |

## Experimental Rigor

| Requirement | Status | Notes |
|-------------|--------|-------|
| Benchmark construction documented | DONE | Protocol section added |
| Query categories specified | DONE | 6 categories, 30 queries |
| Relevance labeling rubric | PARTIAL | 0-3 scale described, inter-annotator agreement pending |
| Reproducibility instructions | DONE | Commands in EXPERIMENT_ROADMAP.md |
| Error taxonomy | DONE | Added to benchmark protocol |
| Statistical significance | PARTIAL | CI reported, p-values pending |
| Baseline comparisons | PARTIAL | Ablation vs baselines, external baselines pending |

## Technical Completeness

| Requirement | Status | Notes |
|-------------|--------|-------|
| All algorithms described | DONE | Formalization section |
| Hyperparameters documented | PARTIAL | retrieval.yaml exists, not all in paper |
| Model architecture clear | DONE | Hybrid scoring, RRF fusion |
| Graph schema complete | DONE | Schema tables added |
| Affordance definitions | DONE | Affordance table added |
| Scoring formula explicit | DONE | Theorem in formalization |

## Related Work

| Requirement | Status | Notes |
|-------------|--------|-------|
| IR baselines cited | DONE | BM25, dense, hybrid |
| KG methods cited | DONE | PathSim, metapath2vec, TransE |
| Domain infrastructure cited | DONE | DANDI, OpenNeuro, NWB, BIDS |
| Metadata standards cited | DONE | Schema.org, DataCite, PROV-O added |
| European infrastructure cited | DONE | EBRAINS, openMINDS added |

## Formatting

| Requirement | Status | Notes |
|-------------|--------|-------|
| Float specifiers corrected | DONE | [h] -> [t] |
| No overfull hbox warnings | DONE | Checked long sequences |
| tabularx available | DONE | Package added |
| References complete | PARTIAL | Some TODO entries |
| Figures vector graphics | DONE | TikZ throughout |
| Color-blind friendly | PARTIAL | Not explicitly verified |

## Reproducibility Artifacts

| Artifact | Status | Location |
|----------|--------|----------|
| Benchmark queries | DONE | data/eval/benchmark_queries.yaml |
| Relevance labels | DONE | data/eval/relevance_labels_v01.jsonl |
| Retrieval config | DONE | data/config/retrieval.yaml |
| Evaluation scripts | DONE | neural_search/evaluation/ |
| Report generation | DONE | run_benchmark.py, run_ablation.py |
| Graph building | DONE | neural_search/graph/ |

## Known Gaps

### High Priority (Before Submission)

1. **Inter-annotator agreement**: Need second annotator for subset
2. **External baselines**: Compare to Elasticsearch, simple embedding search
3. **Corpus expansion**: Current N~150 is small
4. **User study**: No usability evaluation yet

### Medium Priority (Revision Round)

1. **Embedding bakeoff**: Compare models systematically
2. **Scalability benchmarks**: Test on larger corpus
3. **Affordance validation**: Manual verification of predictions
4. **Hard-negative expansion**: 50+ adversarial queries

### Lower Priority (Future Work)

1. **Latent neural search**: Prototype only
2. **Cross-species benchmark**: No ground truth yet
3. **Computational model integration**: Not implemented

## Venue-Specific Requirements

### For ICLR/NeurIPS

- [ ] Broader impact statement
- [ ] Ethics review if human subjects involved
- [ ] Checklist completion (NeurIPS)

### For Neuroinformatics Journals

- [ ] Data availability statement
- [ ] Software citation (CITATION.cff)
- [ ] RRID for tools/resources

### For CS/IR Venues (SIGIR, CIKM)

- [ ] Baseline ladder with standard IR metrics
- [ ] Statistical significance tests
- [ ] Efficiency benchmarks

---

## Sign-off

- [ ] Technical review complete
- [ ] Writing review complete
- [ ] All claims verified
- [ ] Reproducibility verified
- [ ] Submission package assembled

---

*Last updated: 2026-05-26*
