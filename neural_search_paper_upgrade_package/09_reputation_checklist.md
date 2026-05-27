# Neural Search Paper Reputation Checklist

Use this checklist before treating the paper as submission-ready.

## Claims

- [ ] Every major claim is labeled as implemented, measured, partially implemented, proposed, or speculative.
- [ ] No unsupported “state-of-the-art” claims remain.
- [ ] Results are not overstated beyond the current benchmark.
- [ ] Future work is clearly separated from current implementation.
- [ ] The abstract does not imply experiments that have not been run.

## Benchmark

- [ ] Corpus size is reported.
- [ ] Query categories are listed.
- [ ] Query examples are provided.
- [ ] Relevance labeling rubric is included.
- [ ] Hard-negative definition is included.
- [ ] At least one held-out evaluation set exists or is clearly planned.
- [ ] Metrics include P@5, Recall@10, MRR, NDCG@10, and hard-negative violations.
- [ ] Confidence intervals are included or explicitly marked TODO.
- [ ] Error taxonomy is included.

## Baselines

- [ ] Keyword baseline exists.
- [ ] BM25 baseline exists.
- [ ] Field-weighted BM25 baseline exists.
- [ ] Dense-only baseline exists.
- [ ] BM25 + dense fusion baseline exists.
- [ ] Ontology expansion ablation exists.
- [ ] Graph-feature ablation exists.
- [ ] Full system is compared against all prior systems.

## Graph and metadata

- [ ] Node types are documented.
- [ ] Edge types are documented.
- [ ] Edge provenance is documented.
- [ ] Edge confidence scoring is documented.
- [ ] Graph construction pipeline is reproducible.
- [ ] Graph errors and uncertainty are discussed.

## Analysis affordances

- [ ] Each affordance has required fields.
- [ ] Each affordance has optional fields.
- [ ] Each affordance has confidence logic.
- [ ] Missing metadata produces explanations.
- [ ] Affordance false positives are tracked.
- [ ] Affordance validation experiment is included or marked TODO.

## Related work

- [ ] DANDI discussed.
- [ ] OpenNeuro discussed.
- [ ] NWB discussed.
- [ ] BIDS discussed.
- [ ] EBRAINS KG / openMINDS discussed.
- [ ] PROV-O discussed.
- [ ] RO-Crate discussed.
- [ ] DataCite discussed.
- [ ] Schema.org Dataset / Bioschemas discussed.
- [ ] LinkML discussed.
- [ ] BM25 and hybrid retrieval discussed.
- [ ] RRF discussed.
- [ ] ColBERT / late interaction discussed.
- [ ] SPLADE / learned sparse retrieval discussed.
- [ ] Cross-encoder reranking discussed.
- [ ] PathSim and metapath2vec discussed.
- [ ] GraphRAG discussed.

## Experiments

- [ ] Baseline ladder implemented or clearly planned.
- [ ] Hard-negative benchmark implemented or clearly planned.
- [ ] Affordance validation implemented or clearly planned.
- [ ] Cross-dataset pairing benchmark implemented or clearly planned.
- [ ] Metadata robustness experiment implemented or clearly planned.
- [ ] Embedding model bakeoff implemented or clearly planned.
- [ ] Graph link-prediction experiment implemented or clearly planned.
- [ ] Latent neural signature prototype implemented or clearly marked future work.
- [ ] Causal claim graph prototype implemented or clearly marked future work.

## Reproducibility

- [ ] Repository structure is described.
- [ ] Benchmark query files are included.
- [ ] Label files are included or marked TODO.
- [ ] Config files are versioned.
- [ ] Random seeds are controlled where relevant.
- [ ] Embedding/model versions are documented.
- [ ] Report generation commands are documented.
- [ ] LaTeX table generation is automated or documented.

## LaTeX quality

- [ ] Paper compiles cleanly.
- [ ] Overfull boxes are fixed or visually harmless.
- [ ] Long tables use `tabularx` or reduced font.
- [ ] Figures fit the template.
- [ ] Floats use `[t]` or `[htbp]` rather than brittle `[h]`.
- [ ] Long code identifiers do not run off the page.
- [ ] Citations are complete and verified.

## Submission readiness

- [ ] Abstract is honest and specific.
- [ ] Introduction clearly states the gap.
- [ ] Method section is reproducible.
- [ ] Evaluation section is not hand-wavy.
- [ ] Limitations section is candid.
- [ ] Future work is ambitious but separated from results.
- [ ] The paper answers: why is this more than metadata search?
- [ ] The paper answers: why is this more than a knowledge graph?
- [ ] The paper answers: why does this matter to neuroscience?
