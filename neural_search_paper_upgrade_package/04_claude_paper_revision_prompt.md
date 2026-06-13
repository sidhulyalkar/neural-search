# Claude Prompt: Paper Revision

Paste this into Claude.

```text
Read both LaTeX drafts.

Use `neural_search_whitepaper.tex` as the main manuscript and treat `neural_search_whitepaper.tex` as a mathematical/technical appendix source.

Revise the main paper to make it more scientifically reputable and submission-like.

Required changes:

1. Remove or soften unsupported claims such as “state-of-the-art” unless directly backed by baseline comparisons.

2. Add a “Claim Status and Evidence” table separating implemented, measured, partially implemented, and proposed components.

3. Expand the benchmark section into a full reproducibility protocol:
   - corpus size
   - query categories
   - labeling rubric
   - annotator protocol
   - relevance scale
   - hard-negative definitions
   - bootstrap confidence interval method
   - baseline ladder
   - ablation matrix
   - error taxonomy

4. Add a stronger related-work section covering:
   - DANDI
   - OpenNeuro
   - NWB
   - BIDS
   - EBRAINS Knowledge Graph / openMINDS
   - Schema.org Dataset
   - DataCite
   - PROV-O
   - RO-Crate
   - LinkML
   - BM25
   - dense retrieval
   - reciprocal rank fusion
   - ColBERT / late interaction retrieval
   - SPLADE / learned sparse retrieval
   - cross-encoder reranking
   - GraphRAG
   - PathSim
   - metapath2vec
   - TransE
   - heterogeneous graph attention networks

5. Add a concise main-text graph schema table:
   - node types
   - edge types
   - evidence fields
   - confidence model

6. Add a concrete analysis-affordance table:
   - choice decoding
   - Q-learning
   - state-space modeling
   - neural-behavior alignment
   - cross-session analysis
   - causal perturbation analysis
   - representational similarity analysis

7. Add a “Retrieval as Three-Layer Scientific Matching” section with:
   - surface matching
   - experimental-context matching
   - relational graph matching

8. Add a “Standards-Aligned Provenance” section mapping Neural Search concepts to:
   - PROV-O
   - RO-Crate
   - DataCite
   - Schema.org Dataset
   - Bioschemas
   - LinkML

9. Add a “Future Experiments” section with:
   - baseline ladder
   - hard-negative adversarial benchmark
   - affordance validation
   - cross-dataset pairing benchmark
   - metadata robustness perturbation
   - embedding model bakeoff
   - graph link-prediction benchmark
   - latent neural signature search prototype
   - causal claim graph prototype
   - human-in-the-loop dataset recommendation study

10. Add a limitations section with honest weaknesses:
   - small benchmark
   - metadata extraction errors
   - graph edge uncertainty
   - incomplete affordance predicates
   - dense embedding overgeneralization
   - possible curator bias
   - latent neural search not yet validated unless implemented

11. Fix Overleaf warnings:
   - replace `[h]` with `[t]` or `[htbp]`
   - break long math lines
   - convert long appendix tables to `tabularx`
   - avoid long unbreakable `\texttt{}` field lists
   - avoid very long inline equations

12. Ensure the final paper clearly distinguishes:
   - current implementation
   - validated result
   - proposed roadmap
   - speculative long-term vision

13. Do not invent new benchmark numbers. Use placeholders like `TODO`, `to be evaluated`, or `not yet measured` where experiments have not yet been run.

14. Preserve the main thesis:

Neural Search is not only a search engine for neuroscience datasets. It is a retrieval framework for reusable experimental contexts, where datasets are matched by typed scientific meaning, graph relationships, provenance, and analysis affordances.

Deliverables:

- Revised `neural_search_whitepaper.tex`
- Optional appendix sections extracted from `neural_search_whitepaper.tex`
- A changelog summarizing major revisions
- A list of TODOs where real experiments or citations are still needed
```
