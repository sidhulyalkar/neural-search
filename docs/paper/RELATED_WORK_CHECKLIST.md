# Related Work Coverage Checklist

This document tracks coverage of related work in the Neural Search manuscript.

## Neuroscience Data Infrastructure

| Topic | Covered | Citation | Notes |
|-------|---------|----------|-------|
| DANDI Archive | Yes | rubinen2024dandi | Primary data source |
| OpenNeuro | Yes | markiewicz2021openneuro | Primary data source |
| NWB (Neurodata Without Borders) | Yes | teeters2015neurodata | Data standard |
| BIDS (Brain Imaging Data Structure) | Yes | gorgolewski2016brain | Data standard |
| Allen Brain Observatory | Yes | allen2019brain | Data source |
| NeMO Archive | Partial | nemo2023archive | Transcriptomics focus |
| EBRAINS Knowledge Graph | Yes | Added in revision | European infrastructure |
| openMINDS | Yes | Added in revision | Metadata framework |

## Metadata Standards

| Topic | Covered | Citation | Notes |
|-------|---------|----------|-------|
| Schema.org Dataset | Yes | schemaorg2022dataset | General metadata |
| DataCite | Yes | datacite2021metadata | DOI metadata |
| PROV-O | Yes | Added in revision | Provenance standard |
| RO-Crate | Yes | Added in revision | Research objects |
| LinkML | Yes | Added in revision | Schema definition |
| Bioschemas | Yes | Added in revision | Life sciences extension |

## Information Retrieval

| Topic | Covered | Citation | Notes |
|-------|---------|----------|-------|
| BM25 | Yes | robertson2009probabilistic | Lexical retrieval |
| Field-weighted BM25 | Yes | robertson2004simple | Structured retrieval |
| Dense retrieval (DPR) | Yes | karpukhin2020dense | Neural retrieval |
| ColBERT | Yes | khattab2020colbert | Late interaction |
| SPLADE | Yes | formal2021splade | Learned sparse |
| Reciprocal Rank Fusion | Yes | cormack2009reciprocal | Fusion method |
| Hybrid retrieval | Yes | chen2022hybrid | Combined approaches |
| Learning to rank | Yes | liu2009learning | Supervised ranking |

## Knowledge Graphs

| Topic | Covered | Citation | Notes |
|-------|---------|----------|-------|
| PathSim | Yes | sun2011pathsim | Metapath similarity |
| metapath2vec | Yes | dong2017metapath2vec | Graph embeddings |
| HAN | Yes | wang2019heterogeneous | Attention networks |
| TransE | Yes | bordes2013translating | KG embeddings |
| GraphRAG | No | TODO | Graph-augmented generation |
| Biolink Model | Yes | biolink2020model | Biomedical KG |
| Gene Ontology | Yes | gene2021gene | Reference ontology |

## Neuroscience-Specific

| Topic | Covered | Citation | Notes |
|-------|---------|----------|-------|
| IBL standardization | Yes | ibl2021standardized | Data standardization |
| Data reuse challenges | Yes | ferguson2014big, soska2021data | Motivation |
| Metadata inconsistency | Yes | gleeson2019systematic | Problem statement |
| BMTK | Yes | bmtk2020toolkit | Modeling toolkit |
| NEST simulator | Yes | gewaltig2007nest | Simulation platform |

## Neuro-AI Convergence

| Topic | Covered | Citation | Notes |
|-------|---------|----------|-------|
| Neuro-AI integration | Yes | hassabis2017neuroscience | Vision |
| Deep learning and neuroscience | Yes | richards2019deep | Cross-pollination |
| Neural network interpretability | Yes | saxe2021if | Theory connections |
| NeuroAI initiative | Yes | zador2023catalyzing | Community effort |

## Missing / To Add

- [ ] GraphRAG paper citation if relevant
- [ ] More recent dense retrieval methods (2024-2025)
- [ ] Neuroscience-specific search tools (if any exist)
- [ ] Federated data search systems
- [ ] Active learning for relevance feedback

## Citation Format Notes

- Use natbib plainnat style
- Include URLs for repositories and tools
- Prefer peer-reviewed citations over preprints where available

---

*Last updated: 2026-05-26*
