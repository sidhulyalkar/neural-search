# Related Work and Standards Sources to Verify

Use this as a source checklist. Verify exact citation metadata before adding to the final `.bib` file.

## Neuroscience repositories and standards

### DANDI

Why it matters:

- DANDI is a major neuroscience data archive focused on NWB-style neurophysiology datasets.
- Position Neural Search as complementary: DANDI improves sharing and metadata availability; Neural Search adds typed experimental-context retrieval and analysis affordance matching.

Suggested discussion:

> DANDI provides a FAIR archive for neurophysiology datasets and supports standardization through NWB. Neural Search can ingest DANDI metadata and add experiment-aware retrieval over species, modality, task, brain region, events, analysis affordances, and graph relationships.

Source to verify:

- https://about.dandiarchive.org/

### OpenNeuro

Why it matters:

- OpenNeuro is a major platform for sharing BIDS-compliant neuroimaging and related datasets.
- Useful for positioning across human neuroimaging and non-invasive data.

Suggested discussion:

> OpenNeuro emphasizes standardized sharing of BIDS-compatible datasets. Neural Search can build on such metadata to support cross-dataset semantic retrieval and analysis compatibility ranking.

Sources to verify:

- https://openneuro.org/
- OpenNeuro eLife paper

### NWB

Why it matters:

- NWB is central to standardized neurophysiology data representation.
- Neural Search should eventually parse NWB files directly to validate metadata and extract latent neural signatures.

Suggested discussion:

> NWB provides structured containers for neurophysiology data and experimental metadata. Neural Search treats NWB schema-level evidence as higher-confidence provenance than free-text extraction.

Sources to verify:

- https://www.nwb.org/
- NWB eLife paper

### BIDS

Why it matters:

- BIDS is the dominant standard for neuroimaging and increasingly supports derivatives and additional modalities.
- Neural Search can use BIDS fields and events files for task/event/behavior extraction.

Sources to verify:

- https://bids.neuroimaging.io/
- Gorgolewski et al., Scientific Data, 2016

### EBRAINS Knowledge Graph and openMINDS

Why it matters:

- EBRAINS already uses a knowledge graph and neuroscience metadata standards.
- Neural Search should be positioned as experiment-aware retrieval and analysis affordance search, not as merely “a knowledge graph.”

Sources to verify:

- https://docs.kg.ebrains.eu/
- https://openminds.ebrains.eu/

## Metadata, provenance, and research object standards

### PROV-O

Why it matters:

- Use for provenance modeling: entities, activities, agents.
- Neural Search extraction events and validation steps can map onto PROV-O.

Source to verify:

- https://www.w3.org/TR/prov-o/

### RO-Crate

Why it matters:

- Useful for packaging datasets, metadata, workflows, and outputs as research objects.
- Could serve as export format for Neural Search dataset cards or evidence bundles.

Source to verify:

- https://www.researchobject.org/ro-crate/

### DataCite

Why it matters:

- DOI-backed dataset metadata.
- Helps link datasets to papers, creators, funders, and related identifiers.

Source to verify:

- https://datacite.org/

### Schema.org Dataset and Bioschemas

Why it matters:

- Useful for web-indexable dataset metadata.
- Bioschemas adds life-science-specific profile conventions.

Sources to verify:

- https://schema.org/Dataset
- https://bioschemas.org/

### LinkML

Why it matters:

- Good fit for portable schema definitions, enumerations, validation, and generated data models.
- Could be used to formalize Neural Search metadata and affordance schema.

Source to verify:

- https://linkml.io/

## Hybrid information retrieval

### BM25

Why it matters:

- Standard sparse lexical retrieval baseline.
- Necessary baseline for any search paper.

Citation to verify:

- Robertson and Zaragoza, The Probabilistic Relevance Framework: BM25 and Beyond, Foundations and Trends in IR.

### Reciprocal Rank Fusion

Why it matters:

- Strong, simple rank-fusion baseline.
- Useful for BM25 + dense retrieval.

Citation to verify:

- Cormack, Clarke, and Buettcher, Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods, SIGIR 2009.

### ColBERT / late interaction retrieval

Why it matters:

- Token-level interaction can preserve fine-grained constraints better than single-vector dense embeddings.
- Useful future experiment for scientific queries with multiple constraints.

Citation to verify:

- Khattab and Zaharia, ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT, SIGIR 2020.

### SPLADE / learned sparse retrieval

Why it matters:

- Bridges sparse lexical matching and semantic expansion.
- Useful for neuroscience synonyms and abbreviations.

Citation to verify:

- Formal et al., SPLADE papers.

### Cross-encoder reranking

Why it matters:

- Candidate generation can be broad, while cross-encoder reranking can evaluate query/document compatibility more precisely.

Suggested use:

- Rerank top 50 candidates with query + dataset card + graph explanation.

## Graph retrieval and heterogeneous information networks

### PathSim

Why it matters:

- Classic metapath-based similarity in heterogeneous information networks.
- Directly relevant to typed paths such as dataset -> task -> dataset.

Citation to verify:

- Sun et al., PathSim: Meta Path-Based Top-K Similarity Search in Heterogeneous Information Networks, VLDB 2011.

### metapath2vec

Why it matters:

- Heterogeneous graph embeddings based on metapath-guided random walks.

Citation to verify:

- Dong, Chawla, Swami, metapath2vec: Scalable Representation Learning for Heterogeneous Networks, KDD 2017.

### TransE

Why it matters:

- Classic multi-relational graph embedding baseline.
- Useful for link prediction experiments.

Citation to verify:

- Bordes et al., Translating Embeddings for Modeling Multi-relational Data, NeurIPS 2013.

### Heterogeneous Graph Attention Networks

Why it matters:

- Relevant if Neural Search adds learned graph representations across typed nodes and typed edges.

Citation to verify:

- Wang et al., Heterogeneous Graph Attention Network, WWW 2019.

### GraphRAG

Why it matters:

- Useful for corpus-level graph summarization and question answering.
- Should be framed as an analysis layer, not necessarily the core ranker.

Citation to verify:

- Microsoft GraphRAG paper / arXiv.

## Scientific and biomedical embeddings

### SciBERT

Why it matters:

- Scientific-language encoder trained on scientific text.

Citation to verify:

- Beltagy, Lo, Cohan, SciBERT: A Pretrained Language Model for Scientific Text, EMNLP-IJCNLP 2019.

### BioBERT

Why it matters:

- Biomedical-language encoder trained/adapted on biomedical corpora.

Citation to verify:

- Lee et al., BioBERT: a pre-trained biomedical language representation model for biomedical text mining, Bioinformatics 2020.

### SPECTER / SPECTER2

Why it matters:

- Scientific document embeddings useful for paper/dataset linking.

Citation to verify:

- Cohan et al., SPECTER papers.

## Warning

Do not paste this file directly into the bibliography. Verify exact titles, venues, years, URLs, and BibTeX entries first.
