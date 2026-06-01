# Paper Expansion Instructions

## Recommended paper structure

Use `neural_search_iclr_whitepaper.tex` as the main manuscript. Treat `neural_search_whitepaper.tex` as a technical and mathematical appendix source.

The current paper is strongest when framed as a systems and retrieval paper, not only a vision document. The central upgrade is to build an evidence spine: claims, baselines, ablations, benchmark protocol, failure modes, and explicit separation between implementation and future work.

## Main positioning

Recommended thesis:

> Neural Search is not only a search engine for neuroscience datasets. It is a retrieval framework for reusable experimental contexts, where datasets are matched by typed scientific meaning, graph relationships, provenance, and analysis affordances.

This is more defensible than broad claims of general superiority over existing dataset portals.

## Replace unsupported claims

Avoid phrases such as:

- state-of-the-art
- first system to solve neuroscience search
- fully automatic scientific reasoning
- optimal search across neuroscience
- validated cross-dataset intelligence

Use more defensible phrasing:

- strong performance on an initial expert-curated benchmark
- an implemented prototype of experiment-aware retrieval
- a framework for combining metadata, graph structure, and analysis affordances
- a roadmap toward latent neural and causal retrieval
- an initial validation requiring larger benchmark expansion

## Add a Claim Status and Evidence table

Add this near the end of the introduction.

| Claim | Status | Evidence |
|---|---|---|
| Hybrid metadata retrieval improves initial search quality | Implemented | Initial 30-query benchmark |
| Hard-negative filtering reduces invalid matches | Implemented | Benchmark violation analysis needed |
| Typed scientific labels improve interpretability | Implemented | Manual inspection and ablation needed |
| Graph metapaths improve cross-dataset relatedness | Partially implemented | Pairing benchmark needed |
| Analysis affordance search is a distinctive retrieval layer | Implemented or partially implemented | Affordance validation needed |
| Latent neural signature search | Proposed | Future NWB-derived prototype |
| Cross-species experimental context alignment | Proposed | Future ontology and benchmark work |
| Causal claim graph | Proposed | Future provenance and intervention schema |

## Strengthen the benchmark section

Create a full reproducibility subsection.

Suggested section title:

```latex
\subsection{Benchmark Protocol}
```

Include:

1. Corpus size
   - number of datasets
   - number of papers
   - number of metadata records
   - number of graph nodes and edges
   - number of indexed fields

2. Query set
   - number of queries
   - query categories
   - examples per category
   - held-out queries versus development queries

3. Labeling protocol
   - relevance scale: 0, 1, 2, 3
   - who labeled the results
   - annotation instructions
   - adjudication protocol
   - inter-annotator agreement, if available

4. Metrics
   - Precision@5
   - Recall@10
   - MRR
   - NDCG@10
   - hard-negative violation rate
   - latency
   - explanation completeness

5. Baselines
   - keyword search
   - BM25
   - field-weighted BM25
   - dense-only retrieval
   - BM25 + dense RRF
   - BM25 + dense + ontology
   - BM25 + dense + ontology + graph
   - full Neural Search system

6. Error taxonomy
   - species mismatch
   - modality mismatch
   - task mismatch
   - region mismatch
   - missing metadata
   - false synonym expansion
   - graph edge overgeneralization
   - abstract-level semantic match but dataset-level mismatch

## Add a baseline ladder section

Suggested section:

```latex
\subsection{Retrieval Baselines and Ablations}
```

Evaluation ladder:

1. Keyword search
2. BM25
3. Field-weighted BM25
4. Dense-only retrieval
5. BM25 + dense reciprocal rank fusion
6. BM25 + dense + ontology expansion
7. BM25 + dense + ontology + graph features
8. Full system with affordance scoring and hard-negative constraints

The point is not just to show that the full system wins. The point is to show which layer contributes what.

## Expand the related work section

Organize related work into these subsections:

1. Neuroscience data repositories and standards
   - DANDI
   - OpenNeuro
   - NWB
   - BIDS
   - EBRAINS Knowledge Graph / openMINDS

2. Scientific metadata and provenance standards
   - PROV-O
   - RO-Crate
   - DataCite
   - Schema.org Dataset
   - Bioschemas
   - LinkML

3. Hybrid information retrieval
   - BM25
   - dense retrieval
   - reciprocal rank fusion
   - late interaction retrieval such as ColBERT
   - learned sparse retrieval such as SPLADE
   - cross-encoder reranking

4. Graph retrieval and heterogeneous information networks
   - PathSim
   - metapath2vec
   - TransE
   - heterogeneous graph attention networks
   - GraphRAG

5. Scientific search and dataset recommendation
   - dataset discovery
   - research object retrieval
   - semantic search over experimental metadata
   - model/data matching

## Add retrieval as three-layer scientific matching

Suggested section:

```latex
\section{Retrieval as Three-Layer Scientific Matching}
```

### Layer 1: Surface matching

Text, title, description, abstract, keywords, authors, and repository metadata.

Methods:

- BM25
- field-weighted BM25
- dense embedding retrieval
- query expansion

### Layer 2: Experimental-context matching

Typed slots:

- species
- modality
- brain region
- behavioral task
- behavioral event
- preparation
- device
- data format
- sampling regime
- perturbation
- disease or condition
- analysis affordance

This is the key originality layer.

### Layer 3: Relational matching

Graph relationships:

- dataset to paper
- dataset to task
- dataset to brain region
- dataset to modality
- task to cognitive construct
- modality to analysis method
- analysis method to required metadata
- dataset to model benchmark

This is where dataset search becomes scientific context search.

## Add a compact graph schema in the main text

Suggested table:

| Node type | Examples |
|---|---|
| Dataset | DANDI dandiset, OpenNeuro dataset |
| Paper | DOI, PubMed, Semantic Scholar record |
| Species | mouse, rat, macaque, human |
| Brain region | V1, PFC, hippocampus, striatum |
| Modality | Neuropixels, calcium imaging, EEG, fMRI |
| Task | Go/NoGo, probabilistic reversal learning, delay discounting |
| Event | cue, lick, reward, choice, stimulus onset |
| Analysis affordance | choice decoding, Q-learning, state-space modeling |
| Model | GLM, RNN, SSM, encoding model |
| Data standard | NWB, BIDS, Zarr |
| Software | Suite2p, DeepLabCut, SpikeInterface, MNE |

Suggested edge types:

| Edge type | Meaning |
|---|---|
| HAS_MODALITY | dataset contains modality |
| RECORDED_FROM | dataset targets brain region |
| USES_TASK | dataset uses behavioral task |
| HAS_EVENT | dataset contains event labels |
| SUPPORTS_ANALYSIS | dataset satisfies an analysis affordance |
| DERIVED_FROM_PAPER | metadata came from publication evidence |
| VALIDATED_BY_SCHEMA | metadata was verified from NWB/BIDS schema |
| SIMILAR_TASK_CONTEXT | task-level similarity |
| COMPARABLE_ACROSS_SPECIES | cross-species comparison candidate |
| BENCHMARKS_MODEL | dataset can evaluate a model class |
| REQUIRES_METADATA | analysis requires specific fields |

## Add standards-aligned provenance

Suggested section:

```latex
\subsection{Standards-Aligned Provenance}
```

Map internal concepts to standards:

| Neural Search concept | External standard |
|---|---|
| Dataset entity | Schema.org Dataset / DataCite |
| Provenance source | PROV-O Entity |
| Extraction process | PROV-O Activity |
| Curator, model, or pipeline | PROV-O Agent |
| Dataset package | RO-Crate |
| Schema validation | LinkML |
| Life-science web markup | Bioschemas |

## Add concrete analysis affordance predicates

The paper should define affordances as testable predicates, not vague tags.

Example:

> A dataset supports Q-learning model fitting if it contains trial order, subject/session identity, choices, reward outcomes, and task state or stimulus identity. Reaction time or action latency increases confidence but is not strictly required.

Suggested table:

| Affordance | Required fields | Optional fields |
|---|---|---|
| Choice decoding | neural activity, trial labels, choice labels | reaction time, confidence, stimulus type |
| Q-learning | choices, rewards, trial order, subject/session IDs | reaction time, block structure, task state |
| State-space modeling | continuous time series, timestamps, sampling rate | trial boundaries, behavior labels |
| Cross-session stitching | subject IDs, session dates, unit/channel metadata | drift metrics, anatomical coordinates |
| Causal perturbation analysis | intervention labels, timing, control condition | dose, stimulation parameters, randomization |
| Representational similarity | aligned stimuli, population responses | model features, train/test split |
| Neural-behavior alignment | neural time series, behavior time series, shared timestamps | event labels, pose coordinates |

## Add a future experiments section

Include these experiments:

1. Baseline ladder
2. Hard-negative adversarial benchmark
3. Affordance validation
4. Cross-dataset pairing benchmark
5. Metadata robustness perturbation
6. Embedding model bakeoff
7. Graph link-prediction benchmark
8. Latent neural signature search prototype
9. Causal claim graph prototype
10. Human-in-the-loop dataset recommendation study

## Add a limitations section

A reputable paper needs a sharp limitations section.

Suggested limitations:

- Metadata extraction quality limits search quality.
- Dataset-level descriptions can obscure file-level or session-level variation.
- Graph edges may encode curator bias or extraction error.
- Affordance predicates are incomplete and require domain-specific validation.
- Benchmarks are initially small and may reflect the authors' scientific priors.
- Dense embeddings may overgeneralize across superficially similar neuroscience terms.
- Hard-negative constraints reduce false positives but may reduce recall.
- Latent neural search requires real data access and standardized feature extraction.

## Add a reproducibility section

Include:

- repository structure
- data sources
- benchmark query files
- labels
- config files
- random seeds
- model versions
- embedding provider versions
- generated reports
- how to reproduce tables
- how to regenerate figures

## Key tone shift

Move from:

> This system solves neuroscience search.

To:

> This system proposes and implements an experiment-aware retrieval architecture, then evaluates how typed metadata, provenance, graph structure, and analysis affordances contribute to scientific dataset discovery.
