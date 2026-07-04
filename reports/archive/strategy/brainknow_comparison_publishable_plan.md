# BrainKnow/LBD Comparison and Publishable Neural Search Roadmap

Date: 2026-06-20

This note compares BrainKnow / Linked Brain Data (LBD) with Neural Search and turns the comparison into a concrete development plan. It is grounded in:

- BrainKnow paper: https://arxiv.org/pdf/2403.04346
- LBD links supplied by Sid:
  - http://www.linked-brain-data.org/connectome.jsp
  - http://www.linked-brain-data.org/about.jsp?link=link6
  - http://www.linked-brain-data.org/querybox.jsp?link=link1
  - http://www.linked-brain-data.org/sparqlsearch.jsp?link=link0
  - http://www.linked-brain-data.org/treeVisual.jsp?link=link2
- Local Neural Search artifacts:
  - `docs/whitepaper/neural_search_whitepaper.tex`
  - `docs/project_vision.md`
  - `docs/field_state_memory_graph.md`
  - `docs/evaluation_protocol.md`
  - `neural_search/graph/schema.py`
  - `neural_search/literature/relationship_builder.py`
  - `neural_search/embeddings/semantic_fingerprint.py`

Note: the old linked-brain-data.org pages were not fetchable through the browser tool during this review. The LBD interpretation below uses the text Sid supplied plus the BrainKnow paper, which appears to be the successor or contemporary public write-up of the same project lineage.

## 1. What BrainKnow/LBD Is

BrainKnow is a literature-derived neuroscience knowledge engine. Its core move is simple and powerful:

1. Build a controlled list of neuroscience concepts.
2. Scan PubMed titles and abstracts.
3. Add a relationship when two controlled concepts occur in the same sentence.
4. Store supporting paper metadata, evidence sentence, extraction date, publication date, and inferred species.
5. Collapse the concept graph into Node2Vec embeddings for semantic relatedness, indirect discovery, and visualization.

The reported scale is the headline:

- 37,011 neuroscience concepts.
- 3,626,931 concept relationships.
- 41,547,471 triples.
- Knowledge extracted from 1,817,744 PubMed articles by February 2024.
- Daily PubMed update-file synchronization.
- Node2Vec over the full undirected weighted graph; 128-dimensional vectors, walk length 80, 18 walks, 10 epochs, window size 16, p=q=0.25.
- AUROC 0.93 for predicting whether embedding similarity corresponds to known concept relationships.

LBD, from the supplied description, is the broader linked-data platform around the CAS Brain Knowledge Base. It emphasizes extraction, linking, representation, integration, visualization, semantic search, reasoning, and support for RDF/SPARQL-style knowledge access across sources such as PubMed, INCF-CUMBO, Allen Reference Atlas, NIF, NeuroLex, MeSH, DBPedia/Wikipedia, and related brain resources.

## 2. BrainKnow's Strengths

BrainKnow is strongest where Neural Search is currently weakest:

- Scale: millions of edges and more than a million papers.
- Concept coverage: diseases, cognitive functions, drugs, regions, neurons, genes/proteins, pathways, neurotransmitters.
- Literature sweep: it builds a field-scale map from PubMed rather than a dataset-centric registry.
- Temporal updating: new PubMed update files are scanned as they arrive.
- Simple evidence surface: for a concept pair, it can show the supporting sentences and papers over time.
- Network synthesis: Node2Vec exposes indirect relatedness, multi-concept relatedness, and possible future links.
- Research navigation: concept category browsing, search boxes, relation graphs, and visual summaries are core product surfaces.

## 3. BrainKnow's Weaknesses and Openings

The paper is candid about the main limitation: BrainKnow does not differentiate relationship types. It turns the graph into an undirected weighted co-occurrence network. That gives scale, but it loses mechanistic semantics.

Important limitations:

- Same-sentence co-occurrence is not relation extraction. "A inhibits B", "A does not affect B", and "A was compared with B" all become generic relationships unless downstream text is inspected.
- Edges are mostly unsigned and untyped. Direction, polarity, negation, effect size, causal interpretation, experimental manipulation, species, condition, and measurement modality are not core graph semantics.
- Embeddings represent co-occurrence topology, not typed scientific claims.
- It is concept-centered, not dataset-centered. It does not answer: "Which dataset can I actually download and analyze to test this?"
- It does not appear to model analysis affordances, raw-vs-processed availability, notebook readiness, dataset licensing, source archive quality, or hard negative constraints.
- Its own future-work section says relation typing and LLM-assisted extraction are needed.

This is the opening for Neural Search: not to out-scale BrainKnow immediately, but to make a more useful scientific object graph where typed relations and reusable datasets matter.

## 4. What Neural Search Already Does Differently

Neural Search is a research-object retrieval system rather than a concept-only literature KG.

Implemented or partially implemented strengths:

- Dataset nodes are first-class: DANDI, OpenNeuro, NeuroVault, Zenodo, OSF, Allen, IBL, NeuroMorpho, ModelDB, MICrONS, CellxGene, and others appear across artifacts.
- Paper nodes and paper-dataset linking exist.
- Graph schema supports datasets, papers, tasks, modalities, recording scales, brain regions, species, behavioral events, analysis affordances, required signals, methods, findings, institutions, authors, and retrieval/evaluation artifacts.
- Evidence is explicit: graph nodes and edges carry source type, source id, source field, evidence text, extractor name/version, and confidence.
- Retrieval is experiment-aware: structured constraints, hard negatives, ontology matching, usefulness scoring, graph context, and explanation surfaces.
- Analysis affordances are first-class: the system tries to answer "what analysis can this dataset support?"
- The literature layer already has finding extraction and cross-finding edges with support/contradiction based on result direction, shared region/task, and cross-paper evidence.
- Semantic fingerprints already separate dimensions: text, task, modality, behavior, analysis, region, and experimental design.

The strongest position for Neural Search is:

> BrainKnow maps what neuroscience papers talk about. Neural Search should map what neuroscience datasets can be used to test, reproduce, compare, or extend.

## 5. The Publishable Differentiator

The next paper-worthy claim should not be "we built another brain KG." BrainKnow already owns that at scale.

The differentiator should be:

> A provenance-backed, relation-typed, dataset-linked neuroscience knowledge graph that retrieves reusable experimental contexts and exposes mechanistic dimensions such as spatial localization, frequency bands, periodicity, directionality, negation, counteractivation, and analysis affordance.

In practical terms, this means Neural Search should build a "claim-to-dataset-to-analysis" graph:

```text
paper -> reports_finding -> finding
finding -> involves_region -> brain_region
finding -> involves_signal -> neural_signal
finding -> has_direction -> increase/decrease/no_change/mixed
finding -> has_polarity -> positive/negative/null
finding -> has_frequency_band -> theta/gamma/beta/...
finding -> has_periodicity -> oscillatory/transient/tonic/event_locked
finding -> has_spatial_frame -> atlas/ROI/electrode/site/cell_type
finding -> measured_in -> dataset
dataset -> supports_analysis -> affordance
dataset -> has_raw_signal -> signal
dataset -> contraindicated_for -> query/analysis
```

Then retrieval can answer questions BrainKnow cannot:

- "Find datasets where hippocampal theta phase relates to spatial navigation, excluding fMRI."
- "Show papers reporting OFC counteractivation during reward reversal and datasets that could test it."
- "Find reusable ephys datasets with negative/no-change evidence for mPFC choice encoding."
- "Find comparable calcium and Neuropixels datasets spanning VTA dopamine reward prediction signals."
- "Which datasets can reproduce or falsify this claim?"

## 6. Proposed Graph Semantics

Add a richer middle layer between raw literature findings and dataset records.

### New node types

- `neural_signal`: spike_rate, lfp_power, calcium_trace, bold_signal, eeg_power, meg_source, synapse_count.
- `frequency_band`: delta, theta, alpha, beta, gamma, high_gamma, ripple.
- `temporal_pattern`: event_locked, oscillatory, transient, sustained, periodic, aperiodic, phase_locked.
- `spatial_frame`: atlas_region, electrode_site, voxel, cortical_area, cell_type, tract, connectome_edge.
- `effect_direction`: increase, decrease, no_change, mixed, correlation_positive, correlation_negative.
- `experimental_condition`: task_epoch, stimulus, reward, error, sleep_stage, disease_state, manipulation.
- `negative_relation`: absence_of_effect, failed_replication, contraindication, hard_negative.
- `basis_representation`: task_basis, region_basis, signal_basis, method_basis, evidence_basis.

### New edge types

- `finding_has_effect_direction`
- `finding_has_frequency_band`
- `finding_has_temporal_pattern`
- `finding_has_spatial_frame`
- `finding_has_condition`
- `finding_counteractivates_region`
- `finding_negates_relation`
- `finding_measured_in_dataset`
- `dataset_contains_signal`
- `dataset_has_sampling_rate`
- `dataset_has_task_epoch`
- `dataset_can_test_finding`
- `dataset_cannot_test_finding`
- `dataset_reproduces_finding`
- `dataset_contradicts_finding`
- `finding_supports_claim`
- `finding_contradicts_claim`
- `claim_has_basis_representation`

### Required edge properties

- `species`
- `modality`
- `measurement_unit`
- `time_window`
- `frequency_range_hz`
- `atlas`
- `coordinate_space`
- `statistical_relation`
- `effect_size`
- `p_value`
- `n_subjects_or_sessions`
- `source_sentence`
- `source_section`
- `extractor_version`
- `confidence`
- `review_status`

## 7. Embedding Plan: Multi-View, Not One Vector

BrainKnow's Node2Vec is useful because it compresses graph topology. Neural Search should keep this idea but make it typed and multi-view.

Recommended embedding families:

1. Text embeddings: title, abstract, dataset description, methods.
2. Field embeddings: separate vectors for task, modality, region, species, signal, analysis affordance.
3. Graph embeddings: typed metapath embeddings rather than only undirected Node2Vec.
4. Claim embeddings: a finding-level embedding that includes relation polarity and evidence context.
5. Dataset affordance embeddings: whether a dataset can support analyses, not just what it mentions.
6. Negative-space embeddings: explicit "should not match" dimensions for excluded modalities, null findings, contradiction, and contraindication.
7. Basis representations: sparse, interpretable basis vectors over region x signal x task x condition x direction x evidence strength.

Practical approach:

- Phase 1: Keep existing semantic fingerprints and add signal/frequency/temporal/spatial/effect dimensions.
- Phase 2: Add typed metapath scores, e.g. dataset-paper-finding-region, dataset-affordance-signal, finding-region-frequency-task.
- Phase 3: Train/evaluate a heterogeneous graph embedding model: start with Node2Vec baselines, then compare metapath2vec/HAN/R-GCN/GraphSAGE if enough labels exist.
- Phase 4: Use a cross-encoder or LLM judge only as a reranker or evidence extractor, never as the ungrounded source of truth.

## 8. Development Roadmap

### Milestone 0: Snapshot discipline

Goal: make every paper number reproducible.

Tasks:

- Freeze one corpus manifest, one graph manifest, one index manifest, and one benchmark query file.
- Add a single command that rebuilds the graph and writes a manifest with source counts, edge counts, extractor versions, and checksum hashes.
- Fix or quarantine placeholder reports with impossible metrics such as all-zero NDCG or synthetic perfect ablations.
- Add a "claim status" table that says implemented, validated, preliminary, or future work.

Publishability value: avoids fragile or contradictory claims.

### Milestone 1: BrainKnow-style baseline inside Neural Search

Goal: create an honest baseline that reproduces BrainKnow's central method on your corpus.

Tasks:

- Build concept co-occurrence edges from OpenAlex/PubMed-like abstracts using same-sentence co-occurrence.
- Add undirected weighted edges and Node2Vec embeddings.
- Evaluate concept relatedness and dataset retrieval using this as the "BrainKnow-style KG baseline."
- Compare against typed Neural Search graph features.

Publishability value: lets the paper say "we compare against a literature co-occurrence KG baseline inspired by BrainKnow."

### Milestone 2: Typed finding graph

Goal: turn findings into relation-rich KG units.

Tasks:

- Extend normalized finding schema with `effect_direction`, `negation`, `frequency_band`, `temporal_pattern`, `spatial_frame`, `condition`, `signal_type`, and `statistical_relation`.
- Extend `neural_search/literature/relationship_builder.py` beyond supports/contradicts/co-occurs.
- Add extractors for frequency bands and temporal language: theta, gamma, ripple, oscillation, phase-locking, periodic, transient, event-locked, sustained.
- Add extraction tests with adversarial examples for negation: "did not increase", "no significant activation", "reduced coupling", "counteractivated".
- Create 100 manually reviewed finding records as gold extraction evaluation.

Publishability value: establishes a novel relation-typed neuroscience finding KG.

### Milestone 3: Dataset-to-finding bridge

Goal: connect papers and findings to reusable datasets.

Tasks:

- Strengthen DOI/title/source exact matching.
- Add Data Availability statement parsing.
- Add dataset mention extraction for DANDI/OpenNeuro/Zenodo/OSF/NeuroVault IDs.
- Add `finding_measured_in_dataset` and `dataset_can_test_finding` edges.
- Generate dataset cards that show "claims this dataset can test" and "claims this dataset originally supported."

Publishability value: turns the KG from literature map into reusable science infrastructure.

### Milestone 4: Retrieval benchmark

Goal: prove usefulness.

Tasks:

- Build 100 benchmark queries across:
  - known-item lookup,
  - analysis affordance search,
  - spatial/region search,
  - frequency/periodicity search,
  - negative constraints,
  - contradiction/counteractivation search,
  - paper-to-dataset tracing,
  - cross-dataset comparison.
- Pool top 20-50 candidates from BM25, dense, BrainKnow-style co-occurrence, typed KG, and full system.
- Human-label at least 1,500 query-dataset pairs.
- Report NDCG@10/20, MRR, Recall@50, Precision@10, hard-negative violation rate, contradiction retrieval precision, and bootstrap confidence intervals.

Publishability value: this is the paper's evidence backbone.

### Milestone 5: Interface and accessibility

Goal: make useful information easily findable and accessible.

Tasks:

- Add "Evidence Graph" view to each dataset card:
  - linked papers,
  - extracted findings,
  - regions/signals/frequencies,
  - supported analyses,
  - contradictions/null findings,
  - source evidence.
- Add query facets for frequency, temporal pattern, effect direction, region, species, modality, and analysis affordance.
- Add a "why this dataset can test your hypothesis" explanation panel.
- Add exportable JSON-LD or RDF for graph interoperability.
- Add SPARQL-like advanced query only after the typed graph is stable; do not make it the main UX.

Publishability value: turns the method into a usable platform rather than just an offline artifact.

## 9. Recommended Paper Framing

Working title:

> Neural Search: A Provenance-Backed, Relation-Typed Knowledge Graph for Reusable Neuroscience Dataset Discovery

Core contributions:

1. A dataset-centered neuroscience KG linking datasets, papers, findings, regions, modalities, signals, tasks, species, and analysis affordances.
2. A typed relation schema for mechanistic neuroscience retrieval, including negation, contradiction, frequency, temporal pattern, and spatial grounding.
3. A multi-view retrieval model combining sparse text, field embeddings, graph metapaths, hard negatives, and affordance scoring.
4. A benchmark with human qrels over realistic neuroscience dataset-search tasks.
5. A reusable interface exposing provenance and analysis readiness.

Main comparison:

- BrainKnow: huge concept-literature map; same-sentence co-occurrence; Node2Vec; strong for concept exploration.
- Neural Search: smaller but typed, dataset-linked, provenance-backed, analysis-aware; strong for finding reusable data and testing claims.

## 10. Immediate Next Sprint

Highest leverage next sprint:

1. Add the typed finding schema extensions.
2. Build 30-50 gold reviewed finding examples.
3. Build BrainKnow-style same-sentence co-occurrence baseline from your OpenAlex tier.
4. Freeze one reproducible corpus/graph/index snapshot.
5. Create 20 benchmark queries that specifically require typed semantics:
   - 5 frequency/periodicity,
   - 5 spatial/region,
   - 5 negative/counteractivation,
   - 5 dataset-paper-finding bridge.
6. Run a small ablation:
   - BM25,
   - dense field embeddings,
   - BrainKnow-style co-occurrence graph,
   - typed Neural Search graph,
   - full system.

The target publishable mini-result is not a massive benchmark yet. It is a clean demonstration that typed relation semantics retrieve useful datasets that generic co-occurrence and text search miss.

