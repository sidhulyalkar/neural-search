# Neural Search Whitepaper Critique and Claude Implementation Instruction Set

**Document reviewed:** `neural_search_whitepaper.tex`
**Review date:** May 27, 2026
**Goal:** Turn Neural Search from a strong prototype/whitepaper into a credible provenance-aware retrieval system for scientific datasets that combines structured constraints, ontology-aware metadata, knowledge graph linkages, and learned embeddings to identify datasets that are experimentally reusable, not merely topically similar.

---

## Executive Summary

The whitepaper has a genuinely strong core: **experiment-aware scientific dataset retrieval**. The most differentiated contribution is not generic “neural search,” because that phrase already overlaps with existing vector-search and OpenSearch-style systems. The sharpest contribution is:

> **Finding scientific datasets by experimental reusability: structured constraints, analysis affordances, provenance, ontology alignment, dataset/paper linkages, and eventually content-derived neural signatures.**

The current paper already contains several valuable ideas:

- Slot-based experimental context representations.
- A provenance-weighted knowledge graph.
- Multi-signal retrieval with BM25, dense embeddings, ontology matching, graph features, and affordance scoring.
- Analysis affordances as predicates over dataset features.
- A first benchmark and ablation ladder.
- A useful limitations section that acknowledges the current evaluation gaps.

However, the current version sometimes sounds more mature than the evidence supports. It should be revised from an “ICLR-level complete system” posture into a more credible **prototype + formal framework + reproducible benchmark + implementation roadmap** posture.

The biggest upgrades needed are:

1. **Tighten claims.** Clearly mark what is implemented, validated, partially implemented, or proposed.
2. **Anchor against existing neuroscience infrastructure.** Treat DANDI, OpenNeuro, EBRAINS KG/openMINDS, NWB, BIDS, DataCite, RO-Crate, and FAIR as serious predecessors and interoperability targets.
3. **Build real embedding-based retrieval.** Do not rely only on diagrams or a generic dense-retrieval placeholder. Add named embedding fields, model comparisons, evaluation harnesses, and hybrid fusion.
4. **Make dataset linkage first-class.** Dataset-to-dataset, dataset-to-paper, dataset-to-method, dataset-to-affordance, and dataset-to-analysis edges should be evidence-backed, versioned, and auditable.
5. **Validate experimental reusability.** Affordance predictions need to be tested against actual data contents and required fields, not just metadata vibes.
6. **Expand the benchmark.** Add hard negatives, multi-annotator labels, pairwise dataset linkage evaluation, content-signature retrieval, and per-query failure analysis.

The paper’s future thesis should be:

> **Neural Search is a provenance-aware retrieval system for scientific datasets that combines structured constraints, ontology-aware metadata, knowledge graph linkages, and learned embeddings to identify datasets that are not merely topically similar, but experimentally reusable.**

That sentence should become the project’s North Star.

---

## Part I: Whitepaper Critique and Analysis

### 1. Core Framing

#### What works

The opening problem is strong: neuroscience datasets are growing across repositories, but researchers still struggle to find datasets that satisfy specific experimental constraints and analysis needs.

The abstract’s phrase “not merely textually similar, but scientifically reusable” is excellent. Keep that. It separates this project from generic semantic search.

The paper’s best conceptual move is to treat datasets as **structured scientific objects**, not documents. That should stay at the center.

#### What needs improvement

The title and abstract currently lean too heavily on “novel retrieval system” and “ICLR-level whitepaper.” The paper should avoid presenting itself as fully validated until the evaluation matures.

Recommended framing:

> We introduce a prototype framework for experiment-aware neuroscience dataset retrieval. The framework combines structured experimental metadata, ontology-aware constraints, provenance-weighted knowledge graph linkages, analysis affordance validation, and learned embeddings. Initial results on a small expert-curated benchmark suggest promise; larger-scale multi-annotator validation remains future work.

That is more trustworthy and harder for a reviewer to swat away.

#### Specific edits

Current abstract language:

> “We present Neural Search, a novel retrieval system...”

Recommended replacement:

> “We present Neural Search, a prototype framework for provenance-aware, experiment-aware retrieval of reusable neuroscience datasets...”

Current empirical language:

> “These results demonstrate...”

Recommended replacement:

> “These initial prototype results suggest...”

Current broad bridge-to-AI language should be shortened unless directly supported by implementation artifacts.

---

### 2. Claim Discipline

The paper already includes a claim status table around lines 300–314. That is excellent, but it should become more rigorous and visible.

#### Current issue

The paper reports strong metrics:

- Precision@5: 76.7%
- Recall@10: 87.8%
- MRR: 0.950
- NDCG@10: 0.937
- Hard-negative violations: 0/50

These are useful prototype metrics, but they are based on only 30 expert-curated queries and one annotation flow. The limitations section correctly acknowledges this, but the main claims still sound too conclusive.

#### Recommended claim ledger

Add a table near the end of the introduction:

| Claim | Status | Evidence Artifact | Risk | Required Upgrade |
|---|---|---|---|---|
| Structured metadata improves retrieval over keyword search | Prototype validated | 30-query benchmark | Small corpus/query set | Expand to 200+ queries and 500+ datasets |
| Hard-negative filtering reduces invalid matches | Prototype validated | 0/50 violations | Hand-built adversarial set | Add compositional negation benchmark |
| Ontology matching improves constraint satisfaction | Prototype validated | Ablation ladder | Ontology coverage incomplete | Add ontology coverage report |
| Graph metapaths improve dataset relatedness | Partial | Graph infrastructure | No pairwise linkage benchmark | Build dataset-pair benchmark |
| Analysis affordance search identifies reusable datasets | Partial/prototype | Rule detectors | Not validated by actual analysis attempts | Build affordance validation suite |
| Learned embeddings improve retrieval | Partial | Dense baseline described | No systematic model comparison | Add SPECTER2/SciBERT/PubMedBERT/ColBERT evaluation |
| Content-derived neural signature search works | Proposed | Schema only | No extraction/eval | Build NWB/BIDS feature extractor |
| Cross-species experimental alignment works | Proposed | Ontology structure | No benchmark | Build cross-species alignment benchmark |

This table should prevent the paper from sounding like a velvet rope around unfinished machinery.

---

### 3. Existing Infrastructure Must Be Treated as Serious Prior Art

The paper should stop implying that existing systems merely fail. A more reputable posture is:

> Existing repositories and knowledge graphs solve important pieces of the problem. Neural Search builds a retrieval and reasoning layer over these assets, with a focus on experimental constraints, analysis affordances, and auditable dataset linkages.

#### DANDI

DANDI is not just a source of raw data. It is a BRAIN Initiative archive for standardized neurophysiology data, especially NWB. The paper should treat it as a structural backbone.

Current paper line ~230 says DANDI hosts “over 500 dandisets containing petabytes.” This needs correction or timestamping. A 2025 Scientific Data paper states that DANDI held over 400 NWB dandisets at the time, while a current DANDI DataLad/GitHub snapshot reports 1138 dandisets and 965.0 TB. These numbers change, so the paper should programmatically generate source-count snapshots and cite date/version.

Suggested wording:

> DANDI has rapidly grown as a standardized archive for neurophysiology datasets in NWB. Because repository scale changes continuously, Neural Search records source snapshots with access date, adapter version, dataset count, byte count when available, and content hashes.

Sources:

- DANDI Scientific Data 2025 article: https://www.nature.com/articles/s41597-025-06285-x
- DANDI GitHub/DataLad snapshot: https://github.com/dandi
- DANDI documentation: https://docs.dandiarchive.org/

#### OpenNeuro

Current line ~230 says OpenNeuro has thousands of datasets and millions of hours of MRI scanner time. This should be changed. The current OpenNeuro homepage reports 1,741 public datasets and 78,785 participants.

Suggested wording:

> OpenNeuro provides a large public archive of BIDS-compliant MRI, PET, MEG, EEG, and iEEG datasets. Neural Search should ingest OpenNeuro through BIDS-aware metadata adapters and preserve task, modality, participant, session, and derivative structure.

Sources:

- OpenNeuro homepage: https://openneuro.org/
- OpenNeuro paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8550750/

#### EBRAINS Knowledge Graph and openMINDS

EBRAINS Knowledge Graph is a direct precedent. The paper should not treat KG-based neuroscience discovery as mostly absent. EBRAINS links neuroscience data, models, and software using the openMINDS metadata model.

Recommended differentiation:

> EBRAINS KG provides FAIR metadata and graph-based organization of neuroscience assets. Neural Search complements this by adding retrieval-time experimental constraint satisfaction, hard-negative handling, analysis affordance validation, hybrid lexical/dense retrieval, and benchmarked dataset reusability ranking.

Sources:

- EBRAINS Data and Knowledge: https://ebrains.eu/data-tools-services/data-knowledge
- EBRAINS Find Data: https://ebrains.eu/data-tools-services/data-knowledge/find-data
- EBRAINS metadata structures/openMINDS docs: https://docs.kg.ebrains.eu/8387ccd27a186dea3dd0b949dc528842/metadata_structures.html

#### NWB and BIDS

NWB and BIDS should be treated as programmatic affordance surfaces, not just file standards.

- NWB enables structured extraction of neurophysiology metadata, trials, units, devices, electrodes, processing modules, imaging planes, behavior signals, and time intervals.
- BIDS enables structured extraction of participants, sessions, tasks, modalities, runs, events, channels, derivatives, and sidecar metadata.

The core claim should become:

> Neural Search derives reusability signals from standards-aware inspection of NWB/BIDS structure, not only from repository descriptions.

Sources:

- NWB: https://nwb.org/
- BIDS: https://bids.neuroimaging.io/

#### DataCite, RO-Crate, FAIR, PROV-O

The provenance story should be upgraded from custom confidence scoring into standards alignment.

Recommended alignments:

- Use DataCite identifiers and related identifiers for dataset-paper and dataset-dataset linkages.
- Use RO-Crate concepts for packaging dataset metadata, workflows, licenses, contributors, files, and provenance.
- Use PROV-O concepts for entity/activity/agent provenance where possible.
- Use FAIR as the conceptual foundation for findability, interoperability, and reusability.

Sources:

- FAIR principles: https://www.nature.com/articles/sdata201618
- DataCite RelatedIdentifier docs: https://support.datacite.org/docs/connecting-to-works
- RO-Crate: https://www.researchobject.org/specs/
- PROV-O: https://www.w3.org/TR/prov-o/

---

### 4. Related Work Needs More Direct Retrieval Baselines

The current related work includes BM25, dense retrieval, ColBERT, RRF, knowledge graph embeddings, and scientific KGs. That is good, but it needs sharper baselines and more modern scientific retrieval references.

#### Add or strengthen these references

1. **BEIR**  
   Useful because it showed that dense retrieval does not automatically dominate BM25 across diverse retrieval tasks. This supports your hybrid approach.

2. **SPECTER2**  
   Scientific-document embeddings with task-specific adapters across multiple fields and tasks. Directly relevant for paper/dataset-description embedding.

3. **ColBERTv2**  
   Late-interaction retrieval is relevant because dataset cards are structured and terminology-sensitive. ColBERT-style token-level matching may outperform single-vector bi-encoders for scientific constraints.

4. **SciRepEval**  
   Useful as a scientific embedding evaluation framing.

5. **OpenAlex/Semantic Scholar**  
   Useful for dataset-paper linkage and citation graph enrichment.

Sources:

- SPECTER2 blog: https://allenai.org/blog/specter2-adapting-scientific-document-embeddings-to-multiple-fields-and-task-formats-c95686c06567
- SPECTER2 Hugging Face: https://huggingface.co/allenai/specter2
- SPECTER2 GitHub: https://github.com/allenai/SPECTER2
- ColBERTv2 paper: https://arxiv.org/abs/2112.01488
- ColBERTv2 ACL Anthology: https://aclanthology.org/2022.naacl-main.272/
- BEIR: https://arxiv.org/abs/2104.08663

#### Baseline ladder to add

The experimental section should compare:

1. Keyword search.
2. BM25.
3. Field-weighted BM25.
4. Generic sentence-transformer dense retrieval.
5. Scientific embedding model, e.g. SPECTER2 over dataset cards and linked papers.
6. ColBERT-style late interaction over dataset cards.
7. Hybrid BM25 + dense via reciprocal rank fusion.
8. Hybrid + ontology expansion.
9. Hybrid + graph signals.
10. Hybrid + affordance validation.
11. Hybrid + content-derived neural signatures, once implemented.

The strongest paper will show not that dense retrieval magically works, but that **dense retrieval helps in specific failure modes while structured constraints and ontologies protect scientific correctness**.

---

### 5. Analysis Affordances Are the Most Original Piece

The paper’s affordance section is the crown gear. It formalizes analysis methods as predicates over dataset features. This is exactly what makes the system more than semantic search.

#### Current strength

The paper defines affordances such as:

- Choice decoding.
- Q-learning model fitting.
- Event-aligned PSTH.
- Cross-area analysis.
- Dimensionality reduction.
- Functional connectivity.

This is a powerful design because it asks:

> Does this dataset have the required experimental structure to support this analysis?

Not:

> Does the dataset description contain similar words?

#### Main weakness

The affordance detectors appear rule-based and not yet validated against actual analysis attempts. The limitations section admits this. That admission should move into the main experiment discussion.

#### Needed implementation upgrade

Affordance validation should inspect actual file/schema contents:

For NWB:

- `units` table exists and has spike times or rate data.
- `trials` table exists and has trial start/stop times.
- Trial columns include choice, reward, stimulus, condition, outcome, or relevant labels.
- Behavioral time series exist and are aligned to neural data.
- Electrodes/brain regions are present.
- Imaging planes/ROIs exist for calcium imaging.
- Processing modules contain usable outputs.

For BIDS:

- `events.tsv` exists for relevant runs.
- `participants.tsv` exists.
- task labels exist.
- modality folders are present.
- sidecar JSON files contain required acquisition metadata.
- EEG/MEG/iEEG channel files exist where appropriate.
- derivatives are present when requested.

#### Suggested affordance schema

```yaml
id: q_learning
label: Q-learning model fitting
required_features:
  - trial_table
  - ordered_trials
  - choice_sequence
  - reward_signal
  - outcome_labels
optional_features:
  - reaction_time
  - stimulus_identity
  - block_or_context_label
  - session_id
negative_conditions:
  - only_summary_statistics
  - no_trialwise_behavior
validation_methods:
  - nwb_trials_column_check
  - bids_events_column_check
confidence_rules:
  high: all required features found in structured files
  medium: required features inferred from metadata but not verified in files
  low: only textual evidence
```

#### Add false-positive examples

The paper should include examples where a dataset is topically relevant but not reusable:

- Mentions “decision-making” in description but has no trial-level choices.
- Has neural recordings and rewards but no behavioral event alignment.
- Has fMRI task data but no condition labels.
- Has calcium imaging but no ROI masks or deconvolved traces.
- Has behavior videos but no synchronized timestamps.

That is where the system’s value becomes obvious.

---

### 6. Embedding Architecture Needs To Become Real Retrieval Machinery

The embedding section is ambitious, but currently reads more like architecture poetry than a validated system. It should be split into:

1. **Implemented embedding retrieval.**
2. **Planned embedding retrieval.**
3. **Content-derived neural signatures.**

#### Three-layer embedding strategy

##### Layer A: Dataset-card text embeddings

Build canonical dataset cards and embed them.

Each card should include:

```text
title
description
repository
species
modality
brain region
task
stimuli
behavioral events
recording type
NWB/BIDS fields
linked papers
analysis affordances
quality/provenance summary
license/version/access info
```

Named vector fields:

```yaml
embedding_text_general
embedding_scientific_specter
embedding_task_context
embedding_modality_context
embedding_region_context
embedding_affordance_context
```

Candidate models:

- `sentence-transformers/all-MiniLM-L6-v2` or similar baseline.
- `allenai/specter2` for scientific title/abstract/dataset-card retrieval.
- SciBERT/PubMedBERT/BioBERT variants for biomedical language.
- ColBERT or ColBERTv2 for late interaction.

##### Layer B: Structured graph embeddings

Use graph structure to support dataset linkage.

Edges:

```text
dataset -> paper
dataset -> modality
dataset -> species
dataset -> brain_region
dataset -> task
dataset -> stimulus
dataset -> analysis_affordance
dataset -> repository
dataset -> lab
dataset -> method
paper -> method
paper -> dataset
paper -> citation
paper -> software
```

Start simple:

- node2vec/metapath2vec for quick graph embeddings.
- Then GraphSAGE/HGT if needed.
- Use graph embeddings for candidate expansion and dataset-dataset relatedness, not as the only retrieval signal.

Named vector fields:

```yaml
embedding_graph_node2vec
embedding_graph_metapath_task
embedding_graph_metapath_method
embedding_graph_dataset_pair
```

##### Layer C: Content-derived neural signatures

This is the moonshot and should be explicitly labeled as future or experimental until implemented.

Extract fingerprints from actual NWB/BIDS contents:

```yaml
sampling_rate
recording_duration
number_of_units_or_rois_or_channels
trial_count
event_types
stimulus_types
brain_regions
inter_spike_interval_summary
firing_rate_distribution
psth_modulation_summary
roi_trace_statistics
lfp_bandpower_summary
behavior_alignment_score
population_dimensionality_estimate
missingness_profile
quality_metrics
```

This enables queries like:

> Find datasets with similar event-aligned population dynamics to this session.

That is the dragon egg. Protect it by building it carefully.

---

### 7. Dataset Linkage Should Be a Product, Not a Side Effect

The paper currently describes graph traversal and metapaths, but dataset linkage should become its own first-class evaluation target.

#### Linkage types

```yaml
same_task_family
same_modality
same_species
same_brain_region
same_behavioral_events
same_stimulus_family
shares_publication
cites_same_method
uses_same_software
uses_same_standard
same_lab_or_consortium
same_instrument
derived_from
is_supplement_to
is_reanalysis_of
compatible_for_meta_analysis
compatible_for_model_training
compatible_for_affordance
similar_neural_signature
```

#### Edge schema

```yaml
edge_id: string
source_id: string
target_id: string
edge_type: string
confidence: float
evidence:
  - evidence_type: structured_metadata | text_span | doi_relation | file_schema | content_signature | manual_label
    source: string
    field_path: string | null
    text: string | null
    extractor: string
    timestamp: string
review_status: unreviewed | machine_validated | human_validated | rejected
provenance:
  source_repository: string
  adapter_version: string
  corpus_snapshot_id: string
```

Every edge should answer:

> Why do we believe these two things are connected?

No evidence, no edge.

---

### 8. Evaluation Needs Four Benchmarks

#### Benchmark 1: Dataset retrieval

Goal: retrieve datasets satisfying structured scientific queries.

Query examples:

- “mouse Neuropixels decision-making not visual cortex”
- “human fMRI reinforcement learning with trial-level feedback”
- “calcium imaging in mouse visual cortex with natural movie stimuli”
- “datasets with behavior and electrophysiology suitable for choice decoding”

Metrics:

```text
Precision@5
Recall@10
NDCG@10
MRR
hard_negative_violation_rate
constraint_satisfaction_rate
coverage_by_query_type
```

#### Benchmark 2: Affordance validation

Goal: predict whether a dataset can support a specified analysis.

Affordances:

- Event-aligned PSTH.
- Choice decoding.
- Q-learning fitting.
- Cross-area interaction analysis.
- Dimensionality reduction.
- Behavioral state decoding.
- Functional connectivity.
- Stimulus-response modeling.

Metrics:

```text
affordance_precision
affordance_recall
false_positive_rate
missing_required_field_rate
schema_validation_accuracy
manual_review_agreement
```

#### Benchmark 3: Dataset linkage

Goal: given a seed dataset, retrieve scientifically useful related datasets.

Pair labels:

```yaml
0: unrelated
1: topically related
2: experimentally related
3: reusable/comparable
```

Metrics:

```text
pairwise_NDCG@10
link_type_accuracy
edge_precision
edge_evidence_completeness
human_preference_rate
```

#### Benchmark 4: Content-derived neural signature retrieval

Goal: retrieve datasets/sessions based on actual data-derived structure, not just metadata.

Metrics:

```text
task_recovery_rate
modality_recovery_rate
species_recovery_rate
region_recovery_rate
nearest_neighbor_interpretability
cluster_purity
signature_ablation_score
```

This benchmark is the long-term differentiator.

---

### 9. Paper Structure Recommendation

Recommended revised structure:

```text
1. Introduction
   1.1 Dataset discovery as experimental constraint satisfaction
   1.2 Why text search and faceted search are insufficient
   1.3 Contributions with explicit status labels

2. Background and Related Work
   2.1 Scientific dataset search
   2.2 Neuroscience repositories and metadata standards
   2.3 EBRAINS KG/openMINDS and FAIR infrastructure
   2.4 Hybrid retrieval, scientific embeddings, and late interaction
   2.5 Knowledge graph retrieval and provenance

3. Problem Formulation
   3.1 Dataset objects
   3.2 Queries as hard constraints + soft preferences
   3.3 Experimental context slots
   3.4 Reusability and analysis affordances

4. System Architecture
   4.1 Ingestion adapters
   4.2 Dataset cards
   4.3 Ontology normalization
   4.4 Provenance graph
   4.5 Embedding fields
   4.6 Retrieval and fusion
   4.7 Explanation generation

5. Analysis Affordances
   5.1 Formal definition
   5.2 Required-field schemas
   5.3 NWB/BIDS validation
   5.4 Failure cases and false positives

6. Evaluation
   6.1 Corpus snapshot
   6.2 Query benchmark
   6.3 Baseline ladder
   6.4 Hard-negative benchmark
   6.5 Affordance validation
   6.6 Dataset linkage benchmark
   6.7 Error analysis

7. Discussion
   7.1 What structured retrieval gets right
   7.2 Where learned embeddings help
   7.3 Why provenance matters
   7.4 Limitations

8. Roadmap
   8.1 Expanded corpus
   8.2 Neural content signatures
   8.3 Cross-species alignment
   8.4 User studies
```

---

### 10. Concrete Whitepaper Edits

#### Abstract

Replace “novel retrieval system” with “prototype framework.”

Add:

> We distinguish implemented components from proposed extensions and evaluate the current system on a small expert-curated benchmark.

Remove or soften broad “cross-disciplinary discovery” claims unless supported by implemented cross-domain retrieval.

#### Introduction

Replace:

> “The scale of this opportunity is staggering.”

With:

> “The scale and heterogeneity of public neuroscience data make dataset discovery an increasingly structured retrieval problem.”

Fix DANDI/OpenNeuro counts with timestamped snapshots.

Add a table comparing DANDI, OpenNeuro, EBRAINS KG, Google Dataset Search, and Neural Search.

#### Related Work

Add direct sections on:

- EBRAINS KG/openMINDS.
- DANDI/NWB as a structured data archive.
- OpenNeuro/BIDS as a structured neuroimaging archive.
- Scientific embeddings, especially SPECTER2.
- Late-interaction retrieval, especially ColBERTv2.
- FAIR/RO-Crate/DataCite/PROV-O.

#### Methods

Separate current implementation from proposed architecture.

Add an explicit `DatasetCardV1` schema.

Add a `CorpusSnapshot` schema.

Add a `ProvenanceEdge` schema.

Add an `AffordanceRequirement` schema.

#### Experiments

Label current benchmark as prototype-scale.

Add per-query error analysis.

Add baseline implementation details:

- embedding model name/version
- vector dimension
- vector index
- BM25 parameters
- RRF `k`
- ontology expansion rules
- graph edge types
- candidate pool size

Add a “not yet evaluated” table for proposed components.

#### Code and data availability

Current line ~1800 points to `https://github.com/neural-search/neural-search`. This appears placeholder-like and potentially conflicts with existing “neural-search” naming. Replace with the actual repository URL or write:

> Code and benchmark artifacts will be released with the reproducibility package.

Do not include a fake or placeholder repo in a paper.

---

## Part II: Claude Instruction Set

The following is a standalone instruction set you can paste into Claude Code. It assumes Claude has access to development skills. It asks Claude to inspect the repository, read available skill manifests, and apply them to the implementation without inventing unsupported claims.

---

# Claude Code Instruction Set: Upgrade Neural Search Into a Provenance-Aware Scientific Dataset Retrieval System

## Mission

You are working on the Neural Search repository. Your goal is to turn the project into a credible provenance-aware retrieval system for scientific datasets that combines:

1. Structured experimental constraints.
2. Ontology-aware metadata normalization.
3. Evidence-backed knowledge graph linkages.
4. Learned embeddings.
5. Analysis affordance validation.
6. Reproducible evaluation.

The system should identify datasets that are not merely topically similar, but **experimentally reusable**.

Do not focus on frontend polish, demos, or speculative UI. Prioritize the retrieval core, corpus quality, benchmarking, provenance, and scientific correctness.

## Important working style

Before coding, inspect the repo carefully.

1. Read `README.md`, `CLAUDE.md`, `CONTRIBUTING.md`, `pyproject.toml`, `setup.py`, `requirements*.txt`, and any existing architecture docs.
2. Discover available Claude skills. Look for `.claude/skills`, `skills/`, or project-specific skill manifests. Read relevant skill instructions before using them.
3. Use development skills where appropriate for:
   - repository mapping
   - refactoring
   - test generation
   - benchmark creation
   - documentation updates
   - schema design
   - code review
   - scientific writing
4. Keep changes modular and testable.
5. Avoid large rewrites unless absolutely necessary.
6. Preserve existing passing behavior.
7. Create small commits or clearly separable change sets.
8. Every new claim in docs must point to an artifact, test, benchmark, or explicit future-work label.

## Hard rules

- Do not fabricate benchmark results.
- Do not claim a component is validated unless a test or benchmark supports it.
- Do not silently change public APIs without migration notes.
- Do not add heavyweight dependencies without documenting why.
- Do not build frontend features unless needed for testing the retrieval core.
- Do not make the whitepaper sound more complete than the system is.
- Do not replace structured constraints with pure embeddings. Dense retrieval is a recall layer, not the whole brainstem.

---

## Phase 0: Repository Archaeology and System Map

### Goal

Understand the current codebase and produce a concise architecture map before implementing new features.

### Tasks

1. Inspect current modules for:
   - ingestion
   - metadata normalization
   - retrieval
   - embeddings
   - graph schema/building/querying
   - affordance detection
   - evaluation/benchmarks
   - tests
   - docs/whitepaper

2. Produce `docs/CURRENT_SYSTEM_MAP.md` with:
   - module map
   - current data flow
   - existing retrievers
   - existing schemas
   - current tests and benchmark commands
   - known gaps
   - recommended implementation order

3. Produce or update `docs/CLAIM_LEDGER.md` with:

```markdown
| Claim | Status | Evidence Artifact | Risk | Next Validation |
|---|---|---|---|---|
```

Statuses must be one of:

```text
implemented
prototype_validated
partially_implemented
proposed
not_started
```

### Acceptance criteria

- `docs/CURRENT_SYSTEM_MAP.md` exists and accurately reflects the repo.
- `docs/CLAIM_LEDGER.md` exists.
- No code behavior is changed in this phase unless needed to fix broken tests.
- Existing tests still pass or known failures are documented.

Suggested commands:

```bash
find . -maxdepth 3 -type f | sort
pytest -q
python -m compileall .
ruff check . || true
```

Adapt commands to the repo’s actual tooling.

---

## Phase 1: Corpus Snapshot and Dataset Card V1

### Goal

Create a stable, auditable data object that every retrieval signal can use.

### Rationale

Embedding search, BM25, graph construction, affordance detection, and explanations should not each invent their own view of a dataset. They should share a canonical `DatasetCardV1` object.

### Tasks

1. Define `DatasetCardV1` schema.
2. Define `CorpusSnapshot` schema.
3. Add serialization to JSON/JSONL.
4. Add validation tests.
5. Add a command to export dataset cards from the normalized corpus.
6. Add source snapshot metadata: source name, adapter version, retrieval date, dataset count, hash, and optional byte count.

### Suggested schema

```python
@dataclass
class DatasetCardV1:
    dataset_id: str
    source: str
    source_url: str | None
    version: str | None
    title: str
    description: str | None
    license: str | None
    organism: list[str]
    species: list[str]
    modality: list[str]
    brain_region: list[str]
    task: list[str]
    stimuli: list[str]
    behavioral_events: list[str]
    data_standards: list[str]  # NWB, BIDS, etc.
    file_modalities: list[str]
    n_subjects: int | None
    n_sessions: int | None
    n_trials: int | None
    linked_publications: list[str]
    linked_dois: list[str]
    analysis_affordances: list[str]
    quality_flags: list[str]
    provenance: list[dict]
    text_card: str
```

```python
@dataclass
class CorpusSnapshot:
    snapshot_id: str
    created_at: str
    repo_commit: str | None
    source_counts: dict[str, int]
    adapters: dict[str, str]
    records_hash: str
    notes: str | None
```

### Acceptance criteria

- Dataset cards can be exported deterministically.
- Snapshot hash changes only when records change.
- Tests verify required fields, serialization, stable hashing, and text-card generation.
- Existing retrievers can consume dataset cards or have a clear adapter path.

Suggested tests:

```bash
pytest tests/test_dataset_card.py tests/test_corpus_snapshot.py -q
```

---

## Phase 2: Structured Constraint Parser and Hard-Negative Guardrail

### Goal

Make scientific constraints explicit before retrieval.

### Rationale

A query like “mouse Neuropixels decision-making NOT visual cortex” should not rely on dense embeddings. Hard constraints should be parsed, normalized, and enforced.

### Tasks

1. Define `QueryIntent` / `StructuredQuery` schema.
2. Parse:
   - species constraints
   - modality constraints
   - task constraints
   - brain-region constraints
   - required affordances
   - excluded concepts
   - soft preferences
3. Add ontology normalization for synonyms.
4. Add hard-negative filtering before final ranking.
5. Add tests for negation scope and exclusions.

### Suggested schema

```python
@dataclass
class StructuredQuery:
    raw_query: str
    required: dict[str, list[str]]
    preferred: dict[str, list[str]]
    excluded: dict[str, list[str]]
    affordances: list[str]
    query_type: str
    confidence: float
```

### Hard-negative examples

```text
mouse Neuropixels decision-making not visual cortex
human fMRI reinforcement learning excluding resting state
calcium imaging in V1 but not auditory cortex
datasets with behavior but no imaging
NWB electrophysiology without optogenetic perturbation
```

### Acceptance criteria

- Hard exclusions are enforced before final results.
- Tests cover single negation, multi-term negation, parenthetical negation if supported, and ambiguous negation.
- Retrieval result explanations show which constraints were satisfied or filtered.

---

## Phase 3: Ontology-Aware Metadata Normalization

### Goal

Improve scientific precision by normalizing entities to controlled concepts.

### Tasks

1. Create or extend ontology registries for:
   - species
   - modalities
   - brain regions
   - behavioral tasks
   - stimuli
   - recording technologies
   - analysis methods
   - file/data standards
2. Add synonym expansion with provenance.
3. Add normalization confidence.
4. Add tests for common aliases:
   - `ephys` -> electrophysiology
   - `Neuropixels` -> extracellular electrophysiology / high-density silicon probe
   - `go/no-go` -> go_nogo
   - `PFC` -> prefrontal cortex
   - `V1` -> primary visual cortex
   - `2p` -> two-photon calcium imaging
5. Add coverage report.

### Acceptance criteria

- Normalized dataset cards include controlled IDs and original text evidence.
- Tests cover alias normalization and ambiguous terms.
- `docs/ONTOLOGY_COVERAGE_REPORT.md` summarizes coverage and gaps.

---

## Phase 4: Provenance-Weighted Knowledge Graph

### Goal

Build an evidence-backed graph that supports dataset linkage and explanations.

### Tasks

1. Define graph node types:
   - dataset
   - paper
   - DOI
   - repository
   - species
   - modality
   - brain_region
   - task
   - stimulus
   - method
   - software
   - affordance
   - lab/consortium if available
2. Define graph edge types:
   - has_species
   - has_modality
   - has_region
   - has_task
   - has_stimulus
   - supports_affordance
   - described_by_paper
   - cites_paper
   - uses_software
   - uses_standard
   - related_dataset
   - derived_from
   - is_supplement_to
   - compatible_for_meta_analysis
   - compatible_for_model_training
3. Define `ProvenanceEdge` schema.
4. Each edge must include evidence.
5. Add graph export to JSONL or another stable format.
6. Add graph query utilities:
   - neighbors by edge type
   - metapath traversal
   - dataset-pair explanation
   - confidence aggregation

### Suggested edge schema

```python
@dataclass
class ProvenanceEvidence:
    evidence_type: str
    source: str
    field_path: str | None
    text: str | None
    extractor: str
    timestamp: str
    confidence: float

@dataclass
class ProvenanceEdge:
    source_id: str
    target_id: str
    edge_type: str
    confidence: float
    evidence: list[ProvenanceEvidence]
    review_status: str
    corpus_snapshot_id: str
```

### Acceptance criteria

- No graph edge exists without evidence.
- Tests cover confidence aggregation and edge provenance.
- Dataset-pair explanation returns human-readable evidence.
- Graph construction is deterministic for a fixed corpus snapshot.

---

## Phase 5: True Embedding-Based Retrieval

### Goal

Implement real embedding-based retrieval as a measurable component, not just an architecture diagram.

### Rationale

Embeddings should help with semantic recall, relatedness, paraphrases, and scientific language variation. They should not override hard constraints.

### Tasks

1. Add embedding provider abstraction.
2. Add named embedding fields.
3. Embed `DatasetCardV1.text_card`.
4. Add model versioning and embedding metadata.
5. Add vector index abstraction.
6. Add dense retrieval baseline.
7. Add hybrid fusion with BM25 using reciprocal rank fusion.
8. Add evaluation comparing dense-only, BM25-only, and hybrid.

### Suggested embedding fields

```yaml
embedding_text_general
embedding_scientific_specter
embedding_task_context
embedding_modality_context
embedding_region_context
embedding_affordance_context
embedding_graph_node2vec
embedding_neural_signature  # future
```

### Provider interface

```python
class EmbeddingProvider(Protocol):
    name: str
    version: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        ...
```

### Embedding record schema

```python
@dataclass
class EmbeddingRecord:
    entity_id: str
    field: str
    vector: list[float]
    provider: str
    provider_version: str
    dimension: int
    text_hash: str
    created_at: str
    corpus_snapshot_id: str
```

### Baselines to implement

At minimum:

1. BM25.
2. Dense single-vector sentence-transformer or local fallback.
3. Hybrid BM25 + dense via RRF.
4. Hybrid + ontology constraints.
5. Hybrid + graph features.
6. Hybrid + affordance scoring.

If practical, add SPECTER2 as an optional provider. If heavyweight model downloads are unsuitable for CI, gate it behind an optional extra and use small deterministic test embeddings in unit tests.

### Acceptance criteria

- Embeddings are cached and versioned.
- Dense retrieval can be run independently.
- Hybrid retrieval can be evaluated through the same benchmark harness.
- Tests do not require network access.
- Benchmark output includes per-query results and aggregate metrics.

---

## Phase 6: Graph Embeddings and Dataset Linkage

### Goal

Make dataset linkage measurable.

### Tasks

1. Build dataset-dataset candidate pairs from graph metapaths.
2. Implement simple graph embeddings or graph similarity scores.
3. Add `DatasetLinkageResult` schema.
4. Add pairwise explanations.
5. Create a seed benchmark of dataset pairs.
6. Add metrics for pairwise relatedness.

### Suggested linkage result schema

```python
@dataclass
class DatasetLinkageResult:
    source_dataset_id: str
    target_dataset_id: str
    linkage_types: list[str]
    score: float
    evidence: list[ProvenanceEvidence]
    explanation: str
```

### Linkage types

```text
same_task_family
same_modality
same_species
same_brain_region
same_behavioral_events
same_stimulus_family
shares_publication
cites_same_method
uses_same_software
uses_same_standard
same_lab_or_consortium
same_instrument
compatible_for_meta_analysis
compatible_for_model_training
similar_neural_signature
```

### Acceptance criteria

- Given a seed dataset, system returns related datasets with typed reasons.
- Pairwise linkage benchmark exists, even if initially small.
- Results distinguish topical relatedness from experimental comparability.

---

## Phase 7: Analysis Affordance Validation

### Goal

Move affordances from rule labels to verified reusability predicates.

### Tasks

1. Define `AffordanceRequirement` schema.
2. Map affordances to required and optional fields.
3. Implement validators for NWB and BIDS where possible.
4. Store validation evidence.
5. Add false-positive and false-negative tests.
6. Create benchmark file for manual review.

### Suggested affordances

```text
event_aligned_psth
choice_decoding
q_learning
stimulus_response_modeling
behavioral_state_decoding
cross_area_interaction
dimensionality_reduction
functional_connectivity
trial_aligned_calcium_analysis
pose_neural_correlation
```

### Suggested schema

```python
@dataclass
class AffordanceRequirement:
    affordance_id: str
    label: str
    required_features: list[str]
    optional_features: list[str]
    negative_conditions: list[str]
    validators: list[str]
```

```python
@dataclass
class AffordanceValidationResult:
    dataset_id: str
    affordance_id: str
    supported: bool
    confidence: float
    missing_required_features: list[str]
    found_features: list[str]
    evidence: list[ProvenanceEvidence]
```

### Acceptance criteria

- Affordance labels include evidence and missing fields.
- Query result explanations show why an affordance is supported.
- Tests cover datasets that mention an analysis but lack required fields.
- False positives are explicitly tracked.

---

## Phase 8: Evaluation Harness and Benchmark Expansion

### Goal

Make retrieval quality measurable and reproducible.

### Tasks

1. Create benchmark query schema.
2. Add query sets:
   - normal scientific queries
   - hard-negative queries
   - affordance queries
   - dataset-linkage queries
   - metadata-robustness queries
3. Add metrics:
   - Precision@K
   - Recall@K
   - NDCG@K
   - MRR
   - hard-negative violation rate
   - constraint satisfaction rate
   - affordance precision/recall
   - pairwise linkage NDCG
4. Add per-query failure reports.
5. Add ablation runner.
6. Add benchmark result export to Markdown and JSON.

### Suggested query schema

```yaml
query_id: q001
query: mouse Neuropixels decision-making not visual cortex
query_type: structured_retrieval
required:
  species: [mouse]
  modality: [neuropixels]
  task: [decision_making]
excluded:
  brain_region: [visual_cortex]
relevant_dataset_ids:
  - dataset_a
  - dataset_b
hard_negative_dataset_ids:
  - dataset_c
notes: Requires decision-making behavior and neural recordings.
```

### Acceptance criteria

- One command runs the benchmark.
- Benchmark results are deterministic for fixed corpus snapshot and model cache.
- Per-query failure report identifies violated constraints, missed synonyms, bad metadata, and false affordance predictions.
- No benchmark numbers are added to the paper unless generated by this harness.

Suggested command:

```bash
python -m neural_search.eval.run_benchmark \
  --corpus data/processed/dataset_cards.jsonl \
  --queries benchmarks/queries_v1.yaml \
  --retrievers bm25,dense,hybrid,hybrid_ontology,hybrid_graph,full \
  --out reports/benchmark_v1
```

Adapt module paths to the actual repo.

---

## Phase 9: Content-Derived Neural Signature Prototype

### Goal

Start the long-term differentiator: retrieval based on actual dataset contents.

### Scope

Keep this prototype small. Do not try to train a massive model yet.

### Tasks

1. Select a small number of NWB datasets.
2. Extract simple content fingerprints:
   - duration
   - number of units/ROIs/channels
   - trial count
   - event labels
   - sampling rate
   - brain regions
   - firing-rate summary
   - ISI summary
   - PSTH summary if trials exist
   - missingness/quality profile
3. Store `NeuralSignatureV1` records.
4. Add simple similarity search over signatures.
5. Compare metadata-only retrieval vs signature-aware retrieval on a tiny benchmark.

### Suggested schema

```python
@dataclass
class NeuralSignatureV1:
    dataset_id: str
    asset_id: str | None
    modality: str
    duration_seconds: float | None
    n_units: int | None
    n_rois: int | None
    n_channels: int | None
    n_trials: int | None
    event_types: list[str]
    brain_regions: list[str]
    summary_vector: list[float]
    feature_names: list[str]
    extraction_version: str
    evidence: list[ProvenanceEvidence]
```

### Acceptance criteria

- Works on at least a few local or mocked NWB-like records.
- Does not require huge downloads in CI.
- Produces a report explaining what content signatures can and cannot prove.
- Clearly labeled as experimental.

---

## Phase 10: Whitepaper Rebuild

### Goal

Update the whitepaper so its claims match the system.

### Tasks

1. Rewrite the abstract with prototype-accurate language.
2. Add comparison table against:
   - DANDI
   - OpenNeuro
   - EBRAINS KG/openMINDS
   - Google Dataset Search
   - generic vector search
   - Neural Search
3. Add `DatasetCardV1`, `CorpusSnapshot`, `ProvenanceEdge`, and `AffordanceRequirement` schemas.
4. Replace stale repository counts with timestamped snapshot counts.
5. Move limitations closer to the results.
6. Add reproducibility section with exact benchmark commands.
7. Add a generated benchmark report table only from actual outputs.
8. Remove fake or placeholder repo URLs.
9. Add a claim ledger appendix.

### Acceptance criteria

- Every major claim is backed by an artifact or marked future work.
- Paper compiles.
- No unsupported benchmark numbers.
- No fake repository links.
- References include EBRAINS KG/openMINDS, DANDI/NWB, OpenNeuro/BIDS, FAIR, DataCite, RO-Crate, SPECTER2, ColBERTv2, and BEIR.

---

## Recommended File/Directory Additions

Adapt names to the actual repo structure.

```text
neural_search/
  schemas/
    dataset_card.py
    corpus_snapshot.py
    provenance.py
    affordance.py
    query.py
    embeddings.py
  ingestion/
    dandi_adapter.py
    openneuro_adapter.py
    nwb_inspector.py
    bids_inspector.py
  normalization/
    ontology_registry.py
    synonyms.py
    normalize.py
  retrieval/
    bm25.py
    dense.py
    hybrid.py
    constraints.py
    rrf.py
  graph/
    schema.py
    builder.py
    linkage.py
    metapaths.py
  affordances/
    requirements.py
    validators.py
    registry.py
  embeddings/
    providers.py
    cache.py
    vector_index.py
  signatures/
    neural_signature.py
    nwb_features.py
  eval/
    metrics.py
    run_benchmark.py
    ablations.py
    failure_report.py
benchmarks/
  queries_v1.yaml
  hard_negatives_v1.yaml
  affordance_validation_v1.yaml
  dataset_linkage_pairs_v1.yaml
docs/
  CURRENT_SYSTEM_MAP.md
  CLAIM_LEDGER.md
  ONTOLOGY_COVERAGE_REPORT.md
  RETRIEVAL_ARCHITECTURE.md
  EVALUATION_PROTOCOL.md
reports/
  benchmark_v1/
```

---

## Definition of Done

The next major version is successful when:

1. The repo exports deterministic dataset cards with provenance.
2. Search supports hard scientific constraints and hard-negative filtering.
3. Ontology normalization works for common neuroscience synonyms.
4. Knowledge graph edges are evidence-backed.
5. Dense retrieval is implemented, versioned, benchmarked, and compared to BM25.
6. Hybrid retrieval beats at least one baseline on a reproducible benchmark without increasing hard-negative violations.
7. Analysis affordance predictions include actual evidence and missing-field explanations.
8. Dataset linkage returns typed, explainable relationships.
9. The benchmark harness exports aggregate metrics and per-query failure reports.
10. The whitepaper’s claims exactly match implemented and benchmarked artifacts.

---

## First Claude Prompt To Run

Paste this into Claude Code first:

```text
You are working in the Neural Search repository. Your mission is to upgrade this project into a credible provenance-aware retrieval system for scientific datasets. Before coding, inspect the repo and any available Claude skills. Read README.md, CLAUDE.md, pyproject/setup files, docs, tests, and any .claude/skills or skills manifests. Use relevant skills for repo mapping, testing, schema design, refactoring, benchmarking, and documentation.

Do not start with frontend work. Focus on core retrieval, corpus quality, provenance, embeddings, ontology-aware normalization, graph linkage, affordance validation, and reproducible evaluation.

First deliverables:
1. docs/CURRENT_SYSTEM_MAP.md describing current modules, data flow, retrievers, graph pieces, embedding pieces, affordance pieces, tests, and known gaps.
2. docs/CLAIM_LEDGER.md listing each major system/whitepaper claim with status: implemented, prototype_validated, partially_implemented, proposed, or not_started.
3. A recommended implementation order for the next 5 coding milestones.
4. Run the existing test/lint commands you discover and document results.

Do not fabricate results. Do not rewrite the whole repo. Preserve existing passing tests. Make small, reviewable changes.
```

---

## Second Claude Prompt: Implement Dataset Cards and Corpus Snapshots

```text
Continue from the system map and claim ledger. Implement DatasetCardV1 and CorpusSnapshot as the canonical auditable objects for retrieval. Adapt to the existing repo structure.

Requirements:
- DatasetCardV1 should capture source, source URL, version, title, description, license, organism/species, modality, brain region, task, stimuli, behavioral events, data standards, file modalities, subjects/sessions/trials if available, linked publications/DOIs, analysis affordances, quality flags, provenance, and a generated text_card field for retrieval.
- CorpusSnapshot should capture snapshot_id, created_at, repo_commit if available, source_counts, adapter versions, records_hash, and notes.
- Add deterministic JSON/JSONL serialization.
- Add tests for required fields, stable hashing, text_card generation, and roundtrip serialization.
- Add or update docs explaining how dataset cards feed BM25, dense retrieval, graph construction, affordance detection, and explanations.

Run relevant tests and report exact commands/results.
```

---

## Third Claude Prompt: Implement True Embedding Retrieval

```text
Implement true embedding-based retrieval as a measured component, not a diagram. Preserve hard constraints as guardrails.

Requirements:
- Add an EmbeddingProvider abstraction with name, version, dimension, and embed_texts().
- Add an EmbeddingRecord schema with entity_id, field, vector, provider, provider_version, dimension, text_hash, created_at, and corpus_snapshot_id.
- Add named embedding fields, at minimum embedding_text_general and embedding_affordance_context. Add SPECTER2 as optional only if dependency/runtime constraints are reasonable; otherwise document it as an optional provider and use a deterministic local test provider for CI.
- Add vector cache/versioning.
- Add dense retriever over DatasetCardV1.text_card.
- Add hybrid BM25+dense retrieval with reciprocal rank fusion.
- Add benchmark support comparing bm25, dense, and hybrid on the same query set.
- Add tests that do not require network access.

Run tests and produce a short report with any benchmark outputs generated by the repo, clearly marked as local/prototype if small.
```

---

## Fourth Claude Prompt: Provenance Graph and Dataset Linkage

```text
Build evidence-backed dataset linkage.

Requirements:
- Define ProvenanceEvidence and ProvenanceEdge schemas.
- Ensure every graph edge has evidence, source, extractor, timestamp, confidence, review_status, and corpus_snapshot_id.
- Add graph node/edge types for dataset, paper, DOI, repository, species, modality, brain_region, task, stimulus, method, software, affordance, and standard.
- Add dataset-to-dataset linkage results with typed linkage reasons such as same_task_family, same_modality, same_species, same_brain_region, same_behavioral_events, shares_publication, cites_same_method, uses_same_standard, compatible_for_meta_analysis, compatible_for_model_training.
- Add pairwise explanation generation.
- Add tests for deterministic graph construction, edge evidence requirements, confidence aggregation, and dataset-pair explanations.
- Add docs/RETRIEVAL_ARCHITECTURE.md or update existing docs.

Do not create edges without evidence. No evidence, no edge.
```

---

## Fifth Claude Prompt: Affordance Validation

```text
Upgrade analysis affordances from rule labels to validated reusability predicates.

Requirements:
- Define AffordanceRequirement and AffordanceValidationResult schemas.
- Create a registry for affordances including event_aligned_psth, choice_decoding, q_learning, stimulus_response_modeling, behavioral_state_decoding, cross_area_interaction, dimensionality_reduction, functional_connectivity, trial_aligned_calcium_analysis, and pose_neural_correlation.
- Each affordance must define required_features, optional_features, negative_conditions, and validators.
- Implement validators against available structured metadata. If NWB/BIDS file inspection exists, use it. If not, create clean interfaces and tests with mocked structured records.
- Store found features, missing required features, confidence, and provenance evidence.
- Add tests for false positives: datasets whose text mentions an analysis but whose structured fields do not support it.
- Update retrieval explanations to show why an affordance is supported or unsupported.

Run tests and document remaining limitations.
```

---

## Sixth Claude Prompt: Evaluation Harness and Whitepaper Repair

```text
Build the evaluation harness and repair the whitepaper so claims match artifacts.

Requirements:
- Add benchmark query schema supporting required constraints, preferred fields, excluded fields, relevant_dataset_ids, hard_negative_dataset_ids, and notes.
- Add metrics: Precision@K, Recall@K, NDCG@K, MRR, hard_negative_violation_rate, constraint_satisfaction_rate, affordance precision/recall where labels exist, and pairwise linkage metrics if labels exist.
- Add ablation runner comparing keyword/BM25/dense/hybrid/hybrid+ontology/hybrid+graph/full where implemented.
- Add per-query failure report identifying missed synonyms, violated constraints, false affordances, graph mistakes, and metadata gaps.
- Export benchmark outputs to JSON and Markdown.
- Update the whitepaper: soften claims, correct DANDI/OpenNeuro counts with timestamped snapshot wording, add EBRAINS/openMINDS comparison, add DatasetCardV1/CorpusSnapshot/ProvenanceEdge/AffordanceRequirement schemas, and remove unsupported or placeholder claims.
- Do not add benchmark numbers to the paper unless generated by the benchmark harness.

Run tests, benchmark commands if feasible, and LaTeX compilation if configured.
```

---

## Final Advice to Future You

Do not try to win by making the paper sound huge. Win by making the system **hard to dismiss**.

The strongest version of Neural Search is not “LLM searches neuroscience data.” It is:

> A retrieval system that knows the difference between a dataset that merely talks about decision-making and a dataset that actually contains the neural recordings, trial events, behavioral labels, provenance, and structure required to run a decision-making analysis.

That is the difference between topical similarity and experimental reusability.

Build that difference into the schema, the graph, the embeddings, the validator, the benchmark, and the paper. Then the project stops looking like a clever search demo and starts looking like scientific infrastructure.
