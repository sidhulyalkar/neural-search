# Project Vision

Neural Search is a **provenance-backed research-object retrieval and experimental design engine** for neural and behavioral data — built on a real, growing corpus and knowledge graph, not a demo fixture.

## Mission

Enable researchers to discover, evaluate, and reuse neuroscience datasets through experiment-aware search that understands scientific intent, verifies claims with evidence, and accelerates the path from research question to analysis — and, increasingly, all the way to a synchronized, shareable view of the experiment itself.

## Core Thesis

Generic semantic search and generic RAG are insufficient for scientific dataset reuse.

**RAG answers questions by retrieving text and generating prose.** Neural Search retrieves **research objects**. A good result is not a paragraph — it is a dataset with:

| Component | Description |
|-----------|-------------|
| **Experimental Labels** | Task type, behavioral events, brain regions, species, modality — grounded in a scientific ontology |
| **Metadata Constraints** | Structured filters that can be precisely satisfied or violated |
| **Provenance** | Source archive IDs, linked papers (5 sources), retraction/correction status, extraction confidence |
| **Analysis Affordances** | What analyses this dataset can actually support, with required signals identified |
| **Match Evidence** | Explanation of why this result matches the query, with supporting quotes and labels |
| **Quality Assessment** | Readiness score, strengths, limitations, missing fields, QA state |
| **Evidence Tier** | Where each claim sits on a 6-rung ladder from metadata guess to human-validated fact |
| **Reusable Artifacts** | Dataset card, starter notebook, design reference, and a shareable ExperimentGlancer scene |

## Five Substrates

The system is best understood as five layers, each with a distinct trust model. This framing — not just a stack of features — is the current architectural mental model:

```
Canonical substrate      corpus JSONL, KG nodes/edges, papers, retraction status,
                         method registry, qrels
Retrieval substrate      BM25 + BGE-large dense embeddings + RRF fusion +
                         graph/concept scoring, ablation ladder gate
Understanding substrate  analysis affordances, evidence tiers, reanalysis/
                         reinterpretation candidates, neuro-judge triage
Bridge/viewer substrate  ExperimentGlancer: compiles a result into a versioned,
                         evidence-tiered scene JSON; a separate viewer renders it
Agent substrate          registry + ledger + playbooks that audit connectivity
                         and gate ranking changes, on a schedule
```

Canonical data is never inferred into existence by a higher layer — the retrieval, understanding, bridge, and agent substrates all read the canonical layer and write back evidence-tiered annotations, never silent overwrites.

## Design Principles

### 1. Experiment-Aware

Search understands the structure of neuroscience experiments:

- **Tasks**: Go/no-go, reversal learning, reaching, motor imagery, speech production
- **Behaviors**: Choice, reward, reaction time, kinematics, pupil, seizure onset
- **Modalities**: Neuropixels, ECoG, calcium imaging, EEG, fiber photometry
- **Regions**: mPFC, VTA, motor cortex, hippocampus, STG
- **Species**: Mouse, rat, macaque, human
- **Analyses**: Q-learning modeling, choice decoding, seizure detection, sleep staging

The ontology and knowledge graph are not just vocabulary — they power matching, ranking, filtering, explanation, and now scene compilation for ExperimentGlancer.

### 2. Provenance-First

Every claim needs evidence, and every claim sits at an explicit evidence tier:

1. `heuristic_candidate` — metadata/profile match only, no verification.
2. `evidence_backed_bridge` — inferred via a similar dataset's linked paper.
3. `source_declared` — the dataset or its paper explicitly declares the field (e.g. a DataCite `relatedIdentifiers` relation).
4. `file_validated` — a live, zero-download header inspection of the actual NWB/BIDS file confirmed it.
5. `human_validated` — an expert reviewed and accepted it.
6. `computed` — the analysis was actually run and passed QC.

Only the last three should read as trustworthy. This is not aspirational — it is an enforced enum (`neural_search/kg/schemas/evidence_tier.py`), checked by tests, and now surfaced to end users: a dataset card or search result whose linked paper was later retracted shows a warning, and a reanalysis suggestion's evidence tier is visible, not just internal graph metadata.

### 3. Analysis-Ready

Results should be actionable:

- **Analysis Affordances**: Can this dataset support Q-learning modeling? Does it have the required signals?
- **Readiness Scoring**: Not all datasets are equally usable — score and explain quality.
- **Starter Notebooks**: Generate code that loads the data and scaffolds the analysis.
- **ExperimentGlancer Scenes**: A synchronized, evidence-tiered timeline view of the specific trial/event/neural/behavior layers a result is relevant to — shareable by URL, never claiming a layer is real when only its metadata was inferred.

### 4. Evaluation-Visible

The system should know when it succeeds or fails:

- **317-query canonical benchmark** (`data/eval/benchmark_queries_canonical.yaml`) with hybrid sparse/dense/graph/concept ranking, evaluated via the ablation ladder (`scripts/eval/run_ablation_ladder.py`).
- **NDCG@10 gate on every KG change.** This is not a suggestion — this project has caught three real ranking regressions this way (two from a new edge type dominating a generic graph-connectivity feature, one from a score weight that had been configured speculatively years before the data it scored existed), and the gate now runs automatically via a scheduled agent (below).
- **Coverage reports**: which domains, regions, and modalities are well-covered, and where the honest gaps are.

### 5. Human-Reviewable

Automated extraction is fallible:

- **QA Workflow**: `unreviewed → auto_generated → needs_review → reviewed → trusted/rejected`.
- **Obsidian vault**: a human-readable, human-editable memory layer over the canonical artifacts — never the database of record itself.
- **Confidence Calibration**: system confidence should correlate with actual accuracy; gold/human-labeled qrels are still 0 rows, an explicitly tracked gap rather than a silently assumed non-issue.
- **Agent-authored audit trail**: an append-only ledger (`artifacts/agents/ledger.jsonl`) and linked Obsidian notes record what an automated audit found and what a human still needs to look at.

## What's Actually Built (not aspirational)

| Layer | State |
|---|---|
| Corpus | 7,171 real records / 625 unique datasets across DANDI, OpenNeuro, NeuroVault, Zenodo, Figshare, NeuroMorpho, Allen, GIN, and more |
| Retrieval | BM25 + BGE-large-en-v1.5 dense field embeddings + RRF fusion + graph/concept scoring; NDCG@10 = 0.8594 on the canonical 317-query benchmark |
| Knowledge graph | ~12,750 nodes / ~150,000 edges: datasets, papers, concepts, brain regions, tasks, methods, affordances, and typed relationship edges, all reachable from `scripts/build_real_corpus_graph.py` |
| Literature linking | 5 sources (OpenAlex, DataCite, Crossref, PubMed/bioRxiv, Semantic Scholar), ~35% combined real paper-dataset link coverage, plus retraction/correction checking |
| File validation | Live, zero-download DANDI (NWB header streaming) and OpenNeuro (BIDS) validators — real evidence, not a claim from metadata alone |
| ExperimentGlancer | A scene-compiler bridge and glassmorphic timeline viewer that turns a search result into a synchronized, evidence-tiered, shareable multimodal scene |
| Agent orchestration | A small registry/ledger/playbook scaffold that runs the connectivity-audit and benchmark-gate discipline on a weekly schedule, unattended |

## Analysis Affordance Framework

Neural Search doesn't just find datasets — it tells you what you can do with them:

| Affordance | Required Signals | Example Use Case |
|------------|------------------|------------------|
| `q_learning` | Choice, reward, trial structure | Fit RL models to behavioral data |
| `event_aligned_psth` | Spike times, event timestamps, neural data | Peri-stimulus time histograms |
| `choice_decoding` | Neural data, choice labels | Decode decisions from population activity |
| `motor_decoding` | Neural data, kinematics | Build movement decoders |
| `speech_decoding` | Neural data, phoneme labels | Speech neuroprosthesis development |
| `seizure_detection` | Neural data, seizure annotations | Clinical seizure detector training |
| `sleep_stage_classification` | EEG, sleep annotations | Automated sleep staging |
| `latent_dynamics_modeling` | Multi-unit data, trial structure | Latent population dynamics analysis |
| `functional_connectivity` | Multi-region data | Network analysis |
| `pose_neural_correlation` | Behavior video + neural data | Pose/video-neural joint analysis |

## What Happens After a Match: ExperimentGlancer

The core interaction ExperimentGlancer is built for is not "show me neuron 542" — it's "show me the moment the animal switched strategy," or "open session X, 2.3 seconds before lick onset, neurons sorted by ramping activity." A search result compiles directly into a scene: Neural Search decides which layers (trials, events, spikes, calcium, pose, model output) are plausible given the query, dataset structure, and any live file validation; the scene JSON records exactly which of those layers are `available` (file-derived), `probable` (metadata-inferred), or a `placeholder` warning (requested but unsupported) — and a separate viewer renders it as a synchronized, shareable timeline. Neural Search never fakes a timestamp or a signal it hasn't verified.

## Provenance Model

Every dataset record maintains a traceable provenance chain — source archive ID, extraction method and confidence, evidence for each label, linked papers with their own evidence tier and retraction status, and QA state. Users should be able to trace any assertion back to its source and its tier.

## Relationship to Existing Tools

| Tool | Focus | Neural Search Difference |
|------|-------|---------------------------|
| **DANDI/OpenNeuro** | Archive access | Adds semantic search, cross-archive, analysis affordances, live file validation |
| **PubMed/OpenAlex/Crossref/DataCite/Semantic Scholar** | Paper search | Links 5 sources' papers to data with an explicit evidence tier per link, plus retraction checking |
| **Google Dataset Search** | General datasets | Neuroscience-specific ontology, knowledge graph, and analysis affordances |
| **ChatGPT/Claude** | General Q&A | Grounded in a verified, evidence-tiered corpus, not hallucinated |
| **Neuroglancer** | 3D volumetric viewer | ExperimentGlancer slices through *experiment time*, not a spatial volume — synchronized trial/event/neural/behavior tracks |

## Future Directions

- **Gold qrels.** Every benchmark number in this project is currently measured against silver/LLM-judged labels, not human ones. This is the single biggest credibility gap and the highest-priority next investment.
- **File validation at corpus scale**, not just the top-N reanalysis suggestions — the validator is proven cheap (seconds, kilobytes); scaling it up converts `probable` claims into genuinely `available` ones across the corpus.
- **Multi-modal retrieval**: "find datasets with neural activity similar to this firing pattern," grounded in real extracted signatures rather than metadata alone.
- **A fuller agentic loop**: the current registry/ledger/playbook scaffold proves the pattern (connectivity audits, benchmark gating) works unattended; extending it to reanalysis scouting and claim synthesis, each gated by the same evidence-tier discipline, is the natural next step.

## Design Philosophy

**Be useful, be honest, be verifiable.**

- **Useful**: Help researchers find data they can actually use, and jump straight to the relevant moment in it.
- **Honest**: Report confidence, flag limitations, show evidence, never claim a layer or a fact is more verified than it is.
- **Verifiable**: Every claim traceable to source and to an explicit evidence tier.

Neural Search succeeds when a researcher can go from "I need data for X" to "here's a scene showing exactly the trial and signal I need, and a notebook analyzing it" — with confidence that the match is real, the analysis is appropriate, and every claim along the way is honestly labeled.
