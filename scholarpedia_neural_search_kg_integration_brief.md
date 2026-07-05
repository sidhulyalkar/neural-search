# Scholarpedia Integration Brief for Neural Search / Neuroscience KG

**Prepared for:** Claude Code / Neural Search development workflow  
**Prepared on:** 2026-06-29  
**Purpose:** Evaluate and implement a safe, useful integration of Scholarpedia into the Neural Search knowledge graph, Obsidian vault, and latent usefulness retrieval system.

---

## 1. Executive Summary

Scholarpedia should be integrated into Neural Search, but not as a raw text dump. Its best role is as a **curated conceptual authority layer** for neuroscience, computational neuroscience, dynamical systems, modeling, signal processing, and adjacent scientific concepts.

The core idea:

> Scholarpedia should provide the conceptual skeleton. OpenAlex, PubMed, arXiv, Semantic Scholar, and full-text papers provide the living literature. DANDI, OpenNeuro, Allen, EBRAINS, NeuroVault, Zenodo, GIN, PhysioNet, and other sources provide the empirical dataset layer. Obsidian provides the human-facing thinking cockpit. The Neural Search KG binds them together.

This integration can improve:

- Query expansion
- Concept disambiguation
- Knowledge graph schema richness
- Dataset-method matching
- Theory-to-dataset retrieval
- Analysis-pipeline recommendation
- Obsidian concept maps
- Field-state summaries
- Latent future usefulness ranking

Do **not** bulk-copy Scholarpedia article text into the repo by default. The safest path is to store metadata, URLs, article IDs, aliases, citations, extracted triples, short paraphrased summaries, and source pointers. Treat full article text as an optional local cache behind an explicit license/compliance flag.

---

## 2. Source Context and Caveats

Scholarpedia describes itself as a peer-reviewed open-access encyclopedia curated by expert communities. It is especially useful for stable scientific concepts rather than fast-moving frontier claims.

Relevant source facts verified on 2026-06-29:

- Scholarpedia presents itself as a peer-reviewed open-access encyclopedia curated by communities of experts.
- DOAJ lists Scholarpedia with ISSN `1941-6016`.
- DOAJ lists Scholarpedia as using a `CC BY-NC-SA` license.
- DOAJ lists Scholarpedia keywords including physics, neuroscience, and computation.
- Creative Commons BY-NC-SA requires attribution, restricts commercial use, and requires adaptations to be shared under the same license family.

Recommended compliance posture:

1. Store article metadata, URLs, titles, authors, revision IDs where available, and citations.
2. Store short human-written or model-generated paraphrased summaries, not full article text, unless the project is clearly non-commercial and compatible with BY-NC-SA obligations.
3. Store extracted relation triples only when they are sufficiently abstracted or derived as factual metadata/conceptual links.
4. Preserve attribution links back to the original Scholarpedia article.
5. Keep license fields explicit on every source-derived record.
6. Add a `license_policy` field and downstream checks so Scholarpedia-derived content is not accidentally mixed into commercial or incompatible exports.

Suggested source references:

- Scholarpedia homepage: `https://www.scholarpedia.org/`
- Scholarpedia Neuroscience page: `http://www.scholarpedia.org/article/Neuroscience`
- Scholarpedia FAQ: `https://www.scholarpedia.org/article/Help:Frequently_Asked_Questions`
- Scholarpedia copyright info: `https://www.scholarpedia.org/article/Help:Copyright_information`
- DOAJ Scholarpedia record: `https://doaj.org/toc/1941-6016`
- Creative Commons BY-NC-SA 3.0 deed: `https://creativecommons.org/licenses/by-nc-sa/3.0/deed.en`
- Creative Commons BY-NC-SA 4.0 deed: `https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en`

---

## 3. Strategic Role in Neural Search

Scholarpedia is useful because Neural Search is not merely retrieving papers or datasets by surface similarity. The deeper research question is:

> How do we retrieve objects by latent future usefulness rather than by keyword overlap, citation count, or shallow semantic similarity?

Scholarpedia helps because it provides high-quality conceptual anchors. These anchors can connect theories, methods, equations, brain systems, modalities, datasets, and analysis pipelines.

### Best uses

1. **Concept authority layer**  
   Stable definitions and topic boundaries for neuroscience concepts.

2. **Ontology seed**  
   Initial graph hierarchy for concepts such as neural coding, dynamical systems, synaptic plasticity, recurrent networks, predictive coding, sparse coding, neural oscillations, criticality, and attractor dynamics.

3. **Query expansion engine**  
   Expand underspecified scientific queries into related methods, concepts, regions, modalities, and measurable signatures.

4. **Concept disambiguation layer**  
   Distinguish overloaded terms such as coding, plasticity, oscillation, field, phase, model, circuit, manifold, representation, and dynamics.

5. **Dataset-method bridge**  
   Link concepts to required observables and feasible analyses.

6. **Obsidian note generator**  
   Produce concise concept cards for human browsing and graph visualization.

7. **KG sanity-check source**  
   Use expert-curated concept relations to test whether KG edges are plausible.

### Weak uses

Scholarpedia is less useful as:

1. A frontier literature tracker.
2. A comprehensive dataset catalog.
3. A source of recent empirical results.
4. A source to copy wholesale into the repo.
5. A replacement for OpenAlex, PubMed, arXiv, Semantic Scholar, or dataset registries.

---

## 4. Proposed Architecture

```text
Scholarpedia
  ↓
Scholarpedia article registry
  ↓
Concept extraction + metadata normalization
  ↓
Concept authority records
  ↓
KG edge generation
  ↓
Obsidian concept notes
  ↓
Query expansion + retrieval priors
  ↓
Latent usefulness scoring
```

### Recommended module layout

```text
neural_search/
  ingestion/
    scholarpedia/
      __init__.py
      fetch.py
      parse.py
      normalize.py
      extract_concepts.py
      build_edges.py
      obsidian_export.py
      license_policy.py
      cli.py
  kg/
    schemas/
      concept_authority.py
      scholarpedia.py
  retrieval/
    query_expansion/
      scholarpedia_expander.py
  eval/
    scholarpedia_kg_tests.py
```

### Data outputs

```text
data/concepts/scholarpedia_articles.jsonl
data/concepts/scholarpedia_concepts.jsonl
data/kg/scholarpedia_edges.jsonl
data/obsidian/generated/scholarpedia/*.md
reports/scholarpedia_ingestion_report.md
reports/scholarpedia_license_report.md
```

---

## 5. Suggested Schema

### 5.1 ScholarpediaArticle

```json
{
  "id": "scholarpedia:neuroscience",
  "source": "scholarpedia",
  "title": "Neuroscience",
  "url": "http://www.scholarpedia.org/article/Neuroscience",
  "article_type": "expert_reference",
  "license": "CC BY-NC-SA",
  "license_policy": "noncommercial_sharealike_attribution_required",
  "retrieved_at": "2026-06-29",
  "authors": [],
  "curators": [],
  "revision_id": null,
  "doi": null,
  "canonical_citation": null,
  "summary_paraphrase": "Short project-authored summary here.",
  "concepts": [],
  "methods": [],
  "models": [],
  "modalities": [],
  "brain_regions": [],
  "analysis_affordances": [],
  "source_text_cached": false,
  "text_cache_path": null
}
```

### 5.2 ConceptAuthorityRecord

```json
{
  "id": "concept:sparse_coding",
  "label": "Sparse coding",
  "aliases": ["sparse representation", "efficient sparse code"],
  "definition_paraphrase": "A representation principle where information is encoded by relatively small active subsets of units.",
  "source_authorities": ["scholarpedia:sparse_coding"],
  "concept_type": "theory_or_principle",
  "field": "neuroscience",
  "subfields": ["neural coding", "computational neuroscience"],
  "related_methods": ["encoding model", "dictionary learning", "population analysis"],
  "related_modalities": ["electrophysiology", "calcium imaging"],
  "related_regions": ["visual cortex", "auditory cortex"],
  "testable_predictions": [],
  "dataset_requirements": [],
  "kg_confidence": 0.8,
  "freshness_class": "stable_reference",
  "license": "CC BY-NC-SA",
  "provenance": {
    "source": "scholarpedia",
    "url": "https://www.scholarpedia.org/article/Sparse_coding",
    "retrieved_at": "2026-06-29"
  }
}
```

### 5.3 KG Edge Record

```json
{
  "source_id": "concept:sparse_coding",
  "target_id": "concept:neural_coding",
  "edge_type": "subtopic_of",
  "evidence_source": "scholarpedia:sparse_coding",
  "evidence_mode": "expert_reference_derived",
  "confidence": 0.78,
  "license": "CC BY-NC-SA",
  "created_at": "2026-06-29"
}
```

---

## 6. Node and Edge Types

### Recommended node types

```text
ScholarpediaArticle
Concept
Theory
Principle
Method
Model
Equation
SignalFeature
BrainRegion
CellType
Circuit
CognitiveProcess
MeasurementModality
AnalysisPipeline
DatasetAffordance
Author
Reference
Assumption
OpenQuestion
```

### Recommended edge types

```text
ARTICLE defines CONCEPT
ARTICLE discusses METHOD
ARTICLE cites PAPER
CONCEPT alias_of CONCEPT
CONCEPT narrower_than CONCEPT
CONCEPT broader_than CONCEPT
CONCEPT related_to CONCEPT
CONCEPT prerequisite_for METHOD
CONCEPT motivates METHOD
METHOD analyzes MODALITY
METHOD estimates SIGNAL_FEATURE
MODEL formalizes PROCESS
MODEL assumes ASSUMPTION
THEORY predicts SIGNAL_FEATURE
CONCEPT testable_with DATASET_AFFORDANCE
DATASET affords METHOD
DATASET measures MODALITY
DATASET relevant_to CONCEPT
BRAIN_REGION participates_in PROCESS
PIPELINE implements METHOD
```

---

## 7. Query Expansion Examples

### Example 1: Predictive coding

User query:

```text
Find datasets for testing predictive coding in sensory cortex.
```

Scholarpedia/KG expansion:

```text
predictive coding
hierarchical inference
generative model
prediction error
Bayesian brain
surprise
mismatch response
visual cortex
auditory cortex
laminar electrophysiology
calcium imaging
stimulus omission
oddball paradigm
encoding model
representational similarity
```

Retrieval priors:

```text
+ sensory cortex region match
+ repeated or structured stimuli
+ population recordings
+ stimulus timing metadata
+ trial-level behavioral/stimulus annotations
+ laminar or layer annotations if available
+ analysis pipeline feasibility
- missing stimulus metadata
- insufficient temporal resolution for prediction-error signatures
```

### Example 2: Sparse coding

User query:

```text
Which datasets can help test sparse coding in natural vision?
```

Expansion:

```text
sparse coding
efficient coding
natural image statistics
population coding
receptive fields
V1
visual cortex
natural scene stimuli
calcium imaging
electrophysiology
encoding models
population activity
```

Dataset affordances:

```text
visual stimulus metadata
population-level recordings
natural image/movie stimuli
single-trial responses
cell-level activity matrix
sufficient number of neurons/ROIs
```

### Example 3: Criticality / aperiodic activity

User query:

```text
Find datasets that could be reanalyzed for aperiodic neural dynamics and criticality.
```

Expansion:

```text
criticality
scale-free dynamics
power-law activity
1/f noise
aperiodic exponent
neuronal avalanches
oscillations
E/I balance
population dynamics
resting state
LFP
ECoG
EEG
MEG
Neuropixels
widefield calcium
```

Dataset affordances:

```text
raw or minimally filtered time series
high temporal resolution
long continuous recordings
resting-state or spontaneous activity epochs
population-level activity
metadata for state, behavior, sleep/wake, anesthesia
```

---

## 8. Obsidian Integration Strategy

Do not mirror full articles. Generate compact concept cards.

### Suggested note path

```text
Obsidian Vault/
  Neural Search KG/
    Concepts/
      Scholarpedia/
        Neuroscience.md
        Sparse coding.md
        Predictive coding.md
        Dynamic causal modeling.md
        Hodgkin-Huxley model.md
        Criticality.md
```

### Suggested note template

```md
---
type: concept_reference
source: scholarpedia
source_url: "https://www.scholarpedia.org/article/<ARTICLE>"
license: "CC BY-NC-SA"
status: curated_reference
freshness: stable_not_frontier
kg_node_type: Concept
aliases: []
related_modalities: []
related_methods: []
related_datasets: []
created_by: neural_search_generated
---

# <Concept Name>

## Canonical paraphrase

Short project-authored summary. Avoid copying long source passages.

## Why this matters for Neural Search

- Query expansion
- Dataset-method matching
- KG concept hierarchy
- Analysis affordance inference

## Key related concepts

- [[Concept A]]
- [[Concept B]]
- [[Concept C]]

## Relevant modalities

- Electrophysiology
- Calcium imaging
- fMRI
- EEG/MEG

## Relevant analysis methods

- Encoding models
- Decoding models
- Dynamical systems analysis
- Spectral analysis

## Dataset affordances needed

- Raw or minimally processed signal
- Trial-level metadata
- Stimulus annotations
- Region/cell-type annotations

## KG edges

```text
<Concept> broader_than <Subconcept>
<Concept> motivates <Method>
<Concept> testable_with <DatasetAffordance>
```

## Source

Scholarpedia: <URL>
```

### Obsidian goal

Obsidian should become a browsable conceptual atlas, not a warehouse. It should show:

- concept neighborhoods,
- method prerequisites,
- dataset affordances,
- links from theories to reusable datasets,
- unresolved questions,
- possible reanalysis opportunities.

---

## 9. Integration With Latent Future Usefulness

Scholarpedia-derived concepts can become priors in the latent usefulness ranking function.

Current or proposed ranking signals may include:

```text
lexical match
semantic embedding similarity
concept coverage
ontology proximity
hypothesis compatibility
analysis feasibility
provenance trust
modality alignment
experimental design similarity
statistical power proxy
metadata completeness
pipeline availability
cross-dataset comparability
novelty/gap value
negative constraints / hard negatives
```

Scholarpedia helps mostly with:

```text
concept coverage
ontology proximity
hypothesis compatibility
method prerequisites
query expansion
concept disambiguation
analysis affordance inference
```

### Example scoring feature

```python
scholarpedia_concept_overlap = weighted_overlap(
    query_expanded_concepts,
    dataset_attached_concepts,
    weights={
        "direct_concept": 1.0,
        "one_hop_related": 0.6,
        "two_hop_related": 0.3,
        "same_subfield": 0.2,
    },
)
```

### Example retrieval explanation

```text
This dataset ranks highly because it measures population activity in visual cortex during natural image stimulation, which matches the sparse-coding concept neighborhood: neural coding, efficient coding, receptive fields, sensory population responses, and encoding-model feasibility.
```

This is important because Neural Search should explain *why* a dataset is useful, not simply return it.

---

## 10. Implementation Plan for Claude Code

### Phase 0: Repository inspection

Before modifying code:

1. Inspect existing ingestion adapters.
2. Inspect current KG schema and graph builder.
3. Inspect Obsidian export utilities, if present.
4. Inspect retrieval/query expansion modules.
5. Inspect tests for corpus normalization and KG edge generation.
6. Identify the existing style for JSONL artifacts and reports.

Expected output:

```text
reports/scholarpedia_integration_repo_assessment.md
```

Include:

- where this integration fits,
- which existing interfaces should be reused,
- schema compatibility concerns,
- implementation risks,
- test plan.

### Phase 1: Add data models

Add typed models for:

- `ScholarpediaArticle`
- `ConceptAuthorityRecord`
- `ScholarpediaKGEdge`
- `LicensePolicy`

Suggested locations:

```text
neural_search/kg/schemas/scholarpedia.py
neural_search/kg/schemas/concept_authority.py
```

Requirements:

- Use existing project style, dataclasses or Pydantic depending on repo conventions.
- Include source URL, retrieval timestamp, license, provenance, confidence.
- Include field for `source_text_cached: bool`.
- Include field for `commercial_use_allowed: false` or equivalent.

### Phase 2: Build article registry

Create a seed registry of high-value Scholarpedia article URLs.

Initial seed categories:

```text
core neuroscience
computational neuroscience
neural coding
dynamical systems
neural oscillations
plasticity
brain rhythms
criticality
spiking neuron models
connectivity
causal modeling
Bayesian neuroscience
learning rules
neural field models
signal processing concepts relevant to neural data
```

Start with a small manually curated list of 20-50 URLs. Do not scrape the whole site on first pass.

Output:

```text
data/concepts/scholarpedia_seed_articles.jsonl
```

### Phase 3: Fetch and parse metadata

Create fetcher/parser that extracts:

- title,
- URL,
- suggested citation if available,
- authors if available,
- curators if available,
- DOI if available,
- category links if available,
- outbound article links,
- references if available,
- revision/citation metadata if available,
- page text only if explicitly enabled.

Important:

- Respect robots, rate limits, and site load.
- Use polite delay.
- Default to metadata-first mode.
- Make full text caching opt-in.

CLI sketch:

```bash
python -m neural_search.ingestion.scholarpedia.cli fetch \
  --seed data/concepts/scholarpedia_seed_articles.jsonl \
  --out data/concepts/scholarpedia_articles.jsonl \
  --metadata-only \
  --respect-license
```

### Phase 4: Concept extraction

Build a conservative extraction pipeline:

1. Extract article title as canonical concept candidate.
2. Extract aliases from redirects, first paragraph, bold terms, and headings when available.
3. Extract related concepts from internal links.
4. Extract methods/models/modalities/regions using existing dictionaries if present.
5. Optionally run LLM-assisted extraction, but mark as weakly supervised.
6. Keep confidence scores.

Output:

```text
data/concepts/scholarpedia_concepts.jsonl
```

### Phase 5: KG edge generation

Generate conservative first-pass edges:

```text
ARTICLE defines CONCEPT
ARTICLE links_to ARTICLE
CONCEPT related_to CONCEPT
CONCEPT broader_than / narrower_than CONCEPT when reliable
CONCEPT motivates METHOD when high-confidence
METHOD analyzes MODALITY when high-confidence
CONCEPT testable_with DATASET_AFFORDANCE when high-confidence
```

Avoid overclaiming. If an edge is inferred from weak evidence, mark it accordingly:

```json
{
  "evidence_mode": "weakly_inferred",
  "requires_human_review": true
}
```

Output:

```text
data/kg/scholarpedia_edges.jsonl
```

### Phase 6: Obsidian export

Generate compact concept notes from `ConceptAuthorityRecord`.

CLI sketch:

```bash
python -m neural_search.ingestion.scholarpedia.cli export-obsidian \
  --concepts data/concepts/scholarpedia_concepts.jsonl \
  --edges data/kg/scholarpedia_edges.jsonl \
  --out data/obsidian/generated/scholarpedia
```

Requirements:

- Use YAML front matter.
- Include source URL and license.
- Use paraphrased summaries.
- Include KG links.
- Do not include full source article text by default.

### Phase 7: Retrieval integration

Add Scholarpedia-derived query expansion.

Possible file:

```text
neural_search/retrieval/query_expansion/scholarpedia_expander.py
```

Basic algorithm:

1. Detect query concepts using aliases and embeddings.
2. Expand with one-hop related concepts.
3. Add method/modalities/regions only when confidence is high.
4. Return expansion terms with weights.
5. Feed expansion terms into hybrid retrieval and KG scoring.

Example expansion object:

```json
{
  "query": "datasets for predictive coding in sensory cortex",
  "matched_concepts": ["concept:predictive_coding"],
  "expansion_terms": [
    {"term": "prediction error", "weight": 0.9, "source": "scholarpedia"},
    {"term": "hierarchical inference", "weight": 0.7, "source": "scholarpedia"},
    {"term": "sensory cortex", "weight": 0.8, "source": "query"}
  ]
}
```

### Phase 8: Evaluation

Add tests and evaluation artifacts.

Unit tests:

```text
tests/ingestion/test_scholarpedia_models.py
tests/ingestion/test_scholarpedia_parser.py
tests/kg/test_scholarpedia_edges.py
tests/retrieval/test_scholarpedia_query_expansion.py
tests/obsidian/test_scholarpedia_export.py
```

Evaluation checks:

1. **Schema validation:** all records validate.
2. **License propagation:** every derived node/edge retains license/provenance.
3. **No full text by default:** parser/exporter does not store full article body unless explicit flag is set.
4. **Attribution present:** every Obsidian note includes source URL and license.
5. **Query expansion sanity:** expansions for 20 curated queries are plausible.
6. **Retrieval ablation:** compare retrieval with and without Scholarpedia expansion.
7. **Hard negative protection:** expansion should not cause broad concept drift.
8. **Human audit queue:** uncertain edges and expansions are queued for review.

Suggested metrics:

```text
precision@k on concept expansion
human plausibility score for generated edges
retrieval NDCG/MRR delta on adjudicated qrels
hard-negative violation rate
concept drift rate
license/provenance completeness
Obsidian note validity
```

---

## 11. Safety and Quality Guardrails

### License guardrail

Add a central policy check:

```python
if record.source == "scholarpedia":
    record.license = "CC BY-NC-SA"
    record.commercial_use_allowed = False
    record.requires_attribution = True
    record.sharealike_required = True
```

Downstream export modes:

```text
internal_noncommercial: allow Scholarpedia-derived summaries/triples with attribution
public_noncommercial: allow with attribution + compatible license note
commercial_or_unspecified: exclude Scholarpedia-derived adapted content unless legal review clears it
```

### Scientific guardrail

Scholarpedia is concept authority, not empirical evidence authority for modern claims.

Every derived record should include:

```json
{
  "freshness_class": "stable_reference",
  "frontier_claim": false,
  "requires_literature_update": true
}
```

### KG overclaiming guardrail

Use edge confidence and evidence mode:

```text
expert_reference_direct
expert_reference_linked
dictionary_matched
weakly_inferred
llm_inferred_requires_review
human_validated
```

Only `expert_reference_direct`, `expert_reference_linked`, and `human_validated` should influence retrieval strongly.

---

## 12. Suggested Initial Article Seed List

Start small. Candidate topics:

```text
Neuroscience
Computational neuroscience
Neural coding
Sparse coding
Predictive coding
Dynamic causal modeling
Hodgkin-Huxley model
Integrate-and-fire model
Spike-timing-dependent plasticity
Synaptic plasticity
Neural oscillation
Local field potential
Electroencephalography
Magnetoencephalography
Functional magnetic resonance imaging
Dynamical systems
Attractor network
Neural field
Criticality
Population coding
Receptive field
Bayesian inference
Reinforcement learning
Hebbian learning
Connectome
Graph theory
Causality
Information theory
```

For each topic, resolve the actual Scholarpedia URL if it exists. If no Scholarpedia article exists, do not invent one. Store missing articles as `unresolved_seed` records for future curation.

---

## 13. Claude Code Implementation Prompt

Copy/paste this to Claude Code from the root of the Neural Search repository:

```text
We want to integrate Scholarpedia as a curated concept authority layer for the Neural Search neuroscience knowledge graph, retrieval system, and Obsidian vault.

Please use your repository-analysis / brainstorming workflow first. Do not start coding immediately. Inspect the current repo structure, especially ingestion adapters, KG schemas/builders, concept memory, retrieval/query expansion, Obsidian export utilities, artifact/report conventions, and tests.

Goal:
Add a conservative Scholarpedia integration that improves concept hierarchy, query expansion, dataset-method matching, KG edges, and Obsidian concept cards without bulk-copying Scholarpedia article text.

Important license/compliance constraints:
- Treat Scholarpedia-derived content as CC BY-NC-SA unless a page-specific license says otherwise.
- Store metadata, URLs, citations, derived triples, aliases, and short paraphrased summaries by default.
- Do not store/export full article text by default.
- Propagate source URL, license, retrieval timestamp, and provenance into every derived node, edge, report, and Obsidian note.
- Add an explicit license policy field so Scholarpedia-derived content can be excluded from commercial or incompatible exports.

Implementation tasks:
1. Write a repo assessment report at reports/scholarpedia_integration_repo_assessment.md explaining where this integration fits and what existing abstractions should be reused.
2. Add typed records for ScholarpediaArticle, ConceptAuthorityRecord, ScholarpediaKGEdge, and LicensePolicy using existing project conventions.
3. Add a small seed registry for high-value Scholarpedia neuroscience/computational neuroscience articles. Start with 20-50 candidate topics, but resolve only actual URLs.
4. Build a metadata-first fetch/parser module under neural_search/ingestion/scholarpedia/. It should extract title, URL, authors/curators/citation/DOI/revision if available, internal links, categories/headings if available, and references if available. Full text caching must be opt-in only.
5. Build a conservative concept extraction module. Extract canonical concepts, aliases, related concepts, methods, models, modalities, regions, and dataset affordance hints with confidence scores and evidence modes.
6. Build KG edge generation for ARTICLE-defines-CONCEPT, CONCEPT-related_to-CONCEPT, broader/narrower links when reliable, method/modality links when high-confidence, and concept-testable_with-dataset_affordance links when high-confidence.
7. Add Obsidian export for compact concept cards with YAML front matter, paraphrased summaries, source URL, license, related concepts, relevant methods/modalities, dataset affordances, and KG edges. No full source text in notes.
8. Add a Scholarpedia query expander that maps query terms to concept records and returns weighted expansion terms for retrieval/KG scoring.
9. Add tests for schema validation, license propagation, parser behavior, no-full-text-by-default behavior, KG edge generation, Obsidian export, and query expansion.
10. Add a small evaluation report comparing 20 curated queries with and without Scholarpedia expansion. Include expansion precision, concept drift examples, and whether hard-negative violations increased.

Design principles:
- Be conservative. Do not overclaim edges.
- Every derived fact needs provenance.
- Weak inferred edges should be marked requires_human_review=true.
- Scholarpedia should be used as a stable concept authority, not as a frontier literature source.
- Retrieval impact should be ablated and reported before claims are made.

Expected artifacts:
- data/concepts/scholarpedia_seed_articles.jsonl
- data/concepts/scholarpedia_articles.jsonl
- data/concepts/scholarpedia_concepts.jsonl
- data/kg/scholarpedia_edges.jsonl
- data/obsidian/generated/scholarpedia/*.md
- reports/scholarpedia_integration_repo_assessment.md
- reports/scholarpedia_ingestion_report.md
- reports/scholarpedia_query_expansion_eval.md
- reports/scholarpedia_license_report.md
- tests covering the new modules

Before finishing:
- Run formatting/linting/type checks according to repo conventions.
- Run all new tests and relevant existing tests.
- Provide a final summary of files changed, commands run, tests passed/failed, and remaining risks.
```

---

## 14. Recommended Definition of Done

This integration should be considered complete only when:

1. A small seed article set is ingested metadata-first.
2. Every generated record has source URL, license, and provenance.
3. Full article text is not stored by default.
4. Obsidian notes are generated and readable.
5. KG edges are generated with confidence and evidence mode.
6. Query expansion works on a curated query set.
7. Retrieval ablation shows whether the expansion helps or hurts.
8. Hard-negative drift is measured.
9. License report is generated.
10. Human review queue exists for uncertain edges.

---

## 15. Final Recommendation

Yes, integrate Scholarpedia, but as a **conceptual navigation system**, not a content warehouse.

The strongest version of Neural Search needs a way to distinguish:

```text
what a concept means
what evidence exists for it
which datasets can test it
which analyses are feasible
which reanalyses are promising
which claims are established versus speculative
```

Scholarpedia can help with the first layer and partially with the second-order map between concepts. The rest should come from the live literature graph, dataset registries, benchmark evaluations, human audits, and retrieval ablations.

In the long run, Scholarpedia-derived concept structure can become one of the cleanest bridges between neuroscience theory and reusable data. It can help Neural Search answer not just:

```text
What dataset matches this query?
```

but the deeper question:

```text
What existing data could become newly useful if viewed through this conceptual frame?
```

That is the part worth building.
