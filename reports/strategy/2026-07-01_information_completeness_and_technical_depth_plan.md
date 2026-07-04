# Information Completeness and Technical Depth Plan

**Date:** 2026-07-01  
**Purpose:** Reframe Neural Search around completion of the information substrate: references, methodologies, processes, evidence, datasets, knowledge graph structure, and active understanding.

## 1. Reframed Goal

The goal is not only to ship a useful search product. The deeper goal is to build a complete, navigable, evidence-bearing map of neuroscience research objects and the conceptual machinery needed to use them.

Target state:

> Neural Search should know what the field is talking about, what methods exist, what data can support them, what evidence links claims to datasets and papers, what processes produce the artifacts, what is missing, and how confidence changes as new information arrives.

In practical terms, this means the platform needs a governed information set with:

- Reference authorities: stable sources for concepts, methods, theories, regions, organisms, modalities, disorders, and standards.
- Methodology maps: analysis pipelines, prerequisites, required signals, assumptions, outputs, failure modes, and validation criteria.
- Process maps: ingestion, extraction, normalization, KG construction, retrieval, evaluation, audit, and release workflows.
- Evidence maps: every claim, edge, affordance, and result connected to evidence and provenance.
- Coverage maps: what is known, unknown, inferred, reviewed, stale, or contradictory.
- Active memory: a way to preserve, update, and explain current understanding without flattening uncertainty.

## 2. Current Position

### 2.1 Strong Existing Foundations

The repository already contains unusually broad infrastructure:

- Data sources: DANDI, OpenNeuro, NeuroVault, Zenodo, OSF, GIN, Figshare, EBRAINS, Allen, IBL, CRCNS, BlueBrain, NeuroMorpho, PhysioNet, Harvard Dataverse, Buzsaki, Spark, NEMO, HCP, Brain Image Library.
- Domain registries: methods, modalities, oscillations, paradigms, species homology, HCP priors, notebook templates, signatures.
- Graph stack: schema, provenance, graph builder, typed KG features, semantic edges, similarity, transitive edges, temporal edges, metapaths, quality reports.
- Literature stack: OpenAlex ingestion, finding extraction, typed findings, claim synthesis, relationship builder, relationship KG builder, evidence spans.
- Retrieval stack: query intent, constraint parser, usefulness scorer, graph usefulness, set coverage scorer, dataset context bridge.
- Evaluation stack: qrels, hard negatives, ablation ladders, bootstrap CI, failure analysis, source coverage, human audit templates.
- Field-state/concept memory: concept basis, graph builder, retrieval, reranking, evidence, validation, Obsidian export.
- Frontend/API surface: search, results, dataset pages, ontology, graph, coverage, atlas, methods, disorders, evaluation, reports.

### 2.2 Current Scale Signals

From the current artifact manifest:

- Full corpus: 7,171 rows.
- Unique source IDs: 7,121.
- Full graph: 7,593 nodes and 31,920 edges.
- Cross-dataset edges: 2,957.
- Canonical LLM-silver qrels: 13,654 pairs across 317 queries.
- Gold qrels: 0.
- Field-state adjudicated qrels: 3.
- OpenAlex papers ingested: 255,940.
- Operational findings estimate: 190,279.
- Vector index coverage: 625 on-disk IDs, full 7,171-record rebuild pending.

### 2.3 Critical Missing Piece

The system has many components, but it does not yet have a formal completeness ledger for the whole knowledge substrate. Coverage exists for dataset metadata dimensions, but not yet for:

- Concept authority coverage.
- Methodology coverage.
- Process/workflow coverage.
- Evidence provenance coverage by object type.
- Assumption and limitation coverage.
- Dataset-to-method feasibility coverage.
- Reference-to-concept-to-dataset bridge coverage.
- Cross-source conflict and contradiction coverage.

This is the next technical depth layer.

## 3. Completeness Model

Completeness should be defined across eight layers. Each layer should have schemas, artifacts, reports, and tests.

### Layer 1: Source and Reference Authority

Question:

> What sources define or anchor the concepts, methods, entities, standards, and claims we rely on?

Objects:

- Encyclopedia/reference articles.
- Ontology terms.
- Standards documents.
- Archive schemas.
- Dataset registry records.
- Paper records.
- Method papers.
- Software/tool documentation.
- Protocol documents.

High-value sources:

- Scholarpedia for stable conceptual anchors.
- Cognitive Atlas for cognitive/task concepts.
- Allen and UBERON for brain region grounding.
- NWB and BIDS specs for data standards.
- OpenMINDS where useful for neuroscience metadata.
- DANDI/OpenNeuro/EBRAINS/etc. for source records.
- OpenAlex/PubMed/Semantic Scholar/arXiv for literature.
- Method/tool docs for analysis pipelines.

Required records:

- `ReferenceAuthorityRecord`
- `SourceDocumentRecord`
- `StandardRecord`
- `OntologyCrosswalkRecord`
- `LicensePolicyRecord`

Completion metrics:

- Percent of core concepts with at least one authority source.
- Percent of methods with at least one method paper or documentation source.
- Percent of standards with a normalized schema reference.
- Percent of authority-derived objects with license and provenance.

### Layer 2: Concept and Theory Map

Question:

> What does each concept mean, what are its aliases, and how does it relate to other concepts?

Objects:

- Concepts.
- Theories.
- Principles.
- Models.
- Equations.
- Assumptions.
- Predictions.
- Open questions.

Examples:

- Predictive coding.
- Sparse coding.
- Criticality.
- Neural oscillations.
- Attractor dynamics.
- Reinforcement learning.
- Dynamic causal modeling.
- Population coding.
- Synaptic plasticity.
- Aperiodic activity.

Required relations:

- broader_than / narrower_than
- alias_of
- related_to
- motivates_method
- predicts_signal_feature
- assumes
- contradicted_by
- testable_with_affordance

Completion metrics:

- Core concept coverage by subfield.
- Alias coverage.
- Concept-to-method edge coverage.
- Concept-to-dataset-affordance edge coverage.
- Concepts with unresolved ambiguity.

### Layer 3: Methodology and Pipeline Map

Question:

> What methods exist, what inputs do they require, what outputs do they produce, and what datasets can support them?

Objects:

- Analysis methods.
- Processing pipelines.
- Statistical models.
- ML models.
- Reanalysis recipes.
- Notebook templates.
- Software packages.
- Validation protocols.
- Failure modes.

Examples:

- Q-learning model fitting.
- Choice decoding.
- PSTH/event-aligned analysis.
- Encoding models.
- Functional connectivity.
- State-space latent dynamics.
- Spectral/aperiodic analysis.
- Sleep staging.
- Seizure detection.
- Speech decoding.
- Dynamic causal modeling.

Required fields:

- Method label and aliases.
- Required data types.
- Required metadata.
- Required sampling properties.
- Required task/event annotations.
- Compatible modalities.
- Compatible species/regions if relevant.
- Outputs.
- Assumptions.
- Contraindications.
- Minimal viable dataset conditions.
- Gold-standard validation datasets if known.
- Related notebooks and code.

Required relations:

- method_requires_signal
- method_requires_metadata
- method_analyzes_modality
- method_outputs_measure
- method_tests_concept
- dataset_affords_method
- pipeline_implements_method
- method_has_failure_mode

Completion metrics:

- Percent of affordances mapped to method records.
- Percent of methods with required signal lists.
- Percent of methods with contraindications.
- Percent of methods with at least one runnable notebook.
- Percent of datasets with at least one validated method affordance.

### Layer 4: Dataset and Empirical Object Map

Question:

> What empirical objects exist, what do they measure, and what can they support?

Objects:

- Datasets.
- Files/assets.
- Sessions.
- Subjects/specimens.
- Modalities.
- Signals.
- Events.
- Regions.
- Species.
- Standards.
- Linked papers.
- Source archive records.

Required coverage dimensions:

- Source ID and source URL.
- Species.
- Brain regions.
- Modality.
- Recording scale.
- Task.
- Behavioral events.
- Data standard.
- Access/license.
- Raw/processed availability.
- File-level evidence.
- Linked literature.
- Analysis affordances.

Completion metrics:

- Existing coverage ledger metrics.
- File-derived vs inferred ratio.
- Dataset-source URL completeness.
- Dataset-paper link precision.
- Behavioral-event coverage.
- Task coverage.
- Brain-region coverage.
- Method affordance coverage.

### Layer 5: Evidence and Claim Map

Question:

> What claims exist, where did they come from, what evidence supports them, and what contradicts them?

Objects:

- Findings.
- Claims.
- Evidence spans.
- Consensus summaries.
- Contradictions.
- Review judgments.
- Confidence scores.

Required fields:

- Claim/finding text.
- Source paper/document.
- Evidence span or quote pointer.
- Typed fields: region, species, modality, task, direction, signal type, frequency band, injury model, negation, statistical relation.
- Confidence.
- Extractor version.
- Review state.

Required relations:

- paper_reports_finding
- finding_supports_claim
- finding_contradicts_finding
- claim_contradicts_claim
- claim_supported_by_dataset
- dataset_can_test_claim
- finding_mentions_region/method/species/modality

Completion metrics:

- Percent of claims with evidence spans.
- Percent of findings with typed fields.
- Human audit precision by extractor/version.
- Contradiction review rate.
- Consensus summaries with sufficient paper count.
- Claim-to-dataset testability coverage.

### Layer 6: Process and Workflow Map

Question:

> How is knowledge produced, transformed, evaluated, updated, and released inside this platform?

Objects:

- Ingestion workflows.
- Normalization workflows.
- KG build workflows.
- Embedding/index build workflows.
- Evaluation workflows.
- Audit workflows.
- Release workflows.
- Human review workflows.
- Feedback loops.

Required process records:

- Inputs.
- Outputs.
- Commands.
- Dependencies.
- Runtime class.
- Artifact paths.
- Staleness conditions.
- Failure modes.
- Owner/reviewer.
- Tests.
- Downstream consumers.

Required relations:

- process_produces_artifact
- process_consumes_artifact
- process_validated_by_test
- artifact_supersedes_artifact
- artifact_depends_on_source
- report_summarizes_artifact

Completion metrics:

- Percent of generated artifacts with a declared producer process.
- Percent of processes with tests.
- Percent of reports with freshness metadata.
- Percent of stale reports explicitly marked stale.
- Rebuild reproducibility rate.

### Layer 7: Retrieval and Reasoning Map

Question:

> How does the system decide what is relevant, useful, trustworthy, or worth surfacing?

Objects:

- Query intents.
- Query constraints.
- Expansion terms.
- Ranking features.
- Reranking stages.
- KG features.
- Hard negatives.
- Explanations.
- Evaluation judgments.

Required fields:

- Feature name.
- Feature source.
- Feature evidence tier.
- Expected impact.
- Failure modes.
- Calibration status.
- Ablation status.

Required relations:

- query_matches_concept
- query_requires_affordance
- feature_uses_edge_type
- ranking_variant_uses_feature
- eval_report_measures_variant
- hard_negative_blocks_result

Completion metrics:

- Feature-level ablation coverage.
- Edge-family helpfulness by intent.
- Hard-negative violation rate.
- Query expansion drift rate.
- Explanation completeness for top results.
- Gold vs silver evaluation separation.

### Layer 8: Active Knowledge and Governance

Question:

> How does the platform maintain current understanding while preserving uncertainty and provenance?

Objects:

- Review states.
- Confidence transitions.
- Deprecated edges.
- Superseded artifacts.
- Human corrections.
- Feedback events.
- Open gaps.
- Roadmap tasks.

Required relations:

- human_review_validates_edge
- feedback_updates_label
- artifact_superseded_by_artifact
- claim_status_changed_by_evidence
- gap_blocks_capability
- task_closes_gap

Completion metrics:

- Feedback-to-update latency.
- Reviewed edge rate.
- Deprecated stale edge rate.
- Open high-priority gap count.
- Claim status freshness.
- Human correction incorporation rate.

## 4. Technical Build Program

### Workstream A: Unified Information Schema

Build a small set of cross-cutting schema records that can unify the platform:

- `ReferenceAuthorityRecord`
- `ConceptRecord`
- `MethodRecord`
- `ProcessRecord`
- `EvidenceRecord`
- `ArtifactRecord`
- `CoverageGapRecord`
- `ReviewStateRecord`

These should not replace existing domain schemas immediately. They should act as overlays and indexes across existing records.

First artifacts:

- `data/kg/information_schema.json`
- `data/kg/information_node_index.jsonl`
- `data/kg/information_edge_index.jsonl`
- `reports/information_schema_alignment.md`

### Workstream B: Methodology Registry 2.0

Expand the method registry from a taxonomy into a runnable method knowledge base.

For each method:

- Inputs required.
- Data standards supported.
- Modalities supported.
- Signal requirements.
- Metadata requirements.
- Event/task requirements.
- Outputs.
- Assumptions.
- Failure modes.
- Notebook template.
- Example datasets.
- Key references.

Priority methods:

- Q-learning / RL modeling.
- Choice decoding.
- Motor decoding.
- Speech decoding.
- Event-aligned neural activity.
- Encoding models.
- Functional connectivity.
- Latent dynamics.
- Aperiodic/spectral analysis.
- Sleep staging.
- Seizure detection.
- Cross-dataset comparison.

First artifacts:

- `data/methods/method_requirements.yaml`
- `data/methods/method_reference_map.jsonl`
- `data/methods/method_dataset_affordances.jsonl`
- `reports/methodology_coverage_report.md`

### Workstream C: Concept Authority Layer

Use Scholarpedia as one source, not the only source.

Inputs:

- Existing `scholarpedia_neural_search_kg_integration_brief.md`.
- `neural_search/ingestion/scholarpedia_builder.py`.
- Existing concept memory modules.
- Existing ontology/atlas refs.

First goal:

- Create a seed authority layer for 25-50 stable concepts.
- Store metadata, URLs, aliases, provenance, and license.
- Generate conservative concept edges.
- Export compact Obsidian concept cards.

First artifacts:

- `data/concepts/reference_authorities.jsonl`
- `data/concepts/concept_authority_records.jsonl`
- `data/kg/concept_authority_edges.jsonl`
- `reports/concept_authority_coverage.md`

### Workstream D: Process Graph

Map the platform itself.

Every important generated artifact should know:

- Who/what creates it.
- What inputs it consumes.
- What command rebuilds it.
- How freshness is checked.
- Which tests validate it.
- Which reports depend on it.

First artifacts:

- `data/kg/process_graph.json`
- `reports/process_graph.md`
- `scripts/doctor.py` or equivalent health command.

This directly addresses the current split between old release artifacts and the newer full-corpus manifest.

### Workstream E: Evidence Spine

Create one evidence spine across claims, findings, edges, affordances, and retrieval explanations.

Each evidence-bearing object should have:

- Evidence source.
- Evidence mode.
- Evidence text or pointer.
- Source URL/path.
- Extractor/procedure.
- Confidence.
- Review state.
- License.
- Freshness.

First artifacts:

- `data/kg/evidence_spine.jsonl`
- `reports/evidence_coverage_report.md`

### Workstream F: Completeness Dashboard

Move beyond dataset coverage into whole-system coverage.

Dashboard sections:

- Concept authority coverage.
- Methodology coverage.
- Dataset metadata coverage.
- Claim/finding evidence coverage.
- Process artifact coverage.
- Retrieval feature/eval coverage.
- Human review coverage.
- License/provenance coverage.

First artifacts:

- `reports/information_completeness_dashboard.md`
- `reports/information_completeness_dashboard.json`

## 5. Priority Build Order

### Phase 1: Map What Exists

Do not ingest more first. Build the inventory.

Tasks:

- Inventory all data directories and generated artifacts.
- Classify artifacts by layer: source, concept, method, dataset, evidence, process, retrieval, governance.
- Identify stale or superseded reports.
- Create an information completeness dashboard.

Output:

- First honest map of the information substrate.

### Phase 2: Define The Missing Schemas

Tasks:

- Add overlay schemas for methods, references, processes, evidence, and artifacts.
- Reuse existing Pydantic/dataclass style.
- Keep additive compatibility with current records.

Output:

- The platform can represent completeness, not only search results.

### Phase 3: Complete Methodology Coverage

Tasks:

- Turn method taxonomy into method requirements.
- Link methods to dataset affordances.
- Link methods to references and notebooks.
- Add method coverage reports.

Output:

- The system can answer, "What must a dataset contain to support this method?"

### Phase 4: Complete Concept Authority Coverage

Tasks:

- Add Scholarpedia metadata-first authority seed.
- Add other authority sources where needed.
- Build concept-to-method and concept-to-affordance links.
- Export Obsidian concept cards.

Output:

- The system can answer, "What does this concept mean, and what data could test it?"

### Phase 5: Complete Evidence Spine

Tasks:

- Normalize evidence records across claims, findings, graph edges, method affordances, and search explanations.
- Ensure license/provenance propagation.
- Add review states.

Output:

- The system can answer, "Why do we believe this, and how strong is the evidence?"

### Phase 6: Build Active Knowledge Loop

Tasks:

- Add process graph.
- Add artifact rebuild/freshness logic.
- Add feedback-to-review-to-update loop.
- Add change reports.

Output:

- The system can answer, "What changed, what improved, what broke, and what still needs work?"

## 6. Redefined Success Metrics

### Completeness Metrics

- 90% of core concepts have an authority source and aliases.
- 80% of priority methods have requirements, assumptions, failure modes, references, and at least one example dataset.
- 90% of top-level generated artifacts have producer process records.
- 95% of evidence-bearing records have provenance, evidence mode, confidence, and license.
- 80% of top search explanations cite an evidence spine record.
- 75% of priority datasets have source URL, species, modality, region, task, and at least one affordance.

### Depth Metrics

- Every priority method maps to required signals and metadata.
- Every priority concept maps to methods and dataset affordances.
- Every priority claim maps to findings or explicit missing evidence.
- Every edge family has an evaluation or review status.
- Every stale report is marked stale or regenerated.

### Fluidity Metrics

- Fast candidate retrieval remains under target latency.
- KG/evidence loading happens after candidate generation where possible.
- Search explanations render without loading the whole graph.
- Full rebuild processes are documented and reproducible.
- New source ingestion updates coverage dashboards automatically.

## 7. How This Changes The Roadmap

The previous plan centered trust and adoption. That is still important, but this reframing moves the next technical push toward information completeness:

- Less emphasis on immediate external alpha.
- More emphasis on authority layers, method maps, process maps, and evidence spines.
- Validation still matters, but as one part of governance rather than the whole roadmap.
- Scholarpedia integration becomes part of a broader concept authority program.
- Coverage reports expand from dataset metadata to whole-system understanding.

The core design shift:

> Retrieval should sit on top of a complete information substrate, not be the only organizing principle.

## 8. Immediate Next Actions

1. Create an information inventory script/report.
   - Classify current artifacts by layer.
   - Mark stale/superseded reports.
   - Count records by source and evidence tier.

2. Define the overlay schemas.
   - Reference authority.
   - Method requirements.
   - Process graph.
   - Evidence spine.
   - Artifact manifest.

3. Build Methodology Registry 2.0.
   - Start with 12 priority methods.
   - Add required signals, metadata, assumptions, failure modes, example datasets, references.

4. Build concept authority seed.
   - Use Scholarpedia conservatively.
   - Add 25-50 stable concepts.
   - Preserve license/provenance.

5. Build process graph.
   - Map commands, inputs, outputs, tests, and reports.
   - Resolve artifact-regime confusion.

6. Build information completeness dashboard.
   - One report showing concept, method, dataset, evidence, process, retrieval, review, and license coverage.

## 9. Bottom Line

To achieve the goal of a complete information set, Neural Search should become a knowledge operating system for neuroscience reuse:

- Concepts define what questions mean.
- Methods define what can be done.
- Datasets define what exists.
- Evidence defines why we believe links.
- Processes define how knowledge is produced.
- Coverage defines what is missing.
- Retrieval turns all of that into action.

The next technical frontier is not simply a larger KG. It is a complete, governed, multi-layer information map that lets the system preserve breadth while deepening understanding.
