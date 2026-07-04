# Typed Field Coverage + Relationship Expansion Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the 27 typed finding fields from a stable extractor into a coverage-aware relationship layer that exposes both explicit relationships and hidden cross-dataset/cross-paper patterns. The outcome should let Neural Search answer compound questions such as: "Which datasets involve Alzheimer mouse models, hippocampal recordings, theta disruption, and c-Fos/Arc activation?" and "Which pharmacological perturbations repeatedly invert hippocampal theta-memory relationships across papers?"

**Current baseline:** The 27-field extractor is stable on 122K findings. Structural backbone fields exceed 5%; rich annotation fields sit around 2--5%; sparse high-specificity fields are below 2%. All 247 focused extractor tests pass. Coverage is useful but not yet precision-audited, normalized into field-specific ontologies, or promoted into first-class paper/dataset relationship artifacts.

**Architecture:** Coverage profiler -> field lexicon registry -> precision-gated extractor expansion -> typed-field normalizer -> base relationship builder -> hidden relationship miner -> KG/search/report surfaces.

**Tech Stack:** Python 3.11, Pydantic/dataclasses, JSONL artifacts, pytest, existing `neural_search.literature.typed_finding_extractor`, existing literature KG builder, existing graph schema, existing paper-dataset links, optional DuckDB/NetworkX for relationship mining.

## Global Constraints

- Preserve the 27-field public API from `extract_typed_fields()`.
- Improve coverage only with precision gates; do not inflate counts with broad generic matches.
- Every new regex or lexicon entry must have positive and negative tests.
- All generated artifacts must distinguish observed evidence from inferred relationships.
- Hidden relationships must carry support counts, lift/enrichment score, confidence, and provenance.
- Do not make retrieval-quality claims from typed-field coverage alone.
- Commands run from repo root unless explicitly noted.

---

## Coverage Baseline To Preserve

| Tier | Fields | Current role |
|------|--------|--------------|
| Structural backbone | `condition`, `statistical_relation`, `anatomical_direction`, `negation`, `effect_scale`, `behavioral_measure`, `population_type` | Broad interpretive scaffold for most findings |
| Rich annotation | `temporal_pattern`, `molecular_marker`, `frequency_band`, `metabolic_context`, `pharmacological_agent`, `synaptic_plasticity`, `developmental_stage`, `injury_model`, `social_affective`, `genetic_tool`, `connectivity_type`, `network_coupling`, `sensory_stimulus` | Biological, experimental, and mechanistic context |
| Sparse high-specificity | `sleep_stage`, `comparison_condition`, `decoding_type`, `dimensionality_reduction`, `computational_model` | Rare but strong query/discovery anchors |

Coverage targets should be reported as ranges, not hard promises. A field can remain sparse if its matches are specific and scientifically useful.

---

## Artifact Map

**Create:**
```
configs/literature/typed_field_lexicons.yaml
scripts/literature/profile_typed_field_coverage.py
scripts/literature/enrich_typed_fields.py
scripts/literature/build_typed_relationships.py
scripts/literature/mine_hidden_relationships.py
artifacts/literature/typed_fields/coverage_baseline.json
artifacts/literature/typed_fields/coverage_report.md
artifacts/literature/typed_fields/enriched_findings.jsonl
artifacts/literature/typed_fields/base_relationships.jsonl
artifacts/literature/typed_fields/hidden_relationships.jsonl
reports/eval/typed_field_audit_template.csv
reports/eval/typed_relationships_report.md
tests/test_typed_field_coverage_profile.py
tests/test_typed_field_relationships.py
tests/test_hidden_relationship_mining.py
```

**Modify:**
```
neural_search/literature/typed_finding_extractor.py
neural_search/literature/kg_builder.py
neural_search/graph/schema.py
docs/whitepaper/neural_search_whitepaper.tex
```

---

## Task 1: Freeze The 27-Field Coverage Baseline

**Purpose:** Make the coverage report reproducible instead of pasted from a terminal run.

- [ ] Create `scripts/literature/profile_typed_field_coverage.py`.
- [ ] Inputs:
  - `artifacts/literature/findings_tier1_ollama.jsonl`
  - optional `--sample-size`
  - optional `--seed`
- [ ] Output:
  - `artifacts/literature/typed_fields/coverage_baseline.json`
  - `artifacts/literature/typed_fields/coverage_report.md`
- [ ] Report per field:
  - matched finding count
  - coverage percentage
  - top values
  - top regex/lexicon source when available
  - 20 example finding IDs for audit
- [ ] Add tests using synthetic findings.
- [ ] Run:

```bash
pytest tests/test_typed_field_coverage_profile.py -v
python scripts/literature/profile_typed_field_coverage.py --sample-size 5000 --seed 13
```

**Acceptance:** The script reproduces all 27 fields and emits a tiered report matching the current baseline order.

---

## Task 2: Move Field Vocabulary Into A Lexicon Registry

**Purpose:** Make field expansion reviewable without burying everything in Python regex lists.

- [ ] Create `configs/literature/typed_field_lexicons.yaml`.
- [ ] For each field, store entries with:
  - `value`
  - `patterns`
  - `aliases`
  - `negative_patterns`
  - `precision_risk`: `low | medium | high`
  - `examples_positive`
  - `examples_negative`
- [ ] Keep Python defaults for bootstrapping, but allow `typed_finding_extractor.py` to load registry additions.
- [ ] Add a registry validator that rejects:
  - duplicate values within a field
  - empty patterns
  - high-risk entries without negative examples
  - patterns that match all-empty or generic text
- [ ] Add tests for registry validation and extractor integration.

**Acceptance:** Existing 247 tests continue to pass, and registry-added test patterns appear in extraction output.

---

## Task 3: Field-By-Field Coverage Expansion

**Purpose:** Raise useful coverage with targeted lexicon additions while protecting precision.

### Priority A: Highest Query Value

- [ ] `sensory_stimulus`: add common visual/auditory/tactile/olfactory task terms from top unmatched findings.
- [ ] `injury_model`: expand disease model aliases for APP/PS1, 3xTg, tauopathy, alpha-synuclein, rotenone, cuprizone, EAE, LPS, hypoxia/ischemia.
- [ ] `pharmacological_agent`: expand receptor-class and agent aliases while avoiding generic "drug" matches.
- [ ] `molecular_marker`: expand IEG, cytokine, glial, receptor-subunit, assay, and transcriptomics terms.
- [ ] `genetic_tool`: expand viral vector, driver line, reporter, knockout/knockin, inducible system terms.

### Priority B: Relationship Context

- [ ] `connectivity_type`: expand pathway terms and causal/connectivity analysis methods.
- [ ] `network_coupling`: improve theta-gamma, replay, spike-field, synchrony, up/down state detection.
- [ ] `sleep_stage`: expand sleep/wake transition and consolidation terminology.
- [ ] `developmental_stage`: add age-window parsing with species-aware rules.
- [ ] `comparison_condition`: fix rare but important control/baseline/pre-post mentions.

### Priority C: Sparse High-Specificity Anchors

- [ ] `dimensionality_reduction`: expand latent dynamics, GLM-PCA, CCA, jPCA, LFADS, GPFA, demixed methods.
- [ ] `computational_model`: expand biophysical, neural mass, RL, Bayesian, predictive-coding, attractor, network model terms.
- [ ] `decoding_type`: expand classifiers, encoding/decoding models, cross-validation, BCI, readout methods.

**Acceptance:** Each field has targeted positive and negative tests. Coverage increases are reported alongside precision-risk notes, not as standalone success.

---

## Task 4: Normalize Typed Field Values

**Purpose:** Convert string matches into stable values that can be joined across findings, papers, and datasets.

- [ ] Create normalizer functions for field groups:
  - disease/injury model family
  - pharmacological target class
  - molecular marker family
  - sensory modality
  - sleep/wake macro-state
  - analysis method family
  - connectivity pathway class
- [ ] Add normalized companion fields under `_typed_normalized`, for example:

```json
{
  "injury_model": ["alzheimer_app"],
  "_typed_normalized": {
    "disease_family": ["alzheimer"],
    "model_organism_context": ["transgenic_mouse"],
    "marker_family": ["immediate_early_gene"]
  }
}
```

- [ ] Preserve raw extracted values.
- [ ] Add tests for one-to-many and many-to-one normalization.

**Acceptance:** A finding tagged `5xFAD + c-Fos + hippocampus + theta` normalizes into disease family, marker family, region, and frequency dimensions without losing raw values.

---

## Task 5: Build Base Relationships

**Purpose:** Materialize directly observed relationships from single findings and linked papers.

- [ ] Create `scripts/literature/build_typed_relationships.py`.
- [ ] Input:
  - enriched findings JSONL
  - paper metadata shards
  - `artifacts/literature/paper_dataset_links.jsonl`
- [ ] Output `base_relationships.jsonl`.
- [ ] Emit observed relationships:
  - `finding_has_typed_field`
  - `paper_reports_typed_field`
  - `paper_reports_field_pair`
  - `dataset_linked_to_typed_field`
  - `dataset_linked_to_field_pair`
  - `finding_connects_region_to_marker`
  - `finding_connects_model_to_signal`
  - `finding_connects_agent_to_effect`
- [ ] Every row includes:
  - `relationship_id`
  - `relationship_type`
  - `source_id`
  - `target_id`
  - `field`
  - `value`
  - `finding_ids`
  - `paper_ids`
  - `dataset_ids`
  - `support_count`
  - `confidence`
  - `evidence_text`

**Acceptance:** The example `5xFAD + hippocampus + theta + c-Fos` produces typed base relationships linking disease model, region, band, molecular marker, paper, and any linked datasets.

---

## Task 6: Mine Hidden Relationships

**Purpose:** Surface relationships that are not a single explicit edge but recur across papers/datasets more than expected.

- [ ] Create `scripts/literature/mine_hidden_relationships.py`.
- [ ] Compute pair and motif enrichment across:
  - field-field pairs
  - region-field pairs
  - disease-model-signal pairs
  - agent-effect pairs
  - dataset-paper-field bridges
  - three-node motifs such as `injury_model -> region -> frequency_band`
- [ ] Score each candidate with:
  - support count
  - paper count
  - dataset count
  - lift or PMI
  - confidence interval or bootstrap stability
  - contradiction count when opposing directions exist
- [ ] Output `hidden_relationships.jsonl`.
- [ ] Relationship classes:
  - `enriched_cooccurrence`
  - `cross_dataset_bridge`
  - `latent_mechanism_hint`
  - `contested_pattern`
  - `underexplored_gap`
- [ ] Require minimum support thresholds, for example:
  - at least 3 findings
  - at least 2 papers
  - at least 1 linked dataset for dataset-facing relationships

**Acceptance:** Hidden relationships are clearly labeled as inferred patterns and never mixed with direct evidence edges.

---

## Task 7: Promote Relationships Into The KG

**Purpose:** Make typed relationships queryable from the graph layer.

- [ ] Extend graph schema with node/edge types only if needed; prefer existing `finding`, `paper`, `dataset`, and concept-like node types where possible.
- [ ] Add typed-field concept nodes:
  - `typed_field_value:{field}:{value}`
  - labels such as `injury_model:alzheimer_app`
- [ ] Add KG edges:
  - `finding_has_typed_field`
  - `paper_reports_typed_field`
  - `dataset_linked_to_typed_field`
  - `typed_field_cooccurs_with`
  - `typed_field_hidden_relationship`
- [ ] Update `kg_builder.py` or create a dedicated typed relationship ingest module.
- [ ] Add graph tests for edge creation, evidence propagation, and deduping.

**Acceptance:** The KG can answer one-hop and multi-hop queries from dataset -> linked paper -> finding -> typed field values.

---

## Task 8: Build Search And Report Surfaces

**Purpose:** Make the new relationships visible to users and future agents.

- [ ] Create `reports/eval/typed_relationships_report.md`.
- [ ] Include:
  - coverage by field
  - top base relationships
  - top hidden relationships
  - contested patterns
  - underexplored gaps
  - audit queue samples
- [ ] Add API-ready query helpers:
  - find datasets by typed-field conjunction
  - find papers by hidden relationship
  - find contradictions for a field pair
  - find low-coverage but high-specificity fields
- [ ] Update whitepaper status only with operational/audit language.

**Acceptance:** A user can query for Alzheimer + hippocampus + theta + molecular marker and receive papers, findings, and linked datasets with provenance.

---

## Task 9: Precision Audit Loop

**Purpose:** Prevent coverage expansion from becoming noise.

- [ ] Generate `reports/eval/typed_field_audit_template.csv`.
- [ ] Sample by:
  - field
  - value
  - confidence/risk tier
  - high-support hidden relationship
  - sparse high-specificity matches
- [ ] Add columns:
  - `is_correct`
  - `too_broad`
  - `missed_context`
  - `suggested_pattern_change`
  - `reviewer_notes`
- [ ] Add an audit importer that computes precision by field and value.
- [ ] Gate promoted hidden relationships on audited or conservative high-confidence rules.

**Acceptance:** Coverage report shows both match rate and audited precision where labels exist.

---

## Task 10: Documentation And Whitepaper Integration

**Purpose:** Keep claims aligned with validation state.

- [ ] Update `docs/whitepaper/neural_search_whitepaper.tex` with:
  - reproducible coverage artifact path
  - typed relationship artifact path
  - hidden relationship caveat
  - precision audit status
- [ ] Update the claim/vault plan to reference 27 typed fields instead of older 16-field language.
- [ ] Add a short example query:
  - "Alzheimer model + hippocampus + theta + c-Fos/Arc"
  - expected outputs: matching findings, papers, datasets, hidden relationships, contradictions.

**Acceptance:** Documentation distinguishes direct finding evidence, inferred hidden patterns, and validated claims.

---

## Milestones

### Milestone 1: Reproducible Baseline

- Tasks 1--2 complete.
- Coverage report generated from code.
- Registry validates.

### Milestone 2: Precision-Gated Coverage Expansion

- Task 3 complete for Priority A fields.
- Focused tests pass.
- Coverage change report includes examples and risks.

### Milestone 3: Relationship Artifacts

- Tasks 4--6 complete.
- Base and hidden relationship JSONL artifacts exist.
- Example compound queries work offline.

### Milestone 4: KG + Search Integration

- Tasks 7--8 complete.
- KG exposes typed fields and hidden relationships.
- API/query helpers return provenance-rich results.

### Milestone 5: Audit + Publishable Framing

- Tasks 9--10 complete.
- Precision audit loop exists.
- Whitepaper uses careful operational language.

---

## Example End-State Query

**Question:** Which datasets used Alzheimer mouse models with hippocampal recordings and show theta disruption plus c-Fos/Arc activation?

**Expected resolution path:**

1. Match typed fields:
   - `injury_model in [alzheimer_app, alzheimer_tau, alzheimer_amyloid]`
   - `regions contains hippocampus or CA1`
   - `frequency_band contains theta`
   - `molecular_marker contains cfos_ieg or arc`
2. Traverse:
   - finding -> paper
   - paper -> linked dataset
   - dataset -> modality/region/task metadata
3. Return:
   - direct findings
   - supporting papers
   - linked datasets
   - hidden enriched relationships
   - contradictions or missing controls
   - audit status

This is the practical bar for the typed-field expansion: not more tags for their own sake, but richer, provenance-backed cross-dataset reasoning.
