# Vault Enrichment + Novel Claims — Design Spec

**Date:** 2026-06-21
**Branch:** claude/neuronpedia-foundation
**Status:** Approved for implementation
**Priority:** Sub-project A of three (A: Vault + Claims → B: Reprocessing + Methods → C: Foundation Model Strategy)

---

## 1. Goal

Transform the Obsidian vault from an evaluation memory bank into a full knowledge store grounded in 122K literature findings. Synthesize those findings into machine-readable claim objects that a multi-agent neuroscience analysis system can load as domain priors, test against new data, and use to generate hypotheses.

Secondary consumers: whitepaper claims registry (credibility + citations), and eventually the search UI (insight cards alongside results).

---

## 2. Architecture

Four layers, each building on existing infrastructure:

```
LAYER 4: AGENT INTERFACE
  FastAPI /api/claims/* endpoints
  Compact claim objects with agent_digest field
  Contradiction index, gap map, hypothesis seeds

LAYER 3: KNOWLEDGE GRAPH (extends existing)
  New node type: claim  (node:claim:*)
  New edge types: supports, contradicts,
                  supported_by_dataset, supported_by_paper
  Vault = human-readable export of KG claim nodes

LAYER 2: CLAIM SYNTHESIS PIPELINE (new)
  Normalize → Cluster → Synthesize → Validate → Ingest → Export

LAYER 1: EXTRACTION (already exists)
  findings_tier1_ollama.jsonl (122K findings)
  typed_finding_extractor.py (16 rule-based fields)
  finding_extractor.py (LLM prompts, SPO triples)
  paper_dataset_links.jsonl
```

Design principles (from Meta's AI Second Brain architecture):
- **Progressive disclosure**: agents load claim digests by topic, not all 122K findings at once
- **Grounding through real records**: every claim carries `supporting_papers` + `supporting_datasets` — no hallucinated citations
- **Composable pipeline**: each synthesis script is standalone and independently runnable

---

## 3. Ontology Sources

| Ontology | Source | Covers | Use |
|----------|--------|--------|-----|
| Allen CCF v3 | `api.brain-map.org` | Hierarchical rodent brain regions | Primary region normalization (corpus is ~70% rodent) |
| UBERON | OBO Foundry | Cross-species anatomy | Mouse→human homologue mapping |
| Cell Ontology (CL) | OBO Foundry | Cell types | Normalize PV/parvalbumin/fast-spiking interneuron |
| Existing `behavioral_task_ontology.yaml` | Project | Tasks | Unchanged |

Fetched on first run and cached to `data/ontology/allen_ccf.json` and `data/ontology/uberon_bridge.json`.

Region normalization expands each region to its full parent chain (CA1 → hippocampus → limbic system → telencephalon) so claims are searchable at multiple granularities.

---

## 4. Vault Expansion

Three new sections added to the existing eight. Nothing removed.

```
obsidian_vault/
├── 00_Project/
│   └── prompts/                  ← NEW: versioned LLM extraction templates
│       ├── extraction_v2.yaml        (current Claude Haiku finding prompt)
│       ├── synthesis_v1.yaml         (claim synthesis prompt)
│       └── validation_v1.yaml        (contradiction detection prompt)
├── 03_Datasets/                  (unchanged — cards gain method_history field)
├── 08_Dashboards/
│   └── Claim Coverage.md         ← NEW: Dataview dashboard
├── 09_Literature/                ← NEW: one card per paper
│   └── paper_{openalex_id}.md
├── 10_Claims/                    ← NEW: one card per synthesized claim
│   └── cl_{slug}_{index}.md
└── ... (all other sections unchanged)
```

### 4.1 Paper Card Schema (`09_Literature/`)

```yaml
---
paper_id: openalex:W123456789
doi: 10.1016/j.neuron.2021.01.001
title: "Theta oscillations gate spatial memory encoding in CA1"
authors: ["Buzsaki G", "Moser EI"]
year: 2021
n_findings: 4
finding_ids: [finding_W123_0, finding_W123_1, finding_W123_2, finding_W123_3]
linked_datasets: [dandi:000026, openneuro:ds000120]
modalities: [neuropixels, lfp]
regions: [hippocampus, ca1]
species: [mouse]
extraction_model: claude-haiku-4-5-20251001
extraction_prompt_version: extraction_v2
---
## Findings
- [[finding_W123_0]] Theta power increases during correct navigation trials (conf: 0.9)
- [[finding_W123_1]] CA1 phase precession correlates with place field traversal (conf: 0.87)
```

### 4.2 Claim Card Schema (`10_Claims/`)

```yaml
---
claim_id: node:claim:theta_spatial_memory_rodent_001
statement: "Theta oscillations in hippocampus correlate with spatial memory encoding in rodents"
direction: correlation
regions: [hippocampus, ca1]
species: [mouse, rat]
consensus_confidence: 0.87
n_supporting_findings: 47
n_contradicting_findings: 3
magnitude_summary: "r=0.6–0.8 across studies"
timescale: millisecond
evidence_strength: direct
status: active          # active | contested | deprecated
supporting_datasets: [dandi:000026, openneuro:ds000120]
supporting_papers: [openalex:W123, openalex:W456]
contradicted_by: [node:claim:theta_not_required_navigation_002]
synthesis_model: claude-haiku-4-5-20251001
synthesis_prompt_version: synthesis_v1
synthesized_at: 2026-06-21
---
## Agent Digest
47 findings from 12 papers across 6 datasets consistently show theta power
(4–8 Hz) correlates positively (r=0.6–0.8) with correct spatial navigation
trials in CA1. 3 contradicting findings from pharmacological disruption studies.

## Evidence
### Supporting (47)
- [[paper_W123456789]] — "Theta power increases during correct trials" (conf: 0.9)

### Contradicting (3)
- [[paper_W987654321]] — "Spatial memory intact after theta disruption" (conf: 0.75)
```

The `agent_digest` is a pre-synthesized paragraph agents load as a prior — they never scan individual findings to know what the literature says on a topic.

Every claim card stores `synthesis_prompt_version` so changing a prompt reveals exactly which claims need re-synthesis.

---

## 5. Claim Synthesis Pipeline

Seven scripts forming a linear pipeline, each independently runnable:

```
findings_tier1_ollama.jsonl (122K)
        │
        ▼
normalize_regions.py
  - Pull Allen CCF structure tree → data/ontology/allen_ccf.json
  - Pull UBERON bridge → data/ontology/uberon_bridge.json
  - Normalize region strings in all findings
  - Output: artifacts/literature/findings_normalized.jsonl
        │
        ▼
cluster_findings.py
  - Group by (normalized_regions, result_direction, species, tasks)
  - Rule-based exact match first; embedding similarity for fuzzy grouping
  - Min cluster size: 3 findings (configurable)
  - Output: artifacts/claims/finding_clusters.jsonl
        │
        ▼
synthesize_claims.py
  - Claude Haiku per cluster → consensus claim text
  - Fields: statement, magnitude_summary, timescale, evidence_strength
  - Prompt template: configs/literature/synthesis_v1.yaml
  - Output: artifacts/claims/claims_raw.jsonl
        │
        ▼
detect_contradictions.py
  - Pair claims with opposing directions in overlapping (region, task) domains
  - Mark status: active | contested | deprecated
  - Output: artifacts/claims/claims_validated.jsonl
        │
        ▼
ingest_claims_to_kg.py
  - Write claim nodes + edges to KG JSONL
  - Edge types: supports, contradicts, supported_by_dataset, supported_by_paper
  - Output: data/graph/claims_kg.jsonl
        │
        ├──▶ export_literature.py → obsidian_vault/09_Literature/
        └──▶ export_claims.py     → obsidian_vault/10_Claims/
```

Clustering is deterministic and rule-based first to ensure reproducibility. Embedding similarity is used only as a secondary fallback for fuzzy grouping, not as the primary strategy.

---

## 6. KG Integration

All new elements follow the existing `kg_builder.py` pattern — no schema changes required.

### New Node Type: `claim`

```
node:claim:{slug}_{index}

properties:
  statement            str   — the full claim sentence
  direction            str   — increase | decrease | correlation | mechanism | other
  regions              list  — normalized region names (Allen CCF)
  species              list  — canonical species names
  consensus_confidence float — 0.0–1.0
  n_supporting         int
  n_contradicting      int
  magnitude_summary    str   — human-readable effect size summary
  timescale            str   — millisecond | second | minute | hour | day | chronic
  evidence_strength    str   — direct | indirect | computational | review
  status               str   — active | contested | deprecated
  synthesis_model      str
  synthesis_prompt_version str
  synthesized_at       str   — ISO datetime
```

### New Edge Types

```
claim  ──[supports]──────────────▶  finding
claim  ──[contradicts]───────────▶  claim
claim  ──[supported_by_dataset]──▶  dataset
claim  ──[supported_by_paper]────▶  paper
claim  ──[derived_from_finding]──▶  finding
```

All edges carry `confidence` and `evidence_text` — consistent with existing edge schema in `neural_search/graph/schema.py`.

---

## 7. Agent Interface

Six new FastAPI endpoints in `apps/api/claims_router.py`:

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/claims` | List claims; filters: `regions`, `species`, `direction`, `status`, `min_confidence` |
| GET | `/api/claims/{id}` | Full claim with all KG edges |
| GET | `/api/claims/{id}/evidence` | Supporting + contradicting datasets and papers |
| GET | `/api/claims/contradictions` | All contested claim pairs |
| GET | `/api/claims/gaps` | (region, task) combos with no claims; filter by `region` |
| GET | `/api/claims/digest` | Compact batch export for agent context loading |

### `/api/claims/digest` Response Shape

The primary agent primitive. Designed to be injected directly into an agent's system prompt or context window:

```json
{
  "claims": [
    {
      "claim_id": "node:claim:theta_spatial_memory_rodent_001",
      "statement": "Theta oscillations in hippocampus correlate with spatial memory encoding in rodents",
      "confidence": 0.87,
      "direction": "correlation",
      "regions": ["hippocampus", "ca1"],
      "species": ["mouse", "rat"],
      "n_evidence": 47,
      "status": "active",
      "contradicted_by": ["node:claim:theta_not_required_navigation_002"],
      "supporting_datasets": ["dandi:000026", "openneuro:ds000120"],
      "agent_digest": "47 findings from 12 papers across 6 datasets consistently show theta power (4–8 Hz) correlates positively (r=0.6–0.8) with correct spatial navigation trials in CA1. 3 contradicting findings from pharmacological disruption studies."
    }
  ],
  "generated_at": "2026-06-21T00:00:00Z",
  "total": 1
}
```

---

## 8. Prompt Versioning in Vault

LLM extraction and synthesis prompts are stored as versioned YAML in `obsidian_vault/00_Project/prompts/` and as configs in `configs/literature/`. Both locations are kept in sync — the vault copy is for human auditability, the `configs/` copy is what the pipeline reads.

Every claim node and paper card references the prompt version that generated it. Changing a prompt version bumps the version string, making it trivial to identify which claims need re-synthesis via:

```bash
grep synthesis_prompt_version obsidian_vault/10_Claims/*.md | grep v1 | wc -l
```

---

## 9. Sub-project B Integration Points (Groundwork Only)

Within this sprint, dataset cards in `03_Datasets/` gain two new fields that lay the groundwork for Sub-project B (Dataset Reprocessing + Method Bank) without implementing that system:

```yaml
method_history: []          # methods applied to this dataset (populated from findings)
reanalysis_opportunity: null  # scored in Sub-project B
```

---

## 10. File Map

**New files:**
```
scripts/literature/normalize_regions.py
scripts/literature/cluster_findings.py
scripts/literature/synthesize_claims.py
scripts/literature/detect_contradictions.py
scripts/literature/ingest_claims_to_kg.py
scripts/obsidian/export_literature.py
scripts/obsidian/export_claims.py
neural_search/literature/region_normalizer.py
neural_search/literature/claim_synthesizer.py
neural_search/literature/claim_kg_builder.py
apps/api/claims_router.py
configs/literature/synthesis_v1.yaml
configs/literature/validation_v1.yaml
obsidian_vault/00_Project/prompts/          (directory)
obsidian_vault/09_Literature/               (directory)
obsidian_vault/10_Claims/                   (directory)
data/ontology/                              (directory, gitignored content)
artifacts/claims/                           (directory)
tests/test_region_normalizer.py
tests/test_claim_synthesizer.py
tests/test_claim_kg_builder.py
tests/test_claims_api.py
```

**Modified files:**
```
apps/api/main.py                            (include claims_router)
scripts/obsidian/export_dataset_cards.py    (add method_history field)
neural_search/obsidian/templates.py         (add paper + claim templates)
neural_search/obsidian/io.py               (add 09_Literature + 10_Claims paths)
```

---

## 11. Future Sprints (Out of Scope Here)

- **Sub-project B**: Dataset reprocessing tracker — score datasets by reanalysis opportunity, build method validation bank, rank/suggest methods per use case
- **Sub-project C**: Foundation model data strategy — tokenization primitives, cross-dataset combination, pretraining corpus curation
- **Multi-agent analysis system**: Uses this vault + KG + claims as domain prior loading; agents traverse graph, test hypotheses against datasets, generate novel claims
