# Obsidian Eval Memory + Weak Supervision Labeling — Design Spec

**Date:** 2026-06-10
**Branch:** claude/latent-usefulness-v08
**Author:** Sidhu Lyalkar + Claude

---

## 1. Motivation

The current evaluation pipeline requires manual annotation via
`scripts/eval/annotate_candidates.py`. This creates a bottleneck: every
new query or corpus expansion demands fresh human effort, and labels carry
no explicit provenance. The new design replaces hand-labeling with a
weak-supervision stack that:

1. Extracts structured evidence from existing corpus and query fields.
2. Applies deterministic labeling functions for constraint satisfaction.
3. Optionally applies an LLM rubric judge.
4. Combines votes into gold/silver/bronze qrels tiers.
5. Routes only the genuinely uncertain examples to a human audit queue
   surfaced inside an Obsidian vault.

---

## 2. What Gets Deleted (Hand-Labeling Removal)

| Path | Reason |
|---|---|
| `neural_search/labeling/` (entire package) | Interactive session / storage / agreement infrastructure |
| `neural_search/evaluation/label_relevance.py` | Interactive CLI labeler |
| `neural_search/evaluation/relevance.py` | RelevanceJudgment / RelevanceLabelSet model |
| `scripts/run_annotation_session.py` | Annotation session entry point |
| `scripts/eval/annotate_candidates.py` | Pool annotation script |
| `tests/test_relevance.py` | Tests for the above |

**Migration note:** `compute_hard_negative_violations` and the NDCG/MRR
compute functions currently in `neural_search/evaluation/relevance.py`
move to `neural_search/eval/label_ensemble.py` and
`scripts/eval/compute_ir_metrics.py` respectively.  
`tests/test_search_quality.py` is updated to import from the new location.

---

## 3. New Package Layout

```
neural_search/
  eval/                          ← new package (weak-supervision eval)
    __init__.py
    evidence.py                  ← QuerySpec, DatasetEvidence, PairEvidence
    query_decomposition.py       ← parse benchmark_queries.jsonl → QuerySpec
    labeling_functions.py        ← 13 deterministic LFs
    llm_judge.py                 ← optional LLM rubric judge
    label_ensemble.py            ← vote aggregation, qrels tiers
  obsidian/                      ← new package (vault I/O)
    __init__.py
    templates.py                 ← frontmatter dataclasses + Markdown renderers
    io.py                        ← safe read/write that preserves manual edits

scripts/
  eval/
    build_evidence.py            ← new
    run_labeling_functions.py    ← new
    judge_candidates.py          ← new (optional)
    build_qrels_from_votes.py    ← new
    hard_negative_analysis.py    ← new (dedicated script)
    compute_ir_metrics.py        ← extend (--qrels-tier flag, tier warnings)
    compute_calibration.py       ← extend (--qrels flag, tier warnings)
  obsidian/
    init_vault.py
    export_dataset_cards.py
    export_query_cards.py
    export_audit_queue.py
    import_audits.py
    compile_vault.py
    export_claim_registry.py

configs/
  judges/
    rubric_judge_v1.yaml

obsidian_vault/                  ← vault root (NOT in .gitignore)
  00_Project/
  01_Rubrics/
  02_Ontology/
  03_Datasets/
  04_Queries/
  05_Annotations/
    Human Audits/
  06_Evaluations/
  07_Whitepaper/
  08_Dashboards/
  99_Templates/

artifacts/
  eval/
    query_specs.jsonl
    dataset_evidence.jsonl
    pair_evidence.jsonl
    label_function_votes.jsonl
    llm_judgments.jsonl          ← optional
    audit_queue.jsonl
    label_ensemble_report.json
    human_audits.jsonl
  qrels_gold.jsonl
  qrels_silver.jsonl
  qrels_bronze.jsonl

docs/
  OBSIDIAN_EVAL_MEMORY.md
  WEAK_SUPERVISION_LABELING.md
  HUMAN_AUDIT_PROTOCOL.md

reports/
  eval/
    whitepaper_claims_status.md
```

---

## 4. Data Models

### 4.1 QuerySpec (`neural_search/eval/evidence.py`)

Parsed from `artifacts/benchmark_queries.jsonl` fields:

```python
@dataclass
class QuerySpec:
    query_id: str
    query_text: str
    intent: str
    scientific_goal: str
    required_modalities: list[str]
    preferred_modalities: list[str]
    required_species: list[str]
    preferred_species: list[str]
    brain_regions: list[str]
    task_constraints: list[str]
    data_level_requirements: list[str]
    hard_negatives: list[str]        # from known_failure_modes
    analysis_affordances: list[str]
```

### 4.2 DatasetEvidence

Normalized from corpus record fields:

```python
@dataclass
class DatasetEvidence:
    record_id: str                   # "{source}:{source_id}"
    source: str
    title: str
    description: str | None
    species: list[str]               # normalized
    modalities: list[str]            # normalized
    data_levels: list[str]
    tasks: list[str]
    regions: list[str]
    license: str | None
    doi: str | None
    url: str | None
    raw_data_available: bool
    metadata_completeness: float     # 0.0–1.0
    has_behavior: bool
    has_trials: bool
    data_standards: list[str]
```

### 4.3 PairEvidence

```python
@dataclass
class PairEvidence:
    query_id: str
    record_id: str
    query: QuerySpec
    dataset: DatasetEvidence
    pooled_from: list[str]
    min_rank: int
    priority: str
```

### 4.4 LFVote

```python
@dataclass
class LFVote:
    lf_name: str
    label: int          # 0–3
    confidence: float   # 0.0–1.0
    rationale: str
    abstain: bool       # True when LF cannot judge
```

### 4.5 Qrel record (all three tiers share schema)

```jsonl
{"query_id": "q_0001", "record_id": "dandi:000004", "label": 2, "confidence": 0.85,
 "source": "silver", "provenance": ["lf_required_modality", "lf_species_constraint"],
 "created": "2026-06-10T..."}
```

---

## 5. Labeling Functions (13 total)

| LF name | Fires when | Hard override? |
|---|---|---|
| `lf_hard_negative` | candidate text matches known_failure_modes | **Yes — votes label 0 at 0.95** |
| `lf_required_modality` | checks required_modalities vs dataset.modalities | No |
| `lf_partial_modality` | checks preferred_modalities | No |
| `lf_species_constraint` | checks required_species | No |
| `lf_task_constraint` | checks task_constraints | No |
| `lf_region_constraint` | checks brain_regions | No |
| `lf_data_level_required` | checks data_level_requirements | No |
| `lf_raw_data_available` | checks has_raw_data | No |
| `lf_license_reusable` | checks license permissiveness | No |
| `lf_metadata_completeness` | checks metadata_completeness score | No |
| `lf_analysis_affordance` | checks affordance overlap | No |
| `lf_pipeline_reuse` | checks data_standards + modality for pipeline reuse intent | No |
| `lf_meta_analysis_depth` | checks sample size, NWB, behavioral metadata for META_ANALYSIS intent | No |

All LFs return `LFVote`. LFs abstain (rather than emit a low-confidence
vote) when they have no signal.

---

## 6. Label Ensemble

Aggregation order:
1. If any LF votes `lf_hard_negative` → label = 0, source = "bronze" (hard override)
2. Weighted average of non-abstaining votes by confidence
3. Disagreement score = variance of non-abstaining votes
4. Assign tier:
   - **Gold**: human-audited label exists → use it
   - **Silver**: ≥3 non-abstaining LFs, confidence ≥ 0.75, disagreement < 0.5, no hard-negative override
   - **Bronze**: everything else

Audit queue priority score:
```
priority = disagreement_score * (1 + hard_negative_flag) * (1 / min_rank + 1)
```

---

## 7. LLM Rubric Judge (optional)

- Activated only when `configs/judges/rubric_judge_v1.yaml` exists and
  `ANTHROPIC_API_KEY` (or configured provider key) is set.
- Input: query_spec + dataset_evidence + rubric text + LF votes summary
- Output: strict JSON with `label`, `confidence`, `rationale`,
  `supporting_evidence`, `missing_evidence`, `hard_negative_detected`
- Never infers facts absent from the evidence.
- Self-consistency: runs N times, takes majority label.
- LLM votes count toward ensemble with configurable weight (default 1.5×).

---

## 8. Obsidian Vault

The vault is a **human-readable memory bank**. JSONL/YAML artifacts are
the authoritative runtime data. The vault is for:
- Human audit of uncertain examples
- Rubric notes and ontology docs
- Whitepaper claim tracking

**Write safety rule:** `obsidian/io.py` reads existing frontmatter before
writing. Any field present in the file that is not generated (i.e., set by
a human — `label`, `confidence`, `audit_status`, rationale body text) is
**never overwritten** by export scripts.

---

## 9. Qrels Tier Definitions

| Tier | Source | Valid for |
|---|---|---|
| **Gold** | Human-audited labels only | Scientific claims in paper |
| **Silver** | High-confidence ensemble (≥3 LFs, conf ≥ 0.75) | Development metrics, ablations |
| **Bronze** | Rule/LLM/weak-signal labels | Exploratory runs, debugging |

`compute_ir_metrics.py` emits a `WARNING: using {silver|bronze} qrels` to
stderr when tier is not gold. Gold metrics go into whitepaper.

---

## 10. Metrics Changes

`compute_ir_metrics.py` additions:
- `--qrels` now accepts any of the three tier files
- `--qrels-tier` flag (`gold|silver|bronze`) for metadata tagging in report
- `hard_negative_violation_rate` tracked separately per tier
- Bootstrap CI (1000 samples) already implemented; kept

New `hard_negative_analysis.py`:
- Dedicated hard-negative violation report across all tiers
- Outputs violation rate by query, by LF, and by source adapter

---

## 11. Tests to Add

| File | Covers |
|---|---|
| `tests/test_obsidian_io.py` | Markdown/frontmatter roundtrip; manual field preservation |
| `tests/test_obsidian_exports.py` | Dataset card + query card export |
| `tests/test_eval_evidence.py` | Evidence extraction from corpus + query records |
| `tests/test_labeling_functions.py` | All 13 LFs; hard-negative override |
| `tests/test_label_ensemble.py` | Gold/silver/bronze assignment; disagreement scoring |
| `tests/test_qrels_tiers.py` | Tier separation; metrics scripts accept each tier file |

---

## 12. Obsidian Setup (Human Instructions)

**One-time setup (5 minutes):**

1. Download Obsidian: [obsidian.md](https://obsidian.md) — free desktop app.
2. Open as a **new vault** pointing to `obsidian_vault/` inside this repo.
3. Install these Community Plugins (Settings → Community Plugins → Browse):
   - **Dataview** — query notes with SQL-like syntax:
     `TABLE label, confidence FROM "05_Annotations/Human Audits" WHERE audit_status = "pending"`
   - **Linter** — validates and normalises YAML frontmatter on save
4. No other config needed.

**Audit workflow:**
1. Run `python scripts/obsidian/export_audit_queue.py --vault obsidian_vault`
2. Open `05_Annotations/Human Audits/` in Obsidian
3. Each note has a checklist; edit `label:`, `confidence:`, `audit_status: done` in the frontmatter panel
4. Run `python scripts/obsidian/import_audits.py --vault obsidian_vault --out artifacts/qrels_gold.jsonl`

**Dataview dashboard (copy into `08_Dashboards/Audit Progress.md`):**
```dataview
TABLE label, confidence, audit_status
FROM "05_Annotations/Human Audits"
WHERE audit_status != "done"
SORT file.mtime DESC
```

---

## 13. Full Command Sequence

```bash
# 1. Scaffold vault
python scripts/obsidian/init_vault.py --vault obsidian_vault

# 2. Export cards
python scripts/obsidian/export_dataset_cards.py \
  --corpus data/corpus/normalized/combined_corpus.jsonl --vault obsidian_vault
python scripts/obsidian/export_query_cards.py \
  --queries artifacts/benchmark_queries.jsonl --vault obsidian_vault

# 3. Build evidence
python scripts/eval/build_evidence.py \
  --pool reports/eval/benchmark_pool.jsonl \
  --queries artifacts/benchmark_queries.jsonl \
  --corpus data/corpus/normalized/combined_corpus.jsonl \
  --out artifacts/eval/pair_evidence.jsonl

# 4. Run labeling functions
python scripts/eval/run_labeling_functions.py \
  --evidence artifacts/eval/pair_evidence.jsonl \
  --out artifacts/eval/label_function_votes.jsonl

# 5. Optional: LLM judge
python scripts/eval/judge_candidates.py \
  --evidence artifacts/eval/pair_evidence.jsonl \
  --votes artifacts/eval/label_function_votes.jsonl \
  --config configs/judges/rubric_judge_v1.yaml \
  --out artifacts/eval/llm_judgments.jsonl

# 6. Build qrels tiers
python scripts/eval/build_qrels_from_votes.py \
  --evidence artifacts/eval/pair_evidence.jsonl \
  --votes artifacts/eval/label_function_votes.jsonl \
  --llm artifacts/eval/llm_judgments.jsonl \
  --out-gold artifacts/qrels_gold.jsonl \
  --out-silver artifacts/qrels_silver.jsonl \
  --out-bronze artifacts/qrels_bronze.jsonl \
  --audit-queue artifacts/eval/audit_queue.jsonl

# 7. Export + audit in Obsidian
python scripts/obsidian/export_audit_queue.py \
  --audit-queue artifacts/eval/audit_queue.jsonl --vault obsidian_vault
# ... human audits in Obsidian ...
python scripts/obsidian/import_audits.py \
  --vault obsidian_vault --out artifacts/qrels_gold.jsonl

# 8. Metrics per tier
python scripts/eval/compute_ir_metrics.py --qrels artifacts/qrels_gold.jsonl --qrels-tier gold
python scripts/eval/compute_ir_metrics.py --qrels artifacts/qrels_silver.jsonl --qrels-tier silver
python scripts/eval/compute_ir_metrics.py --qrels artifacts/qrels_bronze.jsonl --qrels-tier bronze
python scripts/eval/hard_negative_analysis.py \
  --qrels-gold artifacts/qrels_gold.jsonl \
  --qrels-silver artifacts/qrels_silver.jsonl \
  --qrels-bronze artifacts/qrels_bronze.jsonl
```
