# Neural Search Repository Critique and Claude Continuation Plan

**Prepared for:** Sidharth Hulyalkar  
**Project:** Neural Search  
**Inputs reviewed:** uploaded repository zip `neural-search (5).zip`, uploaded whitepaper `neural_search_whitepaper.tex`, and the EBRAINS GUI search screenshot for `delay discounting`.

---

## 1. Executive Summary

Neural Search has crossed an important threshold: it is no longer just a concept sketch or a toy metadata search demo. The current repository contains enough architecture to become a real scientific retrieval system: ontology matching, structured metadata, graph construction, field embeddings, analysis affordance detection, source quality scoring, benchmark runners, hard-negative query sets, corpus ingestion, dataset cards, and whitepaper artifacts.

The main risk is now **not lack of ambition**. The main risk is **feature sprawl without decisive proof**.

The next phase should not add another shiny subsystem unless it helps answer the central scientific question:

> Can this system identify datasets that are experimentally reusable for a target scientific analysis, not merely topically similar?

That is the North Star. Everything else should orbit it like tiny methodological moons.

The EBRAINS screenshot is the perfect stress test. Searching `delay discounting` in EBRAINS surfaced results containing lexical variants of `delay`, including delayed reach-to-grasp and signal propagation delay. Those are not necessarily delay-discounting datasets. Neural Search should explicitly distinguish:

- true delay discounting / intertemporal choice
- delayed motor planning
- delay-period working memory
- signal propagation delay
- generic delayed stimulus paradigms

That kind of semantic and experimental disambiguation is where this project becomes valuable.

---

## 2. Current Repository Snapshot

The repository currently includes a broad set of modules and generated artifacts.

### 2.1 Strong technical foundations

Current module families include:

- `neural_search/search/`: hybrid search, semantic expansion, query intent, fusion, sparse retrieval, structured constraints, explanations, query building.
- `neural_search/ontology/`: ontology loading, validation, matching.
- `neural_search/graph/`: schema, builder, query, provenance, metapaths, semantic edges, paper linking, graph quality, graph reports.
- `neural_search/embeddings/`: providers, field embeddings, concept embeddings, fingerprints, semantic similarity, sentence-transformer hooks.
- `neural_search/evaluation/`: benchmark runners, ablations, hard-negative benchmarks, relevance labels, calibration, metadata robustness.
- `neural_search/affordances/` and `analysis_affordances.py`: analysis affordance detection.
- `neural_search/file_inspection/`: claims and file inspection scaffolding.
- `neural_search/intelligence/`: coverage, promotion, planning, review, active-learning-like scaffolding.
- `neural_search/cards/`, `notebooks/`, `reports/`, `qa/`, `recipes/`, `latent/`: downstream artifacts and future-facing latent search scaffolds.

This is a strong substrate. The project has “research infrastructure skeleton” energy rather than “weekend CRUD app” energy.

### 2.2 Current reported metrics

The repository includes generated reports such as:

- `data/eval/results/latest_eval_report.md`
- `reports/demo_v02/latest_eval_report.md`
- `data/reports/component_ablation_report.md`
- `data/reports/component_ablation_tuned.md`
- `docs/CLAIM_LEDGER.md`

Observed reported values include:

- 30 benchmark queries.
- Mean Precision@5 around **76.7% to 78.7%**, depending on report version.
- Mean Label Recall@10 around **87.8% to 88.5%**.
- MRR around **0.933 to 0.950** in the demo reports.
- Hard-negative benchmark scaffolding exists.

These are good demo-level numbers, but they should be treated as **prototype evidence**, not final scientific proof.

### 2.3 Important concern: component ablations are not yet convincing

The ablation reports show that some components currently have small, zero, or sometimes negative measurable impact. In particular:

- `no_semantic` sometimes improves MRR or NDCG.
- `no_affordance` often shows no measurable drop.
- `no_graph_context` often shows no measurable drop.
- `no_hard_negative` sometimes performs as well or better on aggregate metrics.

This does not mean these components are useless. It probably means the benchmark is still too small, too label-aligned, or not adversarial enough to reveal their value.

The right response is not to delete these components. The right response is to build a benchmark that can actually test them.

### 2.4 Repository hygiene concerns

The uploaded zip contains several categories of generated or local artifacts that should not be treated as canonical repository source:

- `.git/` directory included in the zip.
- `.pytest_cache/`, `.ruff_cache/`, `__pycache__/`, and `.pyc` files.
- `apps/web/node_modules/` and built frontend artifacts.
- `dist/` wheel and source distribution artifacts.
- several zip packages and handoff kits inside the repo root.
- generated reports tracked or present under `data/reports/`.
- generated benchmark outputs under `data/eval/results/`.
- generated raw data under `data/raw/`.

The `.gitignore` already excludes many of these, but the current working tree and exported zip still include them. Claude should clean the repo export and, where necessary, remove generated artifacts from Git tracking.

### 2.5 Documentation sprawl

The repo has many roadmap and task documents, including `TASK_7` through `TASK_37`, multiple next-phase plans, architecture docs, handoff docs, roadmap docs, and older package artifacts. This is not fatal, but it will make future work harder unless the docs are consolidated.

Recommended doc hierarchy:

```text
README.md
CLAUDE.md or AGENTS.md
/docs/
  PROJECT_VISION.md
  CURRENT_SYSTEM_MAP.md
  ARCHITECTURE.md
  EVALUATION.md
  CLAIM_LEDGER.md
  ROADMAP.md
  CLEANUP_LOG.md
  whitepaper/
    neural_search_whitepaper.tex
/archive_docs/
  older task prompts and historical handoff docs
```

Do not delete old docs blindly. Move superseded plans into an archive folder with a short index.

---

## 3. Core Strategic Critique

### 3.1 The project must become an experimental reusability engine

The current system still risks being interpreted as “metadata search plus ontology tags.” That is not enough.

The defensible thesis is:

> Neural Search identifies datasets that are scientifically reusable for a target analysis by combining structured constraints, ontology-aware metadata, provenance-backed claims, knowledge graph linkages, analysis affordance validators, and learned embeddings.

This is different from EBRAINS, DANDI, OpenNeuro, or Google Dataset Search. Those systems help users find described resources. Neural Search should help users assess whether a dataset can support a specific experiment, model, or analysis.

Example:

A weak system answers:

> Here are datasets that mention delay.

A strong Neural Search answer says:

> This dataset supports delay-discounting model fitting because it contains trial-level choices, reward magnitudes, delay durations, outcomes, subject/session IDs, and behavioral timestamps. This other dataset is rejected because `delay` refers to motor preparation, not intertemporal choice.

That is the whole treasure chest.

### 3.2 The system needs claim-level provenance, not only dataset-level tags

Dataset-level labels are too coarse. Every important assertion should become a provenance-backed claim.

Instead of storing only:

```yaml
has_task: delay_discounting
has_modality: fiber_photometry
has_behavior: choice
```

Store claim records such as:

```yaml
claim_id: claim_000001
subject_id: dataset:DANDI:000000
predicate: has_task
object_id: task:delay_discounting
object_label: Delay discounting
source_type: paper_methods_section
source_ref: paper:doi:...
evidence_text: "subjects chose between immediate and delayed rewards"
extractor: rule.delay_discounting.v2
confidence: 0.91
review_status: unreviewed
created_at: 2026-05-27
```

This allows result cards to say where each match came from: archive metadata, paper methods, README, file inspection, code, inferred ontology synonym, or human review.

### 3.3 Analysis affordances should become executable validators

The affordance layer is currently one of the most promising pieces. It should become stricter and more executable.

Instead of saying:

> This dataset supports Q-learning.

The system should say:

> This dataset supports Q-learning because it has trial-level choices, actions, rewards, outcomes, and task states. Reward timing is absent, so temporal-difference model confidence is medium.

Each affordance should have:

```yaml
affordance_id: q_learning_modeling
required_evidence:
  - trials.choice
  - trials.reward
  - trials.outcome
  - trials.state_or_stimulus
  - subject_id
  - session_id
optional_evidence:
  - reaction_time
  - reward_timestamp
  - neural_activity_aligned_to_trial
hard_blockers:
  - no_trial_table
  - no_choice_variable
  - no_reward_or_outcome
confidence_rules:
  high: all required fields plus timestamps
  medium: required fields but incomplete timing
  low: only description-level evidence
```

This is how Neural Search becomes more than a nice tagger.

### 3.4 The benchmark should punch back

Current metrics are not yet enough. A 30-query demo benchmark is useful, but it does not prove scientific reuse.

The next benchmark should be called something like:

```text
reusability_gold_v1
```

It should include:

- 100 queries eventually, but start with 25.
- expert-reviewed positives and hard negatives.
- graded relevance from 0 to 3.
- explicit “must-have” variables and “hard blocker” variables.
- ambiguity traps.
- analysis-affordance queries.
- cross-modal and cross-species queries.
- natural language queries.

The key is to evaluate whether a result is **experimentally reusable**, not merely whether it has overlapping labels.

### 3.5 Embeddings need to prove their value field by field

Do not assume embeddings are helping because they sound modern. The current ablation reports suggest the semantic component is not yet clearly improving retrieval.

Instead of one generic embedding score, split embeddings by scientific field:

```text
title_embedding
abstract_embedding
methods_embedding
task_embedding
variable_schema_embedding
affordance_embedding
paper_embedding
dataset_card_embedding
```

Then test which embeddings help which query types.

The best embedding work is not merely “use a bigger model.” It is:

> make the embedding target represent experimental structure.

A dataset embedding should not only encode “mouse reward task.” It should encode whether the dataset contains usable choices, rewards, events, timing, neural data, and analysis-ready metadata.

### 3.6 The graph must stay conservative

Knowledge graphs are useful, but they can quietly turn into spiderweb cannons. Broad edges like `mouse -> cortex -> behavior -> reward` can connect everything to everything.

Every edge should carry:

```yaml
edge_type: dataset_has_task
source_type: file_inspection | archive_metadata | paper_methods | paper_abstract | inferred_synonym | broad_taxonomy
confidence: 0.0-1.0
evidence_ref: claim_id
```

Recommended confidence defaults:

```yaml
file_inspection: 0.95
explicit_archive_metadata: 0.90
paper_methods: 0.85
readme_or_code: 0.75
paper_abstract: 0.65
inferred_synonym: 0.55
broad_taxonomy: 0.35
```

Search should reward specific metapaths, such as:

```text
dataset -> trial_variable -> analysis_affordance
dataset -> task -> affordance
dataset -> paper_methods_claim -> task
```

It should penalize weak paths such as:

```text
dataset -> broad_behavior -> broad_task
dataset -> same_species_only -> query
```

### 3.7 EBRAINS should be used as a baseline and source, not framed as an enemy

EBRAINS is a curated FAIR metadata and KG infrastructure. Neural Search should not position itself as “EBRAINS but smaller.”

Better framing:

> EBRAINS, DANDI, OpenNeuro, BICCN, BIDS, NWB, and related platforms make datasets discoverable and reusable at the metadata level. Neural Search builds an experimental retrieval intelligence layer above them, reasoning over metadata, provenance, literature, code, ontology links, file structure, and embeddings to identify which datasets are reusable for a target analysis.

The EBRAINS `delay discounting` screenshot should become a benchmark case study.

---

## 4. Priority Deliverables

The next Claude sprint should produce the following deliverables in order.

### Deliverable 1: Repository cleanup and canonical project map

**Goal:** remove clutter, reduce confusion, and make the project easier for future agents and humans to navigate.

Tasks:

1. Create or update `docs/CURRENT_SYSTEM_MAP.md` with the actual current modules and their responsibilities.
2. Create `docs/CLEANUP_LOG.md` documenting what was moved, removed, or archived.
3. Add an `archive_docs/` directory for historical task prompts and old handoff packages.
4. Remove local generated artifacts from the repository working tree, if tracked:
   - `__pycache__/`
   - `*.pyc`
   - `.pytest_cache/`
   - `.ruff_cache/`
   - `apps/web/node_modules/`
   - `apps/web/dist/`
   - `dist/`
   - root-level `*.zip`
   - generated data reports where appropriate
5. Update `.gitignore` if any generated paths are missing.
6. Add a repo export script that creates a clean zip without `.git`, caches, node modules, build artifacts, and generated reports.

Suggested script:

```bash
mkdir -p scripts
cat > scripts/export_clean_repo.sh <<'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-neural-search-clean.zip}
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
rm -f "$OUT"
git archive --format=zip --output="$OUT" HEAD
printf "Wrote %s\n" "$OUT"
SCRIPT
chmod +x scripts/export_clean_repo.sh
```

Acceptance criteria:

- `git status --short` is understandable.
- No generated caches or node modules are included in repo exports.
- All canonical docs are easy to find.
- Historical prompts are archived, not mixed into the main project surface.

---

### Deliverable 2: Reusability claim schema

**Goal:** make provenance-backed claims the atomic unit of search and explanation.

Create:

```text
neural_search/core/claims.py
```

or extend:

```text
neural_search/file_inspection/claims.py
```

Recommended models:

```python
from enum import Enum
from pydantic import BaseModel, Field

class EvidenceSourceType(str, Enum):
    ARCHIVE_METADATA = "archive_metadata"
    PAPER_ABSTRACT = "paper_abstract"
    PAPER_METHODS = "paper_methods"
    README = "readme"
    CODE = "code"
    FILE_INSPECTION = "file_inspection"
    HUMAN_REVIEW = "human_review"
    INFERRED_ONTOLOGY = "inferred_ontology"

class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    TRUSTED = "trusted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"

class ReusabilityClaim(BaseModel):
    claim_id: str
    subject_id: str
    predicate: str
    object_id: str
    object_label: str | None = None
    source_type: EvidenceSourceType
    source_ref: str | None = None
    evidence_text: str | None = None
    extractor: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    created_at: str | None = None
```

Acceptance criteria:

- Existing extracted labels can be represented as claims.
- Claims can be serialized to JSONL.
- Claims are usable by graph edges and result explanations.
- Every top search result can cite at least one claim.

---

### Deliverable 3: Executable affordance validators

**Goal:** convert analysis affordances from loose labels into inspectable requirements.

Create or upgrade:

```text
neural_search/affordances/validators.py
config/affordance_rubric.yaml
```

Start with these six validators:

1. `delay_discounting_modeling`
2. `q_learning_modeling`
3. `choice_decoding`
4. `motor_decoding`
5. `trial_aligned_neural_analysis`
6. `cross_session_generalization`

Each validator should output:

```yaml
affordance_id: delay_discounting_modeling
status: supported | partial | unsupported
confidence: 0.0-1.0
matched_requirements: []
missing_requirements: []
hard_blockers: []
evidence_claim_ids: []
notes: "..."
```

Acceptance criteria:

- Validators can run over normalized dataset records and claims.
- Missing requirements are explicit.
- Hard blockers prevent high-confidence matches.
- Tests include true positives and false positives.

---

### Deliverable 4: Query sense disambiguation

**Goal:** avoid EBRAINS-style lexical false positives.

Create:

```text
neural_search/search/sense_disambiguation.py
config/query_senses.yaml
```

Initial overloaded concepts:

```yaml
delay:
  senses:
    delay_discounting:
      positive_terms:
        - delay discounting
        - temporal discounting
        - intertemporal choice
        - delayed reward
        - immediate reward
        - impulsive choice
      required_context_any:
        - reward
        - choice
        - value
        - magnitude
      negative_terms:
        - signal propagation delay
        - synaptic delay
        - reach-to-grasp delay
        - delay period working memory
        - delayed stimulus onset
    motor_delay:
      positive_terms:
        - delayed reach
        - instructed delay
        - motor planning
    signal_delay:
      positive_terms:
        - propagation delay
        - conduction delay
        - latency
```

Acceptance criteria:

- Query parser recognizes `delay discounting` as intertemporal choice, not generic delay.
- Hard negatives are penalized or rejected when the wrong sense is detected.
- The EBRAINS screenshot examples can be represented as hard-negative tests.

---

### Deliverable 5: Reusability Gold v1 benchmark

**Goal:** build a benchmark that tests scientific usefulness, not just label overlap.

Create:

```text
data/eval/reusability_gold_v1.yaml
data/eval/reusability_gold_v1_labels.jsonl
docs/REUSABILITY_GOLD_V1.md
```

Start with 25 queries, then expand to 100.

Required categories:

1. Ambiguous construct queries.
2. Analysis-affordance queries.
3. Cross-modal/cross-species queries.
4. Natural-language messy queries.
5. Exact lookup or paper/dataset linkage queries.

Example query spec:

```yaml
id: rg_v1_001
query: "Find datasets suitable for fitting delay discounting models from trial-level behavior."
intent: analysis_reusability
constructs:
  - delay_discounting
must_have:
  - trial_id
  - choice
  - reward_magnitude
  - delay_duration
  - outcome
should_have:
  - reaction_time
  - neural_activity
  - timestamps
hard_negative_senses:
  - motor_delay
  - signal_propagation_delay
  - working_memory_delay
relevance:
  DEMO_DELAY_DISCOUNTING: 3
  DEMO_DOPAMINE_PHOTOMETRY: 2
  DEMO_WORKING_MEMORY_EPHYS: 0
  FTRACT_SIGNAL_DELAY_EXAMPLE: 0
notes: "True delay discounting requires reward-delay tradeoff variables."
```

Metrics should include:

- Precision@k.
- MRR.
- NDCG@k.
- graded relevance.
- hard-negative violation rate.
- missing-requirement penalty.
- claim support rate.
- affordance support accuracy.

Acceptance criteria:

- At least 25 query specs.
- At least 10 hard-negative traps.
- At least 6 affordance-driven queries.
- Search result explanations include matched and missing requirements.

---

### Deliverable 6: Candidate generation plus scientific reranking

**Goal:** replace weighted soup with a clearer retrieval architecture.

Target flow:

```text
Query
  -> intent parser
  -> sense disambiguation
  -> candidate generators
      - sparse/BM25
      - ontology expansion
      - structured filters
      - graph neighborhood
      - field embeddings
      - paper/dataset links
  -> union candidates
  -> scientific reranker
      - construct compatibility
      - variable compatibility
      - affordance support
      - modality/species/region constraints
      - provenance confidence
      - missingness penalties
      - hard-negative violations
  -> evidence-backed result cards
```

Acceptance criteria:

- Candidate-generation traces are available for debugging.
- Reranker feature contributions are visible.
- Each final result has a reasoned score breakdown.
- Ablations can test candidate generators and reranker features separately.

---

### Deliverable 7: Evidence-backed result cards

**Goal:** make search results scientifically inspectable.

Each result should include:

```yaml
dataset_id: DEMO_DELAY_DISCOUNTING
rank: 1
score: 0.92
reusability_status: supported
matched_constructs:
  - delay_discounting
matched_affordances:
  - delay_discounting_modeling
matched_requirements:
  - choice
  - reward_magnitude
  - delay_duration
  - outcome
missing_requirements:
  - reaction_time
hard_negative_flags: []
evidence:
  - claim_id: claim_001
    source_type: paper_methods
    summary: "Methods describe immediate vs delayed reward choices."
  - claim_id: claim_002
    source_type: file_inspection
    summary: "Trial table contains delay_duration and reward_magnitude."
explanation: "This dataset directly supports delay-discounting model fitting..."
```

Acceptance criteria:

- API search responses expose these fields.
- CLI output can print compact evidence cards.
- Benchmark reports show missing requirements and hard-negative flags.

---

### Deliverable 8: Whitepaper and claim ledger synchronization

**Goal:** keep reputation high by preventing inflated claims.

Update:

```text
docs/CLAIM_LEDGER.md
docs/WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md
docs/whitepaper/neural_search_whitepaper.tex
```

Every quantitative claim should have:

```yaml
claim: "Full system achieves 78.7% Precision@5 on demo_v02."
source_artifact: data/reports/component_ablation_tuned.md
replication_command: "python -m neural_search.evaluation.run_benchmark --suite demo_v02"
corpus_version: demo_v02
query_count: 30
status: prototype_validated
limitations:
  - small corpus
  - single annotator
  - label-aligned benchmark
```

Acceptance criteria:

- No whitepaper metric lacks a source artifact.
- Corpus size is consistent across paper, README, and reports.
- Recall@10 is distinguished from Label Recall@10.
- Embedding claims are caveated if ablations do not yet show improvement.

---

## 5. Concrete Cleanup Plan for Claude

Claude should execute cleanup as a sequence of small commits.

### Phase A: Inspect and freeze current state

Commands:

```bash
git status --short
git branch --show-current
python --version
python -m pip install -e ".[dev]"
pytest -q || true
ruff check neural_search tests || true
```

Do not spend hours fixing unrelated snapshot tests yet. The point is to capture the starting state.

Create:

```text
docs/CLEANUP_LOG.md
```

Include:

- current branch
- Python version
- test status
- lint status
- known failures
- files that appear generated or noncanonical

### Phase B: Remove generated/local artifacts

Commands:

```bash
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
rm -rf .pytest_cache .ruff_cache .mypy_cache
rm -rf apps/web/node_modules apps/web/dist
rm -rf dist build *.egg-info
rm -f ./*.zip
```

If any of these are tracked:

```bash
git rm -r --cached apps/web/node_modules apps/web/dist dist || true
git rm -r --cached data/eval/results data/reports data/raw || true
git rm --cached ./*.zip || true
git rm -r --cached '**/__pycache__' || true
git rm --cached '**/*.pyc' || true
```

Then update `.gitignore` if needed.

### Phase C: Archive superseded docs

Create:

```text
archive_docs/README.md
```

Move old handoff/task docs only if they are clearly superseded. Do not move canonical docs needed by current commands.

Suggested archival candidates:

- old generated handoff package directories
- old zip-derived package docs
- duplicated roadmap docs
- older v0.3/v0.4/v0.5 instruction sets if superseded by current plan

Keep in main docs:

- `CLAIM_LEDGER.md`
- `CURRENT_SYSTEM_MAP.md`
- `ARCHITECTURE.md` or one canonical architecture doc
- `EVALUATION.md`
- `ROADMAP.md`
- `REUSABILITY_GOLD_V1.md`
- `CLEANUP_LOG.md`
- whitepaper folder

### Phase D: Add clean export script

Add:

```text
scripts/export_clean_repo.sh
```

Then test:

```bash
bash scripts/export_clean_repo.sh /tmp/neural-search-clean.zip
unzip -l /tmp/neural-search-clean.zip | grep -E 'node_modules|__pycache__|\.git/|\.pyc|dist/' && echo "bad export" || echo "clean export"
```

### Phase E: Commit cleanup

```bash
git status --short
git add .
git commit -m "chore: clean repository artifacts and consolidate project docs"
```

---

## 6. Full Claude Continuation Prompt

Copy and paste this into Claude Code.

```text
You are continuing work on the Neural Search repository. Your goal is not to add broad new feature sprawl. Your goal is to make the project scientifically sharper, cleaner, and more credible as a provenance-aware experimental dataset retrieval system.

North Star:
Neural Search should identify datasets that are experimentally reusable for a target scientific analysis, not merely topically similar. Every major implementation decision should support that goal.

Context:
The current repo already has ontology matching, search modules, graph infrastructure, embeddings, analysis affordances, source quality, benchmarks, hard-negative queries, dataset cards, and whitepaper docs. However, the repo has become cluttered with generated artifacts, many overlapping task docs, and benchmarks that are not yet hard enough to prove the value of embeddings, graph features, hard-negative filtering, or affordance detection. Current ablations show some components have minimal or inconsistent measurable impact, so the next step is to build stronger evaluation and stricter evidence-backed retrieval.

Primary deliverables, in order:

1. Repository cleanup and canonical project map
- Inspect git status, tests, lint, and generated artifacts.
- Create or update docs/CLEANUP_LOG.md.
- Remove local/generated artifacts from the working tree: __pycache__, *.pyc, .pytest_cache, .ruff_cache, node_modules, frontend dist, Python dist/build artifacts, root-level zip packages, and generated reports where appropriate.
- Ensure .gitignore excludes generated artifacts.
- Add scripts/export_clean_repo.sh using git archive so future zips are clean.
- Consolidate docs: keep canonical docs in docs/, move clearly superseded handoff/task docs into archive_docs/ with an index. Do not delete information blindly.

2. Reusability claim schema
- Implement or consolidate a ReusabilityClaim model, ideally in neural_search/core/claims.py or neural_search/file_inspection/claims.py.
- Claims should include claim_id, subject_id, predicate, object_id, object_label, source_type, source_ref, evidence_text, extractor, confidence, review_status, and timestamps.
- Existing extracted labels should be representable as claims.
- Claims should serialize to JSONL and support graph edges and result explanations.

3. Executable affordance validators
- Upgrade affordance detection from loose tags to validators with required evidence, optional evidence, missing requirements, hard blockers, confidence, and supporting claim IDs.
- Start with: delay_discounting_modeling, q_learning_modeling, choice_decoding, motor_decoding, trial_aligned_neural_analysis, cross_session_generalization.
- Add tests with true positives, partial support, unsupported records, and false-positive decoys.

4. Query sense disambiguation
- Add config/query_senses.yaml and neural_search/search/sense_disambiguation.py.
- Start with overloaded terms: delay, reward, choice, memory, spike, signal, model.
- The key demo case is delay discounting: distinguish intertemporal choice from motor delay, working-memory delay, and signal propagation delay.
- Integrate sense penalties or hard-negative flags into search/reranking.

5. Reusability Gold v1 benchmark
- Create data/eval/reusability_gold_v1.yaml and docs/REUSABILITY_GOLD_V1.md.
- Start with 25 queries across ambiguity, analysis affordance, cross-modal/cross-species, natural-language, and exact lookup categories.
- Include graded relevance, must-have variables, should-have variables, and hard-negative senses.
- Add or adapt a runner/report path so the benchmark reports hard-negative violation rate, claim support rate, missing requirement penalties, and graded NDCG.

6. Retrieval architecture cleanup
- Separate candidate generation from scientific reranking.
- Candidate generators may include sparse/BM25, ontology expansion, structured filters, graph neighborhood, field embeddings, and paper/dataset links.
- Reranker should use construct compatibility, variable compatibility, affordance support, modality/species/region constraints, provenance confidence, missingness penalties, and hard-negative violations.
- Preserve traceability: each result should expose score components and evidence.

7. Evidence-backed result cards
- Search responses should include matched constructs, matched affordances, matched requirements, missing requirements, hard-negative flags, evidence claim IDs, and a concise explanation.
- Update CLI/API/report outputs where practical.

8. Whitepaper and claim ledger sync
- Update docs/CLAIM_LEDGER.md and docs/WHITEPAPER_IMPLEMENTATION_ALIGNMENT.md.
- Do not overclaim. Distinguish Recall@10 from Label Recall@10.
- Ensure corpus size and metric values are consistent across README, whitepaper, and reports.
- If embeddings or graph features are not yet clearly improving metrics, state that honestly and frame them as under active evaluation.

Workflow rules:
- Use small commits.
- Prefer tests around new schemas, validators, disambiguation, and benchmark parsing.
- Do not spend the whole session chasing unrelated snapshot failures unless they block the deliverables.
- Avoid frontend work unless required to expose evidence fields.
- Do not add a new subsystem unless it directly supports experimental reusability ranking or evidence-backed explanation.

Suggested commit sequence:
1. chore: clean generated artifacts and add export script
2. docs: consolidate project map and cleanup log
3. feat: add reusability claim schema
4. feat: add executable affordance validators
5. feat: add query sense disambiguation
6. test: add reusability gold v1 benchmark fixtures
7. feat: add evidence-backed reranking/result explanations
8. docs: sync whitepaper claims and claim ledger

Acceptance criteria:
- Clean repo export contains no .git, node_modules, caches, pyc files, build artifacts, or root-level zip packages.
- Tests pass for new claim schema, affordance validators, sense disambiguation, and benchmark loading.
- At least 25 Reusability Gold v1 queries exist.
- Delay discounting query rejects or penalizes motor delay, signal propagation delay, and working-memory delay false positives.
- Top search results expose evidence claims and missing requirements.
- Claim ledger honestly reflects what is implemented, prototype-validated, partially implemented, proposed, or not started.
```

---

## 7. Suggested Acceptance Tests

Claude should add or update tests such as:

```text
tests/test_reusability_claims.py
tests/test_affordance_validators.py
tests/test_query_sense_disambiguation.py
tests/test_reusability_gold_benchmark.py
tests/test_evidence_result_cards.py
```

Minimum test cases:

### Delay discounting sense test

```python
def test_delay_discounting_sense_excludes_signal_delay():
    parsed = disambiguate_query("delay discounting datasets with choices and rewards")
    assert parsed.primary_sense == "delay_discounting"
    assert "signal_propagation_delay" in parsed.negative_senses
```

### Affordance validator true positive

```python
def test_delay_discounting_validator_supported_when_required_variables_present():
    result = validate_affordance(record_with_choice_reward_delay_outcome, "delay_discounting_modeling")
    assert result.status == "supported"
    assert result.confidence >= 0.8
    assert not result.hard_blockers
```

### Affordance validator hard blocker

```python
def test_q_learning_validator_blocks_missing_choice():
    result = validate_affordance(record_with_rewards_but_no_choices, "q_learning_modeling")
    assert result.status == "unsupported"
    assert "no_choice_variable" in result.hard_blockers
```

### Evidence card test

```python
def test_search_result_contains_claim_backed_explanation():
    result = search("delay discounting", limit=1).results[0]
    assert result.evidence
    assert result.matched_requirements
    assert hasattr(result, "missing_requirements")
```

---

## 8. What Not to Prioritize Yet

Do not spend the next sprint on:

- UI polish.
- more architecture diagrams.
- more broad roadmap docs.
- larger whitepaper claims before the benchmark improves.
- adding more generic embedding providers without per-field ablations.
- expanding the corpus blindly without better labels and reusability criteria.
- building a full learned reranker before the hand-engineered features and labels are trustworthy.

The project does not need more fog machines. It needs a sharper lantern.

---

## 9. Final Recommendation

The most valuable next phase is:

> Build a stricter, evidence-backed reusability benchmark and make the search engine explain why a dataset can or cannot support a target scientific analysis.

If the next version can convincingly handle the `delay discounting` case better than a lexical KG search, while showing claim-backed evidence and missing requirements, you will have a compelling demo and a much stronger whitepaper.

Your project should not merely say:

> similar datasets found.

It should say:

> reusable datasets found, with evidence, caveats, and analysis-readiness verdicts.

That is the thing worth building.
