# Neural Search Frontend Playground

This document describes how to use the existing frontend to search datasets,
inspect neuro-judge evidence, and collect downstream retrieval feedback.

> **Important warnings:**
> - Neuro-judge labels are **silver labels** (RAG-grounded LLM outputs). They are NOT human-reviewed ground truth.
> - User feedback is a **downstream signal**, not a gold relevance label.
> - Expert audit is required before any scientific reporting.

---

## Location

The frontend lives at `apps/web/`. It is a Vite + React + TypeScript app.
The API lives at `apps/api/main.py` (FastAPI).

---

## How to Launch

### 1. Start the API

```bash
cd /path/to/neural-search
uvicorn apps.api.main:app --reload --port 8000
```

Or with demo mode (26-record fixture, no full corpus needed):

```bash
NEURAL_SEARCH_DEMO_MODE=1 uvicorn apps.api.main:app --reload --port 8000
```

### 2. Start the Frontend

> **WSL1 users**: npm does not work inside WSL1 due to inotify socket limitations.
> Run from Windows CMD/PowerShell, or upgrade to WSL2.

```bash
# WSL2 or native Linux/Mac:
cd apps/web
npm install
npm run dev
```

```powershell
# Windows CMD/PowerShell:
cd apps\web
npm install
npm run dev
```

The dev server starts at `http://localhost:5173`.

---

## How to Run Search

1. Navigate to `http://localhost:5173`
2. Type a natural-language query, e.g.:
   - "calcium imaging mouse hippocampus during spatial navigation"
   - "reversal learning with reward omission and trial outcomes"
3. Use the **Advanced** panel to add structured constraints (species, modality, task, data standard).
4. Press Enter or click Search.

Results page shows:
- Retrieval score breakdown
- Neuro-judge badge (label 0–3) with **"preliminary neuro-judge, not human gold"** watermark
- Evidence completeness bar
- Abstain warning if judge recommends it

---

## How to Inspect Neuro-Judge Evidence

Click **Evidence** on any result card to expand the evidence panel.

The panel shows:

| Section | Contents |
|---|---|
| Evidence for | Explicit signals supporting relevance |
| Evidence against | Hard negatives and mismatches |
| Missing information | Required dimensions not found |
| Required dimensions present | Dimensions with explicit support |
| Required dimensions missing | Dimensions absent or inferred only |
| Failure modes | Pre-screened warnings |
| Affordance evidence | Task/analysis affordance match status |
| Concept memory matches | RAG-retrieved concept signals |
| Linked papers | Associated publications |
| Raw evidence JSON | Full evidence packet (toggle) |

The badge colour indicates the judge provenance:
- **Amber**: preliminary neuro-judge (not expert-reviewed)
- **Green**: expert-audited consensus (if available)

---

## How to Submit Feedback

Each result card has a feedback panel (click **Feedback** or the thumbs icons).

Usefulness options:
- Useful
- Partially useful
- Not useful
- Unsure

Would-use-for-analysis options: Yes / Maybe / No

Reason tags (select all that apply):
`wrong_modality`, `wrong_species`, `wrong_region`, `missing_raw_data`,
`insufficient_metadata`, `good_match`, `interesting_reuse_candidate`,
`needs_manual_review`, `wrong_task`, `processed_only`, `low_evidence`

Click **Submit Feedback** to save. You can also:
- **Save** a dataset for later review
- **Export** the dataset card (JSON or Markdown)

---

## Where Feedback is Stored

All feedback is written to local JSONL artifact files:

| File | Contents |
|---|---|
| `artifacts/frontend/search_sessions.jsonl` | One record per search query |
| `artifacts/frontend/retrieval_feedback.jsonl` | One record per feedback submission |
| `artifacts/frontend/saved_datasets.jsonl` | One record per saved/exported dataset |

Each record includes `"provenance": "user_feedback_downstream_signal"` to clearly
label it as downstream usage data, not a gold relevance annotation.

---

## How to Compute Downstream Retrieval Success

After accumulating feedback:

```bash
python scripts/eval/compute_downstream_retrieval_success.py
```

Outputs:
- `reports/eval/downstream_retrieval_success.md` — human-readable summary
- `reports/eval/downstream_retrieval_success.json` — machine-readable

Key metrics:
- Usefulness distribution by rank, method, judge label, evidence completeness
- Save/export rate
- Would-use-for-analysis rate
- False-high (judge says relevant, user says not useful) and false-low counts

---

## How to Build a Feedback Audit Queue

```bash
python scripts/eval/build_feedback_audit_queue.py
```

Outputs:
- `artifacts/field_state/feedback_audit_queue.jsonl`
- `reports/eval/feedback_audit_queue.md`

The queue prioritises:
- High-rank results marked not useful
- Judge/user disagreements (false-high and false-low)
- Abstain-flagged results the user found useful
- Low evidence completeness datasets that were saved/exported

---

## How to Generate Dataset Organization Views

```bash
python scripts/eval/build_dataset_organization_views.py \
  --judgments artifacts/field_state/neuro_qrels_judgments_mock.jsonl
```

Outputs:
- `artifacts/field_state/dataset_organization_views.json`
- `reports/eval/dataset_organization_views.md`

Views group datasets into:
- Highly relevant (label=3) / Useful with caveats (label=2) / Weakly related (label=1) / Not relevant (label=0)
- Missing raw data / Missing species or modality metadata / Likely hard negatives
- Needs human audit / Raw-data suitable / Processed only
- High / low evidence completeness
- Modality-specific buckets (electrophysiology, calcium imaging, fMRI/MRI, EEG/MEG, behavior only)

---

## How to Run a Gemini Dry-Run Cost Estimate

The neuro-judge supports Gemini via `--backend gemini`:

```bash
python scripts/eval/run_neuro_qrels_judge.py \
  --packets artifacts/field_state/neuro_judge_evidence_packets.jsonl \
  --out /tmp/gemini_test.jsonl \
  --backend gemini \
  --gemini-model gemini-1.5-flash \
  --limit 5 \
  --dry-run-cost-estimate
```

Requires `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable.
If the key is absent or the `google-generativeai` package is not installed,
the judge gracefully skips and returns a mock judgment with `abstain_recommended=True`.

---

## How to Run Feedback Reranking Priors

```bash
python scripts/eval/build_feedback_rerank_priors.py
```

Outputs:
- `artifacts/field_state/feedback_rerank_priors.json`
- `reports/eval/feedback_rerank_priors.md`

This generates a transparent, rule-based weight table:
- Datasets repeatedly marked useful → small positive adjustment
- Datasets repeatedly flagged with `wrong_species`, `wrong_modality`, etc. → penalty
- Adjustment is capped at ±1.5 log-odds units

The frontend can toggle this layer on/off (visible in the sort/filter panel).

---

## Warnings and Limitations

1. **Neuro-judge labels are not human gold.** The 0–3 relevance labels are
   produced by an LLM with RAG evidence. They contain systematic errors,
   especially for datasets with sparse metadata or unusual experimental designs.

2. **Feedback is a downstream signal.** "Useful" from a user does not mean
   "relevant" in the scientific IR sense. It reflects whether a researcher
   found the dataset useful given their specific query and immediate need.

3. **Evidence completeness is a heuristic.** It measures the fraction of
   required dimensions that have explicit (not inferred) support in the
   metadata. It is not a ground-truth data quality score.

4. **Expert audit is required for publication.** Before reporting any metrics
   that rely on neuro-judge labels or feedback signals in a paper or evaluation
   report, the labels must be reviewed and adjudicated by domain experts.
