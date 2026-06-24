# Peer Validation Protocol — Neural Search

_Last updated: 2026-06-20_

This document specifies how a reviewer can independently verify the claims made in the Neural Search whitepaper and evaluate the system's retrieval quality, extraction quality, and affordance coverage.

---

## 1. System Summary

Neural Search is a **dataset retrieval system** for neuroscience: it indexes research datasets (currently 7,171 records) and retrieves them in response to natural-language queries. The core claim is that structured, multi-dimensional retrieval (region × species × modality × task × affordance) outperforms keyword search for researchers seeking reusable datasets.

---

## 2. What Can and Cannot Be Verified Today

### Tier A — Publishable-grade evidence (verify these)
| Claim | Artifact | How to verify |
|---|---|---|
| 7,171 corpus records | `data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl` | `wc -l` or Python `len(list(open(...)))` |
| 7,593 nodes / 31,920 edges in KG | `data/graph/neural_search_graph.real_corpus.json` | Load JSON, check `len(nodes)` and `len(edges)` |
| 168 DOI-exact + 225 fuzzy paper links | `artifacts/literature/paper_dataset_links.jsonl` | `grep "doi_exact" \| wc -l` |
| API returns results in <500ms (p95) | Run `make test-api` or open `/api/search?query=hippocampus` | Time 10 queries |
| 21 analysis affordance types | `neural_search/analysis_affordances.py` | Count `AFFORDANCE_REGISTRY` entries |

### Tier B — Engineering validation (inspect these)
| Claim | Artifact | Status |
|---|---|---|
| Hybrid scoring (ontology + dense + graph) | `neural_search/` scorer modules | Code review |
| LLM-judge pipeline operational | `scripts/eval/` | Inspect scripts |
| Extraction pipeline: theta/gamma/negation | `neural_search/literature/typed_finding_extractor.py` | Run tests |

### Tier C — Silver diagnostics (do NOT cite as retrieval metrics)
| Artifact | What it is | What it is not |
|---|---|---|
| `artifacts/qrels_silver.jsonl` (175 rows) | LLM-generated relevance pairs | Human-adjudicated ground truth |
| `artifacts/literature/findings_tier1_normalized.jsonl` (~12K) | LLM-extracted findings | Expert-validated findings |
| `reports/eval/brainknow_baseline_summary.md` | Co-occurrence baseline | Publication-ready comparison |

### Tier D — Pending (do NOT cite)
| Item | Status |
|---|---|
| Gold qrels | 0 rows — human annotation campaign not yet run |
| NDCG / MRR retrieval metrics | Blocked on gold qrels |
| Finding extraction precision | Manual audit template exists; not yet filled |
| Affordance precision | Audit template exists; not yet filled |

---

## 3. Benchmark Design

### 3.1 Query Design

Benchmark queries are curated to exercise multi-dimensional retrieval. Three classes:

1. **Field-State queries** — specify a brain region + recording modality + task (e.g., "Neuropixels recordings in hippocampus during spatial navigation")
2. **Affordance queries** — specify an intended analysis (e.g., "dataset suitable for Q-learning model fitting")
3. **Literature-bridged queries** — specify a finding from a paper (e.g., "datasets from papers reporting theta-gamma coupling during memory")

### 3.2 Qrel Annotation Guide

When the human annotation campaign is run, annotators should follow these rules:

| Label | Meaning |
|---|---|
| 2 (highly relevant) | Dataset directly supports the stated analysis; metadata is sufficient to begin coding |
| 1 (relevant) | Dataset is likely useful; some metadata is missing or requires manual verification |
| 0 (not relevant) | Dataset does not match at least 2 of the 4 required dimensions (region, species, task, modality) |

**Adjudication**: for pairs with label disagreement ≥ 1, a third annotator decides. Agreement statistics should be reported (Cohen's κ target ≥ 0.6).

**Minimum for a publishable benchmark**: 100 queries × 15 qrels/query = 1,500 annotated pairs from ≥ 2 annotators.

### 3.3 Priority Qrel Pool

The current silver pool (`artifacts/qrels_silver.jsonl`) contains 175 LLM-generated pairs. Use `scripts/eval/qrels_progress_report.py` to see the top 50 priority pairs sorted by score, ready for human annotation.

---

## 4. Finding Extraction Audit

### Purpose
Validate LLM extraction precision before citing finding counts in the whitepaper.

### How to Run
```bash
python scripts/eval/sample_findings_for_audit.py
# Outputs: reports/eval/finding_audit_template.csv
```

### How to Fill
- Open `reports/eval/finding_audit_template.csv`
- Look up each `paper_id` on OpenAlex: `https://openalex.org/works/<paper_id>`
- For each row, fill: `human_correct` (TRUE/FALSE/PARTIAL), `error_type`, `notes`

### Decision Criteria
| Precision | Whitepaper implication |
|---|---|
| ≥ 80% | Cite finding counts with "LLM-extracted, ~80% precision" |
| 60–80% | Report precision and failure modes; use for qualitative analysis only |
| < 60% | Do not cite finding counts; re-run extraction with improved prompt |

---

## 5. Paper-Dataset Link Audit

### Purpose
Validate whether paper-dataset links are correct before citing link counts.

### How to Run
```bash
python scripts/eval/sample_paper_links_for_audit.py
# Outputs: reports/eval/paper_link_audit_template.csv
```

### Expected Precision
| Link type | Expected precision |
|---|---|
| DOI-exact | ≥ 95% |
| Title-fuzzy | 60–85% |
| Not-found (false negative rate) | unknown — audit determines this |

---

## 6. Affordance Validation Audit

### Purpose
Determine whether affordance labels ("this dataset supports Q-learning") are correct.

### How to Run
```bash
python scripts/eval/sample_affordances_for_audit.py
# Outputs: reports/eval/affordance_audit_template.csv
```

### Decision Criteria
| Precision | Action |
|---|---|
| ≥ 80% | Affordance labels acceptable for whitepaper claim with caveat |
| 60–80% | Report precision with specific failure modes |
| < 60% | Re-tune affordance detector; do not cite |

---

## 7. User Study Protocol (Optional, Post-Benchmark)

For a user study comparing Neural Search to keyword search:

1. Recruit 10–20 researchers in systems/cognitive neuroscience
2. Assign 3 finding-reuse tasks each: find a dataset for a stated analysis goal
3. Half start with keyword search (e.g., DANDI web search), half with Neural Search
4. Measure: time to first relevant result, number of results inspected, self-reported usefulness (1–5)
5. Switch condition after each task

**Minimum for publication**: 60 task completions (10 participants × 6 tasks) with counterbalanced condition order.

---

## 8. Benchmark Safety Gate

Before claiming any NDCG/MRR/Recall metrics, run:

```bash
python scripts/eval/benchmark_safety_gate.py
```

The gate will BLOCK publication of retrieval metrics if `gold_qrels_rows == 0`. This ensures no silver-based metrics are cited as gold-standard.

---

## 9. BrainKnow Baseline Comparison

To reproduce the co-occurrence baseline comparison cited in the whitepaper:

```bash
python scripts/eval/brainknow_baseline.py
# Outputs: reports/eval/brainknow_baseline_graph.json
#          reports/eval/brainknow_baseline_summary.md
```

The comparison is qualitative: the baseline cannot answer "which dataset can I download to test this claim?" — Neural Search can. See `reports/eval/brainknow_baseline_summary.md` for the full comparison table.

---

## 10. Contact

Sid Hulyalkar — sid.soccer.21@gmail.com

For access to artifacts too large to include in this repository (full embedding index, OpenAlex snapshot), contact the author.
