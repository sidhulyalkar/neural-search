# Reproducibility Capsule — Neural Search

_Last updated: 2026-06-20_
_Canonical manifest: `reports/eval/current_artifact_manifest.json`_

This document specifies exactly which artifacts exist, what commands reproduce them, what their expected outputs are, and which artifacts are stale or unverified. A reviewer can run the "< 5 minute" quickstart below to verify the core system.

---

## Reviewer Quickstart (< 5 minutes)

```bash
# 1. Clone and install
git clone <repo_url> && cd neural-search
pip install -e ".[dev]"

# 2. Verify corpus exists and has the right row count
python -c "
from pathlib import Path
rows = sum(1 for l in open('data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl') if l.strip())
print(f'Corpus rows: {rows}')  # Expected: 7171
assert rows == 7171, f'Expected 7171, got {rows}'
print('PASS: corpus row count matches manifest')
"

# 3. Run all validation tests
python -m pytest tests/test_artifact_manifest.py tests/test_benchmark_safety_gate.py tests/test_typed_finding_extractor.py tests/test_brainknow_baseline.py -v

# 4. Verify safety gate (should show: gold_qrels_rows=0, safe=False or blockers listed)
python scripts/eval/benchmark_safety_gate.py --warn-only

# 5. Verify API starts (requires uvicorn)
# uvicorn apps.api.main:app --reload &
# curl http://localhost:8000/api/artifacts/current-manifest | python -m json.tool | head -20
```

All 4 commands should complete in under 5 minutes on a standard laptop without GPU.

---

## Canonical Artifacts

### Corpus

| Field | Value |
|---|---|
| Path | `data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl` |
| Row count | 7,171 |
| Unique source_ids | 7,121 |
| Unique dataset_ids (embedded slice) | 625 |
| SHA256 (head 16 chars) | `4aacecc3e9d37e17` |
| Evidence tier | **Tier A** — publishable |

> **Note**: `combined_corpus.jsonl` is a directory, not a file. The live corpus is the `full_corpus_v09.jsonl` file inside it.

To verify:
```bash
wc -l data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl
# Expected: 7171
```

### Knowledge Graph

| Field | Value |
|---|---|
| Path | `data/graph/neural_search_graph.real_corpus.json` |
| Nodes | 7,593 |
| Edges | 31,920 |
| Cross-dataset edges | 2,957 |
| Evidence tier | **Tier A** — publishable |

To verify:
```bash
python -c "
import json
g = json.load(open('data/graph/neural_search_graph.real_corpus.json'))
print(f'nodes={len(g[\"nodes\"])}, edges={len(g[\"edges\"])}')  # Expected: nodes=7593, edges=31920
"
```

To rebuild (slow, ~2h):
```bash
python scripts/rebuild_full_corpus_graph.py
```

### Qrels

| File | Rows | Status | Evidence tier |
|---|---|---|---|
| `artifacts/qrels_gold.jsonl` | 0 | Pending human annotation | D — do not cite |
| `artifacts/qrels_silver.jsonl` | 175 | LLM-generated, diagnostic | C — diagnostic only |
| `artifacts/qrels_bronze.jsonl` | 319 | LLM-generated, diagnostic | C — diagnostic only |
| `artifacts/field_state/adjudicated_qrels.jsonl` | 3 | Human-adjudicated smoke-test | B — workflow verified |

### Vector Index

| Field | Value |
|---|---|
| Current on-disk IDs | 625 (embedded slice only) |
| Full rebuild pending | Yes — needs full 7,171-record rebuild |
| Evidence tier | **B** — engineering validation |

> **WARNING**: The turbovec index covers 625 records. Retrieval results over the full 7,171-record corpus require a full index rebuild. Do not cite recall or NDCG over the full corpus without rebuilding.

### Literature Findings

| Field | Value |
|---|---|
| OpenAlex papers ingested | 255,940 |
| DOI-exact paper-dataset links | 168 |
| Title-fuzzy paper-dataset links | 225 |
| Operational findings estimate | ~12,050 |
| Evidence tier | **C** — operational estimate, not validated |

> **WARNING**: Do not cite `findings_tier1_ollama.jsonl` finding counts as precision-validated. Finding extraction precision is not yet audited. Use `scripts/eval/sample_findings_for_audit.py` to produce an audit sample.

---

## Commands to Reproduce Key Artifacts

### Artifact Manifest
```bash
python scripts/eval/build_artifact_manifest.py
# Output: reports/eval/current_artifact_manifest.json
```

### Qrels Progress Report
```bash
python scripts/eval/qrels_progress_report.py
# Output: reports/eval/qrels_progress_report.json + .md
```

### Benchmark Safety Gate
```bash
python scripts/eval/benchmark_safety_gate.py
# Output: reports/eval/benchmark_safety_gate_report.json
# Exit 1 if gold claims found while gold_qrels_rows == 0
```

### Finding Extraction Audit Sample
```bash
python scripts/eval/sample_findings_for_audit.py --n 100 --seed 42
# Output: reports/eval/finding_audit_template.csv (fill manually)
```

### Paper-Dataset Link Audit Sample
```bash
python scripts/eval/sample_paper_links_for_audit.py --n-each 50
# Output: reports/eval/paper_link_audit_template.csv (fill manually)
```

### BrainKnow Baseline Graph
```bash
python scripts/eval/brainknow_baseline.py --max-records 80000 --min-edge-weight 3
# Output: reports/eval/brainknow_baseline_graph.json + _summary.md
```

### Typed Finding KG Enrichment
```python
from neural_search.literature.typed_finding_extractor import enrich_finding
enriched = enrich_finding({"finding_text": "Theta increased during reward learning in hippocampus."})
# enriched now has: frequency_band, temporal_pattern, negation, spatial_frame, condition, signal_type, statistical_relation
```

---

## Stale Artifacts — Do Not Cite

| File | Issue |
|---|---|
| `reports/corpus_manifest.json` | Contains 10,404 (stale count); canonical count is 7,171 |
| `reports/eval/ndcg_report.json` | Based on silver qrels; not gold-adjudicated |
| `reports/real_corpus_v11_eval_report.json` | References old corpus size |
| `reports/eval/ablation_ladder_report.md` | Uses silver qrels as if gold; add disclaimer before citing |
| `docs/whitepaper/generated/source_distribution_table.tex` | Needs regeneration from v09 corpus |

These files are documented in `reports/eval/current_artifact_manifest.json` under `stale_reports`.

---

## Test Suite

Run the full test suite to verify the codebase:

```bash
python -m pytest tests/ -v --tb=short
```

Key test files for peer validation:

| Test file | What it verifies |
|---|---|
| `tests/test_artifact_manifest.py` | Manifest parses, all required keys present, counts > 0 |
| `tests/test_benchmark_safety_gate.py` | Safety gate flags NDCG claims when gold is empty |
| `tests/test_typed_finding_extractor.py` | 60 rule-based extraction tests (negation, frequency, temporal, etc.) |
| `tests/test_brainknow_baseline.py` | Co-occurrence baseline builds correctly |

---

## Known Limitations

1. **Retrieval is not yet benchmarked against gold qrels.** No NDCG, MRR, or Recall@k metrics should be cited until the gold annotation campaign is complete.

2. **Embedding index covers only 625 records.** The full 7,171-record corpus requires a rebuild before expanded-corpus retrieval is valid.

3. **Finding extraction precision is unaudited.** ~12K finding estimates are operational. Fill `reports/eval/finding_audit_template.csv` to validate.

4. **Affordance labels are metadata-only.** No file-level inspection has been performed. Affordance precision is unvalidated.

5. **BrainKnow comparison is qualitative.** The co-occurrence baseline is not a head-to-head retrieval comparison; it is a capability contrast illustrating what typed-relation graphs enable.

6. **Silver qrels are diagnostic.** Do not use `qrels_silver.jsonl` to compute retrieval metrics and present them as externally validated.

---

## Citation Note

When citing Neural Search:
- Cite **Tier A** artifacts (corpus size, KG statistics) without caveats.
- Cite **Tier B** claims (affordance types, API endpoints) with "engineering validation" language.
- Do **not** cite Tier C or D artifacts (silver qrels, NDCG, finding counts) as validation evidence.

The `reports/eval/benchmark_safety_gate.py` script automates this check.
