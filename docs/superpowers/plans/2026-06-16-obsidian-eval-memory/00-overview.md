# Obsidian Eval Memory + Weak Supervision — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace manual annotation with a weak-supervision pipeline (deterministic labeling functions + optional LLM judge) that produces gold/silver/bronze qrels, surfacing only uncertain examples to a human auditor via an Obsidian vault.

**Architecture:** Evidence is extracted from the existing corpus and query files, run through 13 deterministic labeling functions, optionally judged by an LLM, then aggregated into tiered qrels. The Obsidian vault is a read/write human-readable memory bank — never the runtime database.

**Tech Stack:** Python 3.11+, dataclasses, PyYAML, pytest, Anthropic SDK (optional for LLM judge)

---

## File Map

### Delete
| File | Reason |
|---|---|
| `neural_search/labeling/` (entire dir) | Interactive session/storage/agreement infra |
| `neural_search/evaluation/label_relevance.py` | Interactive CLI labeler |
| `neural_search/evaluation/relevance.py` | RelevanceJudgment / hand-label model |
| `scripts/run_annotation_session.py` | Annotation session entrypoint |
| `scripts/eval/annotate_candidates.py` | Pool annotation script |
| `tests/test_relevance.py` | Tests for hand-label code |

### Create
```
neural_search/eval/__init__.py
neural_search/eval/evidence.py
neural_search/eval/query_decomposition.py
neural_search/eval/labeling_functions.py
neural_search/eval/llm_judge.py
neural_search/eval/label_ensemble.py

neural_search/obsidian/__init__.py
neural_search/obsidian/templates.py
neural_search/obsidian/io.py

scripts/eval/build_evidence.py
scripts/eval/run_labeling_functions.py
scripts/eval/judge_candidates.py
scripts/eval/build_qrels_from_votes.py
scripts/eval/hard_negative_analysis.py

scripts/obsidian/init_vault.py
scripts/obsidian/export_dataset_cards.py
scripts/obsidian/export_query_cards.py
scripts/obsidian/export_audit_queue.py
scripts/obsidian/import_audits.py
scripts/obsidian/compile_vault.py
scripts/obsidian/export_claim_registry.py

configs/judges/rubric_judge_v1.yaml
obsidian_vault/.gitkeep (plus 10 subfolder .gitkeeps)

tests/test_obsidian_io.py
tests/test_obsidian_exports.py
tests/test_eval_evidence.py
tests/test_labeling_functions.py
tests/test_label_ensemble.py
tests/test_qrels_tiers.py

docs/OBSIDIAN_EVAL_MEMORY.md
docs/WEAK_SUPERVISION_LABELING.md
docs/HUMAN_AUDIT_PROTOCOL.md
reports/eval/whitepaper_claims_status.md
```

### Modify
```
neural_search/evaluation/__init__.py       (remove labeling imports)
tests/test_search_quality.py               (fix broken imports after deletion)
scripts/eval/compute_ir_metrics.py         (add --qrels-tier, tier warning)
scripts/eval/compute_calibration.py        (add --qrels-tier, tier warning)
```

---

## Task Index

| # | File | Description |
|---|---|---|
| [01](01-delete-handlabeling.md) | — | Delete hand-labeling code, fix broken imports |
| [02](02-data-models.md) | `neural_search/eval/evidence.py` | Core data models: QuerySpec, DatasetEvidence, PairEvidence, LFVote |
| [03](03-evidence-extraction.md) | `neural_search/eval/query_decomposition.py` | Parse benchmark queries → QuerySpec |
| [04](04-evidence-extraction.md) | `scripts/eval/build_evidence.py` | Build pair_evidence.jsonl from pool + corpus |
| [05](05-labeling-functions.md) | `neural_search/eval/labeling_functions.py` | All 13 deterministic LFs |
| [06](06-labeling-functions.md) | `scripts/eval/run_labeling_functions.py` | Run LFs over pair evidence |
| [07](07-label-ensemble.md) | `neural_search/eval/label_ensemble.py` | Vote aggregation, tier assignment |
| [08](07-label-ensemble.md) | `scripts/eval/build_qrels_from_votes.py` | Build gold/silver/bronze qrels |
| [09](08-llm-judge.md) | `neural_search/eval/llm_judge.py` | Optional LLM rubric judge |
| [10](09-obsidian.md) | `neural_search/obsidian/templates.py` + `io.py` | Vault I/O with write-safety |
| [11](09-obsidian.md) | `scripts/obsidian/init_vault.py` etc. | Vault scaffold + card exports |
| [12](09-obsidian.md) | `scripts/obsidian/export_audit_queue.py` + `import_audits.py` | Audit loop |
| [13](10-metrics.md) | `scripts/eval/compute_ir_metrics.py` etc. | Metrics tier support |
| [14](10-metrics.md) | `scripts/eval/hard_negative_analysis.py` | Dedicated HN analysis script |
| [15](11-tests.md) | `tests/test_*.py` | All 6 test files |
| [16](12-docs-cleanup.md) | `docs/`, `reports/` | Documentation |
| [17](12-docs-cleanup.md) | — | Full package review + delete unneeded files |
