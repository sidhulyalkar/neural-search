---
run_id: 2026-07-04_kg_connectivity_audit
agent: kg-connectivity-auditor
outcome: ok
created: '2026-07-04'
tags:
- agent-run
- kg-connectivity-auditor
- kg
type: agent_run
---

## First formal run of this playbook

Previous connectivity sweeps (2026-07-01, 2026-07-02, see
`kg_connectivity_audit` project memory) found and resolved 8/8 originally-
orphaned builders plus one dead module, all by hand. This is the first time
the same procedure ran as a written playbook
(`artifacts/agents/playbooks/kg_connectivity_auditor.md`) rather than an
ad-hoc request.

## Module status table

| Module | Status | Evidence |
|---|---|---|
| `citation_builder.py` | **Reconnected this session** | Was an orphan (node-ID-scheme mismatch: `paper:openalex:{id}` vs. production's `node:paper:openalex:{id}`, plus re-scanned raw files instead of the actual graph). Fixed via `build_citation_edges_for_graph()`, now in `orphaned_layers`. 344 real edges. |
| `reprocessing_candidate_builder.py` | New, merged | Attached as a node property (not an edge, matching `attach_retraction_status`'s reasoning), via `attach_reprocessing_candidate_status()`. |
| `ner_builder.py` | Side-channel (confirmed, unchanged) | `artifacts/ner/ner_kg.jsonl` read directly by `search_features._load_ner_index()`. |
| `neurosynth_builder.py` | Side-channel with fallback (confirmed, unchanged) | Per 2026-07-02 audit; falls back to direct build if the artifact is missing. |
| `atlas_builder.py` | Reachable, not orphaned | Called from `apps/api/atlas_router.py` — a dedicated API route, a different reachability path than the KG merge, but a real one. |
| `openalex_citations.py` | Operational script, not orphaned | Has `if __name__ == "__main__"`; likely the upstream producer of `artifacts/citations/citation_edges.jsonl` (the file `citation_builder.py` reads). Same category as `check_paper_retraction_status.py`. |
| `paper_linking.py` | **Inconclusive — needs follow-up** | Reachable via `neural_search/core/query.py` → `retrieval.py` / `evaluation/baseline_ladder.py`, but unclear whether `neural_search/core/*` is on the production `apps/api/main.py` search path or is a separate evaluation-only retrieval implementation. Not resolved this run. |
| `semantic_edges.py` | **New finding: likely dead code** | Zero real callers found for any of its public functions (`add_semantic_edges_to_graph`, `build_semantic_dataset_edges`, `load_and_add_semantic_edges`, `get_semantic_neighbors`, `build_concept_similarity_edges`) beyond the module itself and `graph/__init__.py`'s re-export — the exact same shape as `similarity.py`, which was confirmed dead and deleted on 2026-07-02. Proposed action: re-confirm with one more repo-wide grep, then delete + remove the `__init__.py` re-export, following the `similarity.py` precedent exactly. |

## What's genuinely new vs. re-confirmed

- New, real finding: `semantic_edges.py` is likely dead code. Not yet
  deleted — this playbook's gate is `none` (read-only); deletion needs a
  human/code-review pass, same as any code change would.
- Resolved this session (not by this audit, but confirmed by it):
  `citation_builder.py` orphan status.
- Open item carried forward: `paper_linking.py`'s production-path status.

## Ledger

See `artifacts/agents/ledger.jsonl`, `agent: "kg-connectivity-auditor"`,
`started_at: "2026-07-04T12:20:00+00:00"`.
