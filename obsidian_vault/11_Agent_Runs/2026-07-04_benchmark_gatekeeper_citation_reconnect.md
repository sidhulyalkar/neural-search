---
run_id: 2026-07-04_benchmark_gatekeeper_citation_reconnect
agent: benchmark-gatekeeper
outcome: ok
gate_result: pass
created: '2026-07-04'
tags:
- agent-run
- benchmark-gatekeeper
- kg
type: agent_run
---

## What triggered this run

A production graph rebuild after three changes in the same session:
1. `citation_builder.py` reconnected (was an orphan with a node-ID-scheme
   bug — see the companion `kg-connectivity-auditor` note from the same
   day).
2. New `reprocessing_candidate_builder.py` (NWB version staleness signal).
3. `scripts/validate_top_reanalysis_suggestions.py` scaled from top-50 to
   top-300, feeding more `file_validated` evidence-tier upgrades.

## What ran

```
python scripts/build_real_corpus_graph.py
python scripts/eval/run_ablation_ladder.py --skip-rungs bm25 bm25_structured dense_bge hybrid_rrf
```

## Result

| Rung | Before | After |
|---|---|---|
| hybrid_graph | 0.8594 | 0.8594 |
| typed_kg | 0.8483 | 0.8483 |
| typed_kg_qualified | 0.8483 | 0.8483 |
| full | 0.8594 | 0.8594 |

No regression. Graph grew from 12,748 nodes / 149,654 edges to 12,748 nodes
/ 149,998 edges (+344 `paper_cites_paper` edges from the citation
reconnect; node count unchanged since citations only add edges between
already-existing paper nodes).

## Why this is worth recording, not just "it passed"

This project has hit a real NDCG regression from a new densely-populated
edge type three separate times (see `ablation_results` memory) — each time
it looked exactly this innocuous beforehand. This run passing clean on the
first try is itself informative: it confirms the pre-existing
`graph_degree` exclusion list already covers `paper_cites_paper`-shaped
edges, so no new exclusion was needed this time.

## Ledger

See `artifacts/agents/ledger.jsonl`, `agent: "benchmark-gatekeeper"`,
`started_at: "2026-07-04T12:10:00+00:00"`.
