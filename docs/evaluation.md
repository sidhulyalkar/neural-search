# Evaluation

Neural Search evaluation asks whether search results recover expected scientific labels and rank genuinely relevant datasets higher — not whether generated prose sounds plausible.

## Canonical Benchmark

The canonical benchmark (`data/eval/benchmark_queries_canonical.yaml`) has 317 queries with LLM-judged (silver, not gold) relevance labels across dataset search, adversarial constraints, paper linking, affordance matching, graph reasoning, and experimental design.

```bash
python scripts/eval/run_ablation_ladder.py --skip-rungs bm25 bm25_structured dense_bge
```

This runs the graph-aware rungs and reports NDCG@10 per rung. Current production result:

| Rung | NDCG@10 |
|------|---------|
| hybrid_graph | 0.8594 |
| typed_kg | 0.8483 |
| typed_kg_qualified | 0.8483 |
| full | 0.8594 |

**Every KG-modifying change must re-run this ladder and confirm the number is unchanged (or the regression is understood and fixed) before being considered done.** This project has caught three real ranking regressions this way — a new densely-populated edge type dominating a generic graph-connectivity feature (twice, different edge types) and a score weight configured speculatively before any real data existed to feed it. A weekly scheduled agent (`benchmark-gatekeeper`, see `artifacts/agents/playbooks/`) now runs this check automatically after any graph change.

## The Gold Qrels Gap

The 317-query benchmark's labels are LLM-judged, not human-adjudicated. Gold qrels are currently 0 rows. This means: the NDCG@10 numbers above are useful for **regression detection** (did a change make things worse?) but should not yet be cited as a validated retrieval-quality claim. Closing this gap is the highest-priority open item in this project.

## Other Evaluation Surfaces

- **Coverage reports** (`reports/eval/current_artifact_manifest.json`, regenerated via `scripts/build_artifact_manifest.py`): corpus size, KG node/edge counts by type, paper-link coverage by source, qrels tier counts. This is the single source of truth every other report and the whitepaper's generated statistics are built from — never hand-edit these numbers.
- **File validation reports** (`reports/top_suggestions_validation_report.md`): which reanalysis suggestions were confirmed by live DANDI/OpenNeuro file inspection vs. which sources have no live validator yet.
- **KG connectivity audits** (`artifacts/agents/playbooks/kg_connectivity_auditor.md`, run weekly): is every KG-producing module reachable, side-channel-connected, dead, or a stated orphan?

## Interpreting Failures

A ranking or coverage gap can mean several different things:

- The ontology lacks a synonym.
- The corpus genuinely lacks a relevant dataset for that query.
- Metadata extraction missed a label.
- A ranking weight underemphasizes a useful signal.
- The query expects an analysis concept that's only weakly represented in metadata.

Failures should feed ontology updates, corpus coverage priorities, extraction improvements, and scoring changes — and, where relevant, back into the agent-orchestration ledger so the finding isn't re-derived from scratch next time.

## Future Evaluation

- Human relevance judgments per (query, dataset) pair — the gold-qrels gap above.
- Dual-judge agreement and adjudication for the silver labels already collected.
- Source-specific coverage and skew diagnostics on adjudicated labels.
- Notebook-generation success rates and dataset-card QA agreement as secondary quality signals.
