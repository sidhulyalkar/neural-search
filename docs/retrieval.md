# Retrieval and Ranking

Neural Search uses a small, transparent retrieval pipeline:

1. Parse the query into scientific intent.
2. Match each dataset against ontology labels, metadata, card evidence, readiness, and provenance.
3. Return explanations that expose both positive evidence and reuse risks.

The pipeline is intentionally configurable rather than learned. Weights and parser aliases live in
`data/config/retrieval.yaml`.

## Query Parsing

`parse_query()` detects:

- Task intent from the behavioral task ontology, including labels and synonyms.
- Behavior intent from ontology behavior labels.
- Modality intent from explicit modality terms plus broad phrases such as "neural recordings".
- Species intent from configured aliases such as `mouse`, `rat`, and `human`.
- Brain region intent from ontology region names and aliases.
- Analysis intent from configured phrases such as `decode choice`, `event alignment`, and `latent state modeling`.

The parsed response keeps both compact IDs (`tasks`, `behaviors`, `modalities`, `species`,
`brain_regions`, `analysis`) and evidence-bearing intent records (`task_intent`,
`behavior_intent`, `analysis_intent`, and related fields).

## Ranking Signals

Scores are combined on a 0-1 scale and returned as 0-100. The default weights are:

| Signal | Purpose |
|--------|---------|
| Ontology | Strong boost for exact task matches. |
| Behavior | Strong boost for requested behavioral events or labels. |
| Modality | Rewards matching requested recording/data modalities. |
| Metadata | Rewards requested species, brain regions, and analysis support. |
| Semantic | Lightweight keyword evidence from expanded ontology terms. |
| Readiness | Boosts datasets that are ready for analysis. |
| Paper confidence | Small provenance boost from linked papers. |

Penalties are applied for modality mismatches and missing required metadata. Linked papers are
deliberately capped so they improve confidence without dominating scientific relevance.

## Explanations

Each result includes:

- `why_matched`: human-readable reasons such as task, behavior, modality, readiness, and provenance.
- `matched_terms`: normalized labels and terms that matched the query.
- `inferred_concepts`: concepts inferred during parsing.
- `evidence_snippets`: short source snippets from dataset text or card evidence.
- `missing_metadata_warnings`: required metadata gaps.
- `reusable_reason`: a concise statement of why the dataset is scientifically reusable.
- `dataset_card_preview.score_breakdown`: normalized component scores and penalties.

## Benchmark Feedback

Run the current benchmark:

```bash
python -m neural_search.evaluation.run_benchmark
```

Capture a comparison against a prior JSON report:

```bash
python -m neural_search.evaluation.run_benchmark \
  --compare-to data/eval/results/baseline_before_retrieval_upgrade/latest_eval_report.json
```

The runner writes:

- `data/eval/results/latest_eval_report.md`
- `data/eval/results/latest_eval_report.json`
- `data/eval/results/retrieval_comparison_report.md` when `--compare-to` is supplied
- `data/eval/results/retrieval_comparison_report.json` when `--compare-to` is supplied

Some benchmark queries describe concepts that are not represented in the demo corpus. Those queries
are useful parser-coverage checks, but they cannot be fully recovered by ranking alone until matching
datasets are indexed.
