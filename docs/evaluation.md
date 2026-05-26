# Evaluation

Neural Search evaluation asks whether search results recover expected scientific labels, not whether generated prose sounds plausible.

Benchmark queries live in `data/eval/benchmark_queries.yaml`. Each query defines expected tasks, behaviors, modalities, species, or analysis goals.

## Run Evaluation

```bash
make benchmark
```

or from the UI:

```text
/evaluation -> Run Benchmark
```

The benchmark runner writes reports under `data/eval/results/` when run from the CLI. The API also produces an in-process report for the frontend dashboard.

## Metrics

| Metric | Meaning |
| --- | --- |
| Precision@5 | Fraction of top five results with match evidence |
| Label Recall@10 | Fraction of expected task/modality/behavior labels recovered in top results |
| Task Match Rate | Expected task label coverage |
| Modality Match Rate | Expected modality label coverage |
| Behavior Match Rate | Expected behavior label coverage |
| Queries With Results | Number of benchmark queries that returned at least one result |

The frontend marks a query as passing when:

- Precision@5 is at least 40%.
- Label recall is at least 50%.

These thresholds are demo thresholds, not scientific acceptance criteria.

## What To Inspect

For each benchmark query, inspect:

- The original query.
- Expected tasks and modalities.
- Found tasks and modalities.
- Top returned datasets.
- Warnings about missing expected labels.
- Recommendations generated from repeated misses.

## Current Benchmark Themes

The seed benchmark covers:

- Go/NoGo and response inhibition.
- Reversal learning and reward omission.
- Delay discounting.
- Visual decision-making with Neuropixels.
- Reaching, ECoG/iEEG, and BCI-oriented data.
- Choice decoding.
- Naturalistic vision and arousal.
- Motor imagery EEG.
- Seizure monitoring.
- Social interaction with behavior video.

## Interpreting Failures

A failed benchmark can mean several different things:

- The ontology lacks a synonym.
- Demo seed data does not contain a relevant dataset.
- Metadata extraction missed a label.
- Ranking weights underemphasize a useful signal.
- The query expects an analysis concept that is only weakly represented in metadata.

Failures should feed ontology updates, seed-data coverage, extraction improvements, and scoring changes.

## Future Evaluation

A stronger evaluation set should add:

- Human relevance judgments per dataset.
- Query difficulty levels.
- Source-specific coverage checks.
- Notebook generation success rates.
- Dataset-card QA agreement.
- Latent neural-state retrieval metrics once representation search is added.
