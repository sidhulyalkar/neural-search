# Top Opportunities

Generated: 2026-06-11T01:53:35.996640+00:00

Scoring formula:

`total_score = 0.20 * novelty_score + 0.25 * feasibility_score + 0.20 * impact_score + 0.15 * uncertainty_reduction_score + 0.15 * personal_fit_score - 0.10 * risk_score`

| Rank | Opportunity | Total | Review | Priority | Novelty | Feasibility | Impact | Uncertainty | Fit | Risk |
| ---: | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Human qrels benchmark for dataset-method compatibility | 7.700 | unreviewed |  | 7.0 | 7.0 | 10.0 | 10.0 | 9.0 | 3.0 |
| 2 | Hard-negative generator | 7.350 | unreviewed |  | 6.8 | 8.2 | 8.6 | 8.6 | 8.2 | 3.0 |
| 3 | Dataset future validity score | 7.180 | unreviewed |  | 8.6 | 6.8 | 8.6 | 8.2 | 8.2 | 4.2 |
| 4 | Analysis affordance extraction benchmark | 6.835 | unreviewed |  | 7.6 | 6.8 | 8.2 | 7.8 | 8.1 | 4.1 |
| 5 | Provenance confidence score | 6.665 | unreviewed |  | 6.8 | 7.8 | 7.4 | 7.2 | 7.7 | 3.6 |
| 6 | Metadata richness vs reuse value study | 6.375 | unreviewed |  | 6.5 | 7.5 | 7.1 | 7.0 | 7.4 | 3.8 |
| 7 | Dataset-method transfer map | 6.270 | unreviewed |  | 8.2 | 5.8 | 7.7 | 6.8 | 7.6 | 5.2 |
| 8 | Uncertainty-aware dataset recommendation | 5.930 | unreviewed |  | 7.4 | 5.6 | 7.6 | 6.6 | 7.2 | 5.4 |

## Rationale

### Human qrels benchmark for dataset-method compatibility

Create a focused expert-labeled benchmark for retrieval and ranking metrics.

- ID: `opp_human_qrels_benchmark`
- Next step: Draft 20 compatibility queries and label a small candidate pool.
- Rationale: Highest leverage because it turns retrieval claims into measurable claims.
- Human review: `unreviewed`
- Human status: `candidate`
- Source note: `Field-State/40_Opportunities/candidate/Human qrels benchmark for dataset-method compatibility.md`

### Hard-negative generator

Generate and track near-miss negatives for scientific retrieval queries.

- ID: `opp_hard_negative_generator`
- Next step: Define negative categories and produce a small JSONL fixture.
- Rationale: Practical and directly improves benchmark rigor.
- Human review: `unreviewed`
- Human status: `candidate`
- Source note: `Field-State/40_Opportunities/candidate/Hard-negative generator.md`

### Dataset future validity score

Estimate whether a dataset is likely to remain reusable and scientifically valuable.

- ID: `opp_dataset_future_validity_score`
- Next step: Define reuse proxies and build a simple retrospective label set.
- Rationale: High upside, but depends on careful proxy design.
- Human review: `unreviewed`
- Human status: `candidate`
- Source note: `Field-State/40_Opportunities/candidate/Dataset future validity score.md`

### Analysis affordance extraction benchmark

Evaluate whether extracted affordances match what datasets actually support.

- ID: `opp_analysis_affordance_benchmark`
- Next step: Pick 5 affordances and label 30 dataset-affordance pairs.
- Rationale: Directly validates the recommendation layer.
- Human review: `unreviewed`
- Human status: `candidate`
- Source note: `Field-State/40_Opportunities/candidate/Analysis affordance extraction benchmark.md`

### Provenance confidence score

Score recommendations by the strength and source of their supporting evidence.

- ID: `opp_provenance_confidence_score`
- Next step: Map evidence sources to confidence priors and audit examples.
- Rationale: Good bridge between retrieval scores and scientific trust.
- Human review: `unreviewed`
- Human status: `candidate`
- Source note: `Field-State/40_Opportunities/candidate/Provenance confidence score.md`

### Metadata richness vs reuse value study

Test whether richer metadata predicts reuse-relevant labels or retrieval success.

- ID: `opp_metadata_richness_vs_reuse`
- Next step: Define metadata richness features and correlate against existing readiness signals.
- Rationale: Useful, scoped, and likely to explain current retrieval failures.
- Human review: `unreviewed`
- Human status: `candidate`
- Source note: `Field-State/40_Opportunities/candidate/Metadata richness vs reuse value study.md`

### Dataset-method transfer map

Map which dataset properties transfer across analysis methods and scientific questions.

- ID: `opp_dataset_method_transfer_map`
- Next step: Prototype a small map from methods to required data affordances.
- Rationale: Strategically important but less immediate than benchmark foundations.
- Human review: `unreviewed`
- Human status: `candidate`
- Source note: `Field-State/40_Opportunities/candidate/Dataset-method transfer map.md`

### Uncertainty-aware dataset recommendation

Expose uncertainty and missing evidence alongside ranked dataset recommendations.

- ID: `opp_uncertainty_aware_recommendation`
- Next step: Add a prototype uncertainty explanation to result cards after qrels exist.
- Rationale: Valuable after calibration and provenance signals mature.
- Human review: `unreviewed`
- Human status: `candidate`
- Source note: `Field-State/40_Opportunities/candidate/Uncertainty-aware dataset recommendation.md`

Weights:
- `novelty_score`: +0.20
- `feasibility_score`: +0.25
- `impact_score`: +0.20
- `uncertainty_reduction_score`: +0.15
- `personal_fit_score`: +0.15
- `risk_score`: -0.10