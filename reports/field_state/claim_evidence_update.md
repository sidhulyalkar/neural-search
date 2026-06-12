# Claim Evidence Update Suggestions

## claim_dense_semantic_retrieval

- Current evidence level: `plausible`
- Suggested evidence level: `plausible`
- Safe to auto-apply: `False`
- Reason: Needs qrels-backed ablation comparing dense retrieval against baselines.
- Supporting artifacts:
- `artifacts/field_state/adjudicated_qrels.jsonl`
- `artifacts/field_state/qrels_agreement.json`
- Missing artifacts:
- `reports/eval/dense_ablation.json`

## claim_hard_negatives

- Current evidence level: `supported`
- Suggested evidence level: `supported`
- Safe to auto-apply: `False`
- Reason: Needs hard-negative violation metrics linked to adjudicated qrels.
- Supporting artifacts:
- `artifacts/field_state/adjudicated_qrels.jsonl`
- `artifacts/field_state/qrels_agreement.json`
- Missing artifacts:
- `reports/field_state/hard_negative_review.md`

## claim_human_qrels

- Current evidence level: `plausible`
- Suggested evidence level: `plausible`
- Safe to auto-apply: `False`
- Reason: Human qrels claim remains a design requirement until adjudicated qrels exist.
- Supporting artifacts:
- `artifacts/field_state/adjudicated_qrels.jsonl`
- `artifacts/field_state/qrels_agreement.json`
- Missing artifacts:
- `none`

## claim_metadata_richness

- Current evidence level: `plausible`
- Suggested evidence level: `plausible`
- Safe to auto-apply: `False`
- Reason: Needs metadata quality scores correlated with reuse labels.
- Supporting artifacts:
- `artifacts/field_state/adjudicated_qrels.jsonl`
- `artifacts/field_state/qrels_agreement.json`
- Missing artifacts:
- `reports/field_state/metadata_quality_scoring.md`

## claim_affordance_extraction

- Current evidence level: `plausible`
- Suggested evidence level: `plausible`
- Safe to auto-apply: `False`
- Reason: Needs content or human validation of analysis affordance labels.
- Supporting artifacts:
- `artifacts/field_state/adjudicated_qrels.jsonl`
- `artifacts/field_state/qrels_agreement.json`
- Missing artifacts:
- `reports/field_state/affordance_validation.md`

## claim_graph_proximity

- Current evidence level: `hypothesis`
- Suggested evidence level: `hypothesis`
- Safe to auto-apply: `False`
- Reason: No qrels-backed evidence update available yet.
- Supporting artifacts:
- `artifacts/field_state/adjudicated_qrels.jsonl`
- `artifacts/field_state/qrels_agreement.json`
- Missing artifacts:
- `none`
