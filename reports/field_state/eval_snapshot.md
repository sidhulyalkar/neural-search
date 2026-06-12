# Field-State Evaluation Snapshot

## Qrels Status

- Candidates exported: 10
- Candidates reviewed: 0
- Candidates adjudicated: 0
- Candidates needing adjudication: 0

## Label Distribution

- Relevance:
- Usefulness:
- Hard-negative violations: 0

## Agreement

- Exact agreement rate: None
- Disagreement count: 0

## Available Evaluation Artifacts

- `artifacts/field_state/adjudicated_qrels.jsonl`
- `artifacts/field_state/qrels_agreement.json`
- `reports/eval/corpus_manifest.json`
- `reports/eval/retrieval_baselines_status.json`

## Claim Evidence Update Suggestions

- Strengthen qrels-related claims only after adjudicated labels and metric reports exist.
- Keep dense retrieval, hard-negative, and affordance claims caveated until qrels-backed metrics are generated.

## Recommended Next Actions

- Review unreviewed qrels candidates.
- Adjudicate disagreements.
- Run retrieval metrics against adjudicated qrels.
- Add source-skew and calibration reports before publication-grade claims.