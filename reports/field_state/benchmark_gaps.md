# Benchmark Gaps

Generated: 2026-06-11T01:53:34.661915+00:00

## Human qrels benchmark

A small expert-labeled relevance set for dataset-method compatibility queries.

- ID: `gap_human_qrels_benchmark`
- Severity: 0.95
- Status: `open`
- Why it matters: Without human qrels, nDCG/MRR can look precise while measuring proxy labels.
- Human review: `unreviewed`
- Human status: `open`
- Source note: `Field-State/30_Benchmark_Gaps/active/Human qrels benchmark.md`
- Expected artifacts:
- artifacts/qrels.jsonl
- Available artifacts:
- none
- Blocking questions:
- Which query intents should be labeled first?
- What label scale captures compatibility rather than topical similarity?

## Hard-negative violation tracking

A report that detects when near-miss datasets outrank truly compatible datasets.

- ID: `gap_hard_negative_violation_tracking`
- Severity: 0.88
- Status: `open`
- Why it matters: Scientific search needs to distinguish superficially similar datasets from reusable ones.
- Human review: `unreviewed`
- Human status: `open`
- Source note: `Field-State/30_Benchmark_Gaps/active/Hard-negative violation tracking.md`
- Expected artifacts:
- reports/field_state/hard_negative_violations.md
- Available artifacts:
- none
- Blocking questions:
- What counts as a violation for method compatibility?
- Which negative categories are most damaging?

## Analysis affordance extraction

A gold set for whether datasets support specific analyses.

- ID: `gap_analysis_affordance_extraction`
- Severity: 0.86
- Status: `open`
- Why it matters: Affordance extraction is central to moving beyond keyword search.
- Human review: `unreviewed`
- Human status: `open`
- Source note: `Field-State/30_Benchmark_Gaps/active/Analysis affordance extraction.md`
- Expected artifacts:
- reports/field_state/analysis_affordance_benchmark.md
- Available artifacts:
- none
- Blocking questions:
- Which affordances matter first for neuroscience dataset reuse?
- What evidence is sufficient to mark an affordance as supported?

## Metadata quality scoring

A transparent score for completeness, specificity, provenance, and reuse-relevant fields.

- ID: `gap_metadata_quality_scoring`
- Severity: 0.80
- Status: `open`
- Why it matters: Metadata quality may explain both retrieval errors and dataset reuse potential.
- Human review: `unreviewed`
- Human status: `open`
- Source note: `Field-State/30_Benchmark_Gaps/active/Metadata quality scoring.md`
- Expected artifacts:
- reports/field_state/metadata_quality_scoring.md
- Available artifacts:
- none
- Blocking questions:
- Which metadata fields are predictive rather than merely available?
- How should missingness be handled across archives?

## Calibration/ECE

Expected calibration error for recommendation confidence.

- ID: `gap_calibration_ece`
- Severity: 0.74
- Status: `open`
- Why it matters: Users need to know whether high-confidence recommendations are actually reliable.
- Human review: `unreviewed`
- Human status: `open`
- Source note: `Field-State/30_Benchmark_Gaps/active/CalibrationECE.md`
- Expected artifacts:
- reports/eval/calibration_ece.json
- Available artifacts:
- none
- Blocking questions:
- What score should be calibrated: relevance, usefulness, or compatibility?
- How many labels are needed for a meaningful calibration curve?

## Future reuse prediction

A benchmark for whether dataset signals predict later reuse or scientific value.

- ID: `gap_future_reuse_prediction`
- Severity: 0.72
- Status: `open`
- Why it matters: The project should recommend datasets with future validity, not just current discoverability.
- Human review: `unreviewed`
- Human status: `open`
- Source note: `Field-State/30_Benchmark_Gaps/active/Future reuse prediction.md`
- Expected artifacts:
- reports/field_state/future_reuse_prediction.md
- Available artifacts:
- none
- Blocking questions:
- Which reuse proxy is acceptable for v0.1?
- How should age, source, and popularity bias be controlled?
