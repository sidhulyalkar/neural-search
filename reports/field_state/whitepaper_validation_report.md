# Whitepaper Validation Report

## Current Methodology Status

Neural Search has engineering validation for corpus construction, indexing, and artifact generation. Scientific retrieval validation remains gated by human-reviewed qrels, adjudication, metric reports, source-skew diagnostics, and calibration.

## Engineering Validation

- Corpus and evaluation artifacts are generated locally.
- Field-state reports and Obsidian review memory are reproducible from JSONL artifacts.

## Scientific Retrieval Validation

- Qrels candidates exported: 10
- Qrels candidates reviewed: 0
- Qrels candidates adjudicated: 0
- Candidates needing adjudication: 0

## Human Relevance Validation

Gold retrieval claims should remain caveated until at least two human reviews are available where possible and disagreements have been adjudicated.

## Future Usefulness Validation

Future usefulness claims require longitudinal reuse proxies, content-validated affordances, and calibrated confidence estimates. Current field-state artifacts should treat those claims as hypotheses or preliminary evidence.

## Available Supporting Artifacts

- `artifacts/field_state/adjudicated_qrels.jsonl`
- `artifacts/field_state/qrels_agreement.json`
- `reports/eval/corpus_manifest.json`
- `reports/eval/retrieval_baselines_status.json`

## Publication-Grade Work Remaining

- Complete human qrels review and adjudication.
- Compute nDCG/MRR/Recall metrics from adjudicated qrels.
- Track hard-negative violation rate.
- Add source-skew and calibration reports.
- Validate analysis affordances against human or content-level evidence.