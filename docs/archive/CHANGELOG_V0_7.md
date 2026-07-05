# Neural Search v0.7 Changelog

## Task 9: Real Corpus and File Inspection

- Added manifest-driven `real_v07` fixture ingestion.
- Added DANDI, OpenNeuro, and OpenAlex manifest entries with priority, status, tags, fetch metadata, review notes, and scientific rationale.
- Added tiny deterministic NWB and BIDS file-inspection fixtures.
- Added evidence-backed file-inspection claims under `data/corpus/claims/`.
- Added claim-aware normalized real-corpus records, graph artifact, hashing field embeddings, and reports.
- Added `real_v07` benchmark suite registration.

## Task 10: Release and Product Hardening

- Added versioned API contract models for search responses, dataset cards, graph context, linked papers, workflow outputs, and benchmark audits.
- Added explicit optional search response fields for graph context, linked papers, field/graph scores through score breakdowns, filtered constraints, and missing metadata.
- Added search trace capture for parsed query, filters, score heads, graph/field settings, filtered constraints, timings, and result explanations.
- Added `make release-check` and release summary outputs under `data/reports/release/`.
- Added release notes template with artifact and metric checklist.
