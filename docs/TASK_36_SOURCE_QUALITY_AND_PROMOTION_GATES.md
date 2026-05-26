# Task 36: Source Quality and Promotion Gates

Status: first source-quality slice implemented.

This task keeps source trust separate from scientific relevance. Source quality should inform readiness, QA, and promotion blockers, but it should not silently boost or suppress search relevance until human-label gates justify that behavior.

Implemented behavior:

- Deterministic source quality profiles for DANDI, OpenNeuro, OpenAlex, ModelDB, cellxgene, MICrONS, Allen Brain, NeMO, demo, and manual records.
- Per-record source quality assessments with trust level, quality score, matched standards, missing expected standards, warnings, and fixture-backed status.
- Readiness reports now include a `source_quality` section and warnings for unknown, low-trust, or low-mean-quality source coverage.

Next development:

- Add source quality thresholds to release/promotion reports.
- Add registry-backed profile loading once the parallel `database_registry.yaml` work stabilizes.
- Keep source quality out of ranking defaults until intent-specific evaluation labels show a measurable benefit.
