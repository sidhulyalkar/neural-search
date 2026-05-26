# Task 36: Source Quality and Promotion Gates

Status: source-quality assessment and optional promotion-gate integration implemented.

This task keeps source trust separate from scientific relevance. Source quality should inform readiness, QA, and promotion blockers, but it should not silently boost or suppress search relevance until human-label gates justify that behavior.

Implemented behavior:

- Deterministic source quality profiles for DANDI, OpenNeuro, OpenAlex, ModelDB, cellxgene, MICrONS, Allen Brain, NeMO, demo, and manual records.
- Per-record source quality assessments with trust level, quality score, matched standards, missing expected standards, warnings, and fixture-backed status.
- Readiness reports now include a `source_quality` section and warnings for unknown, low-trust, or low-mean-quality source coverage.
- Promotion reports can optionally consume a scientific readiness JSON report and enforce source-quality gates for mean quality, unknown sources, low-trust sources, and source warnings.
- Default promotion config keeps source-quality gates disabled until reviewed labels and release policy are mature.
- Release summaries can include source-quality status and non-failing release warnings from the scientific readiness report.

Next development:

- Add registry-backed profile loading once the parallel `database_registry.yaml` work stabilizes.
- Keep source quality out of ranking defaults until intent-specific evaluation labels show a measurable benefit.
