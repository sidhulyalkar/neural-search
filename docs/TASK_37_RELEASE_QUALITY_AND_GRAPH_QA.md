# Task 37: Release Quality and Graph QA

Status: first release-visibility slice implemented.

This task makes release summaries more useful for scaling without turning early QA signals into brittle CI blockers. Graph structure quality is now visible beside artifact, benchmark, and source-quality status.

Implemented behavior:

- Release summaries include a `graph_quality` section for graph artifacts.
- Graph QA issues are summarized by availability, pass status, node/edge count, error count, warning count, and issue-code counts.
- Release warnings include graph QA errors and warnings, but `release_ready` remains governed by existing hard blockers until explicit promotion policy changes.
- Release artifact path rendering now supports repo-local and temporary/external artifact paths.
- Release artifact summaries include SHA-256 digests for reproducibility and rollback inspection.

Next development:

- Add configurable graph QA thresholds to release policy.
- Promote dangling references and invalid confidence values to hard blockers once graph artifacts stabilize.
- Add required node/edge-type checks for production graph variants.
- Combine graph QA, source quality, calibration, and benchmark deltas into one promotion-readiness dashboard.
