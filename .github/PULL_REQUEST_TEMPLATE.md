## Problem

What user, engineering, or scientific problem does this PR solve?

## Changes

- 

## Execution-profile impact

Which profiles are affected: `demo`, `researcher`, `corpus-builder`, `evaluator`, `full-stack`? If none, say `None`.

If dependencies, artifact assumptions, setup commands, or runtime behavior changed, update the profile contract and documentation in the same PR.

## Validation

- [ ] `neural-search profile check demo`
- [ ] `neural-search profile check evaluator`
- [ ] `ruff check neural_search apps/api scripts tests`
- [ ] `pytest -q`
- [ ] `cd apps/web && npm run build`
- [ ] Relevant benchmark/evaluation run if retrieval, ranking, graph features, normalization, qrels, or scientific outputs changed

## Scientific/evidence impact

Describe any effect on retrieval metrics, qrels, corpus state, evidence tiers, provenance, generated artifacts, or scientific claims. Write `None` if the change is engineering-only.

Do not describe heuristic/silver labels as expert/gold judgments. If metrics changed, identify whether the cause is code, configuration, corpus state, qrels, model choice, or artifact regeneration.

## Reproducibility and artifacts

List generated files, registered artifact changes, large local assets, migration steps, or regeneration commands required by this change.

- Does a new durable asset need an `ArtifactSpec`?
- Is it required, recommended, or produced by a profile?
- Does a clean clone still complete the `demo` adoption journey?
- If scientific results depend on local artifact state, attach/reference a reproducibility manifest.

## Risks / follow-up

- 
