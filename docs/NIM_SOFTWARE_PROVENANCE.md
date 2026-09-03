# NVIDIA NIM + scientific software provenance

This document defines the first production contract for extending Neural Search from dataset/literature discovery into evidence-gated scientific software auditing and open-data reanalysis.

## Scope

The system is intentionally **not** an autonomous bug-filing agent. It separates four claims that must never be collapsed:

1. **Audit hypothesis** — code reading suggests a behavior deserves investigation.
2. **Verified software finding** — executable or numerical evidence demonstrates the behavior.
3. **Exposure** — a paper is mapped to an affected component or release.
4. **Scientific impact** — a controlled pipeline or open-data reanalysis demonstrates a downstream effect.

A paper being exposed to a code path does not imply that its conclusions are wrong.

## Architecture

```text
literature + datasets + repositories
              |
              v
      scientific dependency graph
              |
              v
        audit prioritization
              |
              v
  model-assisted code hypothesis
              |
              v
 deterministic verification harness
              |
              v
 paper/software exposure join
              |
              v
  open-data reanalysis candidates
              |
              v
 human-gated contribution packet
              |
              v
 upstream maintainer adjudication
```

NVIDIA NIM is an inference backend, not the source of truth. Scientific workflows depend on provider-neutral capabilities such as `code_reasoning`, `structured_extraction`, and `mathematical_review`. Model/provider identity is stored in invocation manifests rather than encoded in artifact names.

## Configure a NIM endpoint

Set only the capabilities that exist in your deployment:

```bash
export NIM_BASE_URL=http://localhost:8000
export NIM_CODE_MODEL='<deployed-code-model-id>'
export NIM_EXTRACTION_MODEL='<deployed-extraction-model-id>'
export NIM_MATH_MODEL='<deployed-math-model-id>'

# Optional for authenticated endpoints
export NVIDIA_API_KEY='...'
```

Inspect configuration without contacting an endpoint:

```bash
neural-search models list
```

Probe the configured OpenAI-compatible `/v1/models` endpoint:

```bash
neural-search models doctor
```

Run one provider-neutral request:

```bash
neural-search inference run \
  --capability code_reasoning \
  --input-revision MouseLand/Kilosort@<commit> \
  --prompt-template audit_numeric_v1 \
  'Identify numerical assumptions in the supplied, retrieved component.'
```

Every returned result carries a `RunManifest` with provider, model, capability, prompt hash, request hash, timestamps, and optional source revision.

## Artifact provenance

Use `neural_search.inference.artifacts.write_json_artifact` for inference-derived artifacts. It writes both the deterministic JSON payload and a `.manifest.json` sidecar containing:

- SHA-256 of content;
- schema version;
- parent artifact IDs;
- all model invocation manifests;
- arbitrary lineage metadata.

Provider-specific filenames such as `*_ollama.jsonl` should be migrated toward provider-neutral artifact names. The provider belongs in the manifest.

## Audit evidence ladder

`neural_search.software.schema.AuditState` implements the following progression:

```text
DISCOVERED
  -> TRIAGED
  -> CODE_REVIEW_HYPOTHESIS
  -> STATICALLY_SUPPORTED
  -> MINIMAL_REPRODUCER
  -> NUMERICALLY_VERIFIED
  -> PIPELINE_VERIFIED
  -> OPEN_DATA_REANALYZED
  -> UPSTREAM_READY
  -> SUBMITTED
  -> MAINTAINER_ADJUDICATED
```

Terminal outcomes can occur from any non-terminal stage:

- false positive;
- already fixed;
- known behavior;
- intentional behavior;
- negligible effect;
- unreproducible;
- insufficient evidence;
- duplicate;
- withdrawn.

Two mechanical guardrails are important:

- `AuditHypothesis` cannot claim executable/numerical verification.
- `AuditFinding` cannot enter evidence-bearing states without one or more `VerificationRun` references.

## Verification

`neural_search.software.verification` executes pre-reviewed argv arrays with `shell=False`, a bounded timeout, captured stdout/stderr hashes, and explicit expected conditions. It is designed to sit inside a stronger container/job sandbox in production.

Verification strategies should prefer independent evidence:

1. analytic or high-precision oracle;
2. property/invariant testing;
3. differential implementation testing;
4. metamorphic tests;
5. synthetic ground-truth simulation;
6. controlled open-data pipeline reruns.

The language model proposes tests; executable evidence decides whether they pass.

## Audit prioritization

`score_audit_priority` ranks code paths using bounded, inspectable factors:

- scientific exposure;
- consequence if wrong;
- open-data reproducibility;
- methodological uncertainty;
- verification feasibility;
- already-addressed penalty;
- software/version mapping uncertainty penalty.

This score allocates reviewer effort. It is not evidence of a defect.

## Exposure and reanalysis

`find_reanalysis_candidates` joins a verified finding to paper/software usage and paper/dataset links. Generic package use is insufficient. A paper must map to either the affected component or an affected release. By default, candidate datasets must expose raw data.

This supports the central Neural Search query:

> Which published neuroscience analyses can be rerun on open data to test a verified software or methodological discrepancy?

## Upstream contribution gate

`ContributionPolicy` records a repository's preferred contribution channel and the documents that were reviewed: README, CONTRIBUTING, code of conduct, changelog/NEWS, issue template, and PR template.

`ContributionPacket` refuses to validate until:

- the finding is `UPSTREAM_READY` or later;
- executable verification exists;
- required project policy documents have been reviewed;
- a regression test is present when required.

`submission_payload()` additionally refuses to produce a public submission payload until `human_approved=True`.

This is deliberate. Public scientific-software contributions remain human-authorized even when AI is used for retrieval, code review, test design, or drafting.

## Knowledge graph extension

`neural_search.software.graph` projects domain records into KG nodes:

- `software_package`
- `software_release`
- `software_component`
- `audit_hypothesis`
- `verification_run`
- `software_audit_finding`
- `maintainer_decision`

Recommended relations include:

- `software_package_has_release`
- `software_package_has_component`
- `software_release_contains_component`
- `paper_uses_software`
- `paper_uses_software_component`
- `software_component_implements_method`
- `audit_hypothesis_concerns_component`
- `software_audit_finding_supported_by_verification`
- `software_audit_finding_affects_release`
- `paper_exposed_to_software_finding`
- `dataset_reanalysis_candidate_for_software_finding`
- `maintainer_decision_adjudicates_finding`

The software graph layer is additive to the existing neuroscience KG; it does not replace dataset, paper, finding, method, task, region, or provenance nodes.

## First prospective targets

Initial validation should deliberately span modalities rather than maximize repository count:

1. **Kilosort** — electrophysiology and Neuropixels spike sorting.
2. **Suite2p** — calcium imaging processing.
3. **AFNI or SPM** — mature neuroimaging/statistical pipelines.

Before prospective filing, build a historical benchmark from known, already-fixed defects. Freeze the repository immediately before each fix and measure component Recall@K, finding precision, reproducer success, patch correctness, regression-test adequacy, duplicate detection, and reviewer/maintainer acceptance.

## Definition of done for public contribution mode

Neural Search is not ready for prospective public contributions merely because the NIM endpoint works. Public contribution mode should remain disabled until the historical benchmark demonstrates acceptable precision and the full path below is exercised end to end:

```text
literature exposure
-> component retrieval
-> model hypothesis
-> adversarial disconfirmation
-> deterministic reproducer
-> independent numerical/pipeline verification
-> open-data impact assessment when feasible
-> contribution-policy review
-> regression test + patch
-> explicit human approval
```

The design goal is useful, reproducible scientific engineering, not PR volume.
