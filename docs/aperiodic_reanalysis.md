# Aperiodic Spectral Phenotype Reanalysis

This document describes the `neural_search.spectral` package: a conservative,
reproducible layer for determining whether datasets can support aperiodic
("1/f") spectral parameterization, computing standardized spectral features
when they can, and exporting the results into the knowledge graph and search
ranking pipeline.

## Scope

The package answers three separate questions, on purpose kept separate so
that a "no" to one never silently downgrades another:

1. **Eligibility** (`neural_search.spectral.eligibility`) — from existing
   normalized dataset metadata alone (modality, usability flags, missing
   fields), is this dataset *plausibly* analyzable with aperiodic spectral
   parameterization? This never touches raw signal data.
2. **Computation** (`psd.py`, `specparam_backend.py`, `irasa_backend.py`,
   `features.py`) — given a raw or precomputed power spectrum, fit an
   aperiodic background (offset, exponent, optional knee) plus periodic
   peaks, with full provenance of the method and settings used.
3. **Quality control** (`qc.py`) — was *this particular fit* trustworthy
   enough to act on?

A dataset can be "eligible" (question 1) without ever having been computed
(question 2), and a computed estimate can fail QC (question 3) without that
invalidating the dataset's eligibility.

## Why this is conservative by design

* Eligibility detection only reasons about metadata that is already present
  on a `NormalizedDatasetRecord` — it never infers "raw signal exists"
  from a dataset's *name* or guesses a sampling rate.
* `unsupported` is reserved for modalities that are structurally
  incompatible (fMRI BOLD's ~1 Hz effective sampling cannot resolve
  aperiodic spectral structure in the 1-100 Hz range used by FOOOF/specparam;
  behavior-tracking-only and purely anatomical/structural datasets carry no
  continuous neural signal at all). `unknown` is used when there simply is
  no modality evidence to reason from — that is different from "no", and
  callers should not treat it as a rejection.
* Every computed `SpectralEstimate` carries the `SpectralRunConfig` that
  produced it (backend name + version, frequency range, peak-fitting
  settings, sample rate, random seed) and a `SpectralQCAssessment`. Nothing
  is reported as a bare number.

## Backends

| Backend | Package | Notes |
|---|---|---|
| `mock` | none (numpy only) | Log-log linear regression with a robust second pass that excludes peak-contaminated bins, plus simple residual peak-picking. Used by default, in tests, and as the automatic fallback. |
| `specparam` / `fooof` | optional `specparam` (or legacy `fooof`) package | True iterative aperiodic/periodic separation. `neural_search.spectral.specparam_backend.get_backend("specparam")` falls back to `mock` automatically if neither package is installed. |
| `irasa` | none (numpy + optional scipy) | A simplified, self-contained IRASA (resampling-based) implementation operating directly on the raw signal. Not a byte-for-byte port of the published algorithm or `yasa.irasa` — prefer `yasa` for publication-grade IRASA. |

Install `neural-search[spectral]` to get the optional `specparam`/`yasa`
packages; the codebase works correctly (with reduced accuracy) without them.

## Critical scientific caveats — read before interpreting any output

**The aperiodic exponent / spectral slope is a descriptive statistic about
the shape of a power spectrum. It is *not*, by itself, a validated direct
measurement of excitation/inhibition (E/I) balance, "neural noise," or any
single cellular or circuit mechanism.** The literature relating spectral
slope to E/I balance is based on specific computational models and specific
experimental paradigms; treating a steeper or flatter exponent as proof of a
particular mechanism in a new dataset, species, or recording configuration
is an overinterpretation that this codebase does not endorse.

Every `SpectralFeatureBundle` carries `interpretation_cautions` — do not
strip these out of downstream reporting. At minimum:

* Aperiodic features are sensitive to recording modality, reference scheme,
  electrode/probe placement, behavioral/arousal state, and preprocessing
  choices. Cross-dataset, cross-species, or cross-condition comparisons
  require matched methods and explicit caveats, not a single shared
  conclusion.
* QC status should gate confidence: discount or exclude `warn`/`fail`
  estimates before drawing scientific conclusions. See
  `docs/aperiodic_qc_policy.md`.
* Confidence in eligibility (question 1 above) is about *whether a fit can
  plausibly be computed*, not about how scientifically meaningful that fit
  would be once computed.

## Module map

```
neural_search/spectral/
  schemas.py            Pydantic models: AperiodicEligibility, SpectralRunConfig,
                         SpectralEstimate, PeriodicPeak, SpectralQCAssessment,
                         SpectralFeatureBundle
  eligibility.py         detect_aperiodic_eligibility(record) -> AperiodicEligibility
  synthetic.py           Synthetic 1/f spectra/signals with known ground truth
  psd.py                 Welch PSD (scipy if available, numpy fallback)
  specparam_backend.py   MockSpectralParamBackend, SpecparamBackend/FooofBackend
  irasa_backend.py       IrasaBackend (simplified IRASA)
  qc.py                  assess_spectral_qc(...) -> SpectralQCAssessment
  features.py            compute_spectral_estimate, build_feature_bundle, summarize_for_card
  kg.py                  build_spectral_subgraph(bundle) -> KnowledgeGraph
  search_features.py     Trigger-term constants + explain helper for search ranking
```

See also `docs/spectral_phenotype_graph.md` (knowledge-graph schema) and
`docs/aperiodic_qc_policy.md` (QC flag definitions and thresholds).

## Running a reanalysis

```bash
python scripts/reanalysis/run_aperiodic_one.py --dataset-id dataset:dandi:000001 --signal-npy path/to/signal.npy --sample-rate 1000
python scripts/reanalysis/run_aperiodic_batch.py --manifest path/to/manifest.jsonl --out artifacts/spectral/bundles.jsonl
python scripts/kg/ingest_spectral_features.py --bundles artifacts/spectral/bundles.jsonl --out artifacts/graph/spectral_subgraph.json
python scripts/eval/evaluate_aperiodic_synthetic.py
```
