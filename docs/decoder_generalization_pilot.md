# Decoder generalization pilot: an evidence-first study

This is an executable **first research slice**, not a claim that decoder
failure mechanisms have already been discovered or causally explained.

## Scientific question

How much does performance of a fixed, simple EEG decoder change when it
encounters a new trial (same subject/session), another recording session
(same subject), or a previously unseen subject?

Three evaluation protocols address different generalization questions:

| Protocol | Training / testing | Interpretable as |
| --- | --- | --- |
| within_session | stratified held-out trials inside each subject-session | performance on unseen trials of the same recording session |
| cross_session | train on other sessions of a subject, evaluate on one wholly withheld session | transfer to a new session of a known person |
| cross_subject | train on other subjects, evaluate on one wholly withheld subject | zero-shot transfer to an unseen person |

These protocols use different training sets and are **not** a controlled
causal isolation of session or subject shift. Report per-person variation
and absolute scores, not just the difference between their averages.

## Existing research context

- BNCI2014_001 (BCI Competition IV 2a): nine participants, two sessions,
  motor imagery; we select the **left/right** two-class MOABB paradigm.
  https://moabb.neurotechx.com/docs/generated/moabb.datasets.BNCI2014_001.html
- MOABB evaluation and reproducibility context:
  https://moabb.neurotechx.com/docs/paper_results.html
- Jayaram and Barachant, 2018, MOABB trustworthy benchmarking:
  https://pubmed.ncbi.nlm.nih.gov/30177583/

## Quick start: no download, no paid API

```bash
python -m neural_search.decoder_generalization \
  --source synthetic --output reports/decoder_generalization/synthetic.json

# Equivalent through the installed top-level CLI:
neural-search decoder-study --source synthetic \
  --output reports/decoder_generalization/synthetic.json

pytest -q tests/test_decoder_generalization.py
```

The synthetic generator deliberately introduces channel-gain and
subject-dependent changes to exercise the pipeline. To inject a known,
non-biological channel-layout failure and test whether cross-session
generalization degrades, run:

```bash
neural-search decoder-study --source synthetic \
  --synthetic-shift channel_swap \
  --output reports/decoder_generalization/synthetic_channel_swap.json
```

The swapped-channel scenario is a controlled synthetic fault injection:
it models an inconsistent channel mapping, **not** biological learning
or subject-specific neural nonstationarity. Its results **must
not** be interpreted as empirical neural generalization evidence.

## Real publicly available data (explicit network access)

```bash
pip install -e '.[decoder]'
neural-search decoder-study --source moabb --subjects 1 2 3 \
  --output reports/decoder_generalization/bnci2014_001_3subjects.json

# Full 9-subject run once initial inspection and audit succeed:
neural-search decoder-study --source moabb --subjects 1 2 3 4 5 6 7 8 9 \
  --output reports/decoder_generalization/bnci2014_001_full.json
```

MOABB may download external EEG files. The study does not make paid
LLM calls or rent GPU instances. The first real run should inspect the
actual MOABB metadata and verify session and class distribution.
All file use and dataset publication must respect source terms.

The adapter obtains epochs through MOABB's `LeftRightImagery` paradigm,
with subject/session metadata. Preprocessing associated with epoch
creation comes from MOABB; all **learned decoder standardization and
regularized covariance** is fit separately on training partitions.
If changing the paradigm, version the preprocessing contract.

## Custom epoched neural data

Provide an `.npz` with:

- `X`: numeric finite `(trial, channel, time)`, each trial a separate epoch;
- `y`: labels, one per trial;
- `subject`: stable subject identifiers, one per trial;
- `session`: stable session identifiers, one per trial.

Avoid object/pickled arrays. A source-specific preparation script
should independently audit acquisition, channel references, event
alignment, recording segmentation, preprocessing, and the origin of
each epoch. Overlapping windows may constitute leakage even when their
byte hashes differ: the current exact-content guard cannot catch this.

```bash
neural-search decoder-study --source npz --npz /path/to/epochs.npz \
  --output reports/decoder_generalization/other_dataset.json
```

## Implemented model and measurements

The baseline uses per-channel log-variance of epochs and diagonal-shrunk
pooled-covariance linear discriminant analysis. There is no hyperparameter
search, pretrained component, subject-ID feature, or use of test data for
train-fitted scaling or covariance.

Per fold: group IDs, train/test counts, confusion matrix, accuracy,
balanced accuracy, descriptive aggregate standardized mean feature shift,\nand retrospective class-conditional feature shift (computed using test\nlabels **after** prediction, not used for model fitting).
The report includes same-subject within-session references and
cross-session gaps, exact-trial leakage checks, a content fingerprint,
elapsed time, model parameters, and limitations.

Not currently measured: CSP, Riemannian baselines, calibration error,
epoch overlap, spectral-shift attribution, label quality,
electrode montage changes, covariate-matched causal interventions,
confidence intervals over independent subjects, or statistical
significance across multiple datasets.

**Do not equate feature shift with neural causation.** A shift may
reflect acquisition configuration, artifacts, protocol differences,
session drift, or preprocessing.

## Baseline acceptance gates

1. Every held-out fold must exclude the evaluated session or subject
   as specified. Byte-identical cross-split trials hard-fail.
2. Inadequate class support is explicitly skipped, never silently scored.
3. Repeated seeded runs reproduce the same fold-level metrics and content hash.
4. Human review confirms label mapping, session boundaries, channels,
   preprocessing and epoch provenance for each real dataset.
5. Report subject-level paired gaps, not just a pooled accuracy number.
6. An empirical neuroscience claim requires independent real-data
   replication and a stronger baseline than this initial log-variance LDA.

## Next scientific experiments (not yet implemented)

1. Freeze and validate BNCI2014_001 against MOABB's relevant protocol;
   generate independent human-reviewed dataset cards.
2. Compare CSP+LDA, filter-bank CSP, regularized covariance/tangent-space,
   and a lightweight EEGNet under equal data and tuning budgets.
3. Add recording/run boundaries and overlapping-window/near-duplicate
   leakage detection. Validate that preprocessing never learns from held-out
   sessions or subjects.
4. Pre-register candidate explanations: class-prior shift, spectral
   shift, channel covariance drift, electrode montage/reference changes,
   artifacts, subject variability and nonstationarity.
5. Test each hypothesis with a targeted intervention or controlled
   sensitivity analysis. Distinguish prediction of failure from causal
   explanation.
6. Replicate on BNCI2014_004, OpenBMI/Lee2019 or another compatible
   multi-session dataset, using **dataset-specific** protocol maps.
7. Report paired uncertainty across independent subjects, failure cases,
   reproducibility capsules, and actual compute/energy/cost telemetry.

## Resource and cost expectations

Synthetic mode: CPU only, usually seconds on a recent laptop for hundreds
of small epochs, $0 incremental paid API/GPU cost. MOABB real mode:
data downloads, epoch processing and repeated LDA fits are CPU-bound;
actual wall time depends on network speed, dataset cache and CPU. No
credible timed estimate can be asserted until a measured pilot run.

For every run record wall time (included now), data download bytes/time,
CPU/memory, and optional cloud billing (future), and do not invent a
monetary price for local computation. No experiment can automatically
spend donor funds in this pilot.

## Relationship to Neural Search

- Register source datasets and paper links through existing ingestion,
  literature, and provenance infrastructure.
- Preserve this independent experiment artifact. Link it into the
  knowledge graph only after a scientific reviewer accepts its context
  and evaluation protocol.
- The next layer can convert repeated verified failure patterns into
  versioned research opportunities and funded experiment proposals.
- This initial module is intentionally additive to avoid conflict with
  the stacked runtime/artifact PRs #10/#11 and NIM PR #12.
