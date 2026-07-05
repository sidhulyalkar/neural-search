# Aperiodic Spectral QC Policy

`neural_search.spectral.qc.assess_spectral_qc(...)` produces one of three
statuses for every `SpectralEstimate`:

* **`pass`** — no flags triggered. The fit can be used with normal scrutiny.
* **`warn`** — at least one non-severe flag triggered. The fit is usable but
  should be visibly flagged (dataset cards, search results, and any
  downstream claim should surface the warning) and weighted down in
  aggregate analyses.
* **`fail`** — at least one severe flag triggered. Treat the numeric
  outputs (offset, exponent, knee, peaks) as unreliable; exclude from
  aggregate analyses by default.

A bundle's `overall_qc_status` is the worst status across its estimates
(`fail` > `warn` > `pass`).

## Flags

| Flag | Severity | Trigger | Rationale |
|---|---|---|---|
| `low_fit_r_squared` | warn below 0.9, contributes to fail below 0.7 | model R² below threshold | Poor fit to the observed spectrum |
| `high_fit_error` | warn above 0.3, contributes to fail above 0.6 | RMSE of the log-power fit residual | Large average deviation from the model |
| `too_few_frequency_bins` | **fail** | fewer than 10 bins (fail below 5) | Not enough resolution to constrain offset/exponent (and knee, if used) |
| `frequency_range_too_narrow` | **fail** | span below 5 Hz (fail below 2 Hz) | A narrow band cannot distinguish a power-law slope from noise |
| `line_noise_overlap` | warn | fit range overlaps 48-52 Hz or 58-62 Hz | Mains line noise can masquerade as or contaminate a periodic peak |
| `many_peaks_possible_overfit` | warn | more than 6 peaks fitted | Likely fitting noise as oscillatory structure |
| `missing_sampling_rate` | warn | sample rate unknown | Cannot map fit results to calibrated Hz with full confidence |
| `missing_channel_metadata` | warn | channel/probe metadata unavailable | Cannot attribute the estimate to a specific recording site |
| `missing_region_metadata` | warn | brain region metadata unavailable | Cannot attribute the estimate to a specific brain region |
| `missing_task_state` | warn | task/behavioral state unknown | Aperiodic features are state-dependent (e.g. sleep vs. awake); without this, comparisons across estimates are confounded |
| `flat_or_zero_signal` | **fail** | raw signal has ~zero variance | No real spectral content to fit |
| `nan_or_inf_values` | **fail** | raw signal contains NaN/Inf | Fit would be computed on corrupted data |

Thresholds are defined as module-level constants in `qc.py`
(`MIN_R_SQUARED_WARN`, `MIN_R_SQUARED_FAIL`, `MAX_FIT_ERROR_WARN`,
`MAX_FIT_ERROR_FAIL`, `MIN_FREQUENCY_BINS_WARN`,
`MIN_FREQUENCY_BINS_FAIL`, `MIN_FREQ_RANGE_WARN_HZ`,
`MIN_FREQ_RANGE_FAIL_HZ`, `MAX_PEAKS_BEFORE_OVERFIT_WARNING`,
`LINE_NOISE_BANDS_HZ`) — adjust there, not inline, if a project needs
different conservatism.

## Why missing-metadata flags are `warn`, not `fail`

A fit can be numerically sound (good R², reasonable error, enough bins)
while still lacking the contextual metadata (channel, region, task state,
sampling rate) needed to *interpret* it safely. We deliberately keep these
as `warn` rather than `fail`: the number itself may be fine, but any claim
built on top of it should explicitly note what's missing. Treating missing
context as an automatic hard failure would make the QC system reject
otherwise-valid numerical fits; treating it as silent would let
under-contextualized claims slip through unflagged. `warn` forces the
caveat to stay attached to the result.

## How QC interacts with search ranking and dataset cards

* Dataset cards (`spectral_phenotype.qc_summary` /
  `overall_qc_status`) always show the pass/warn/fail breakdown alongside
  the mandatory `interpretation_cautions` — never just a bare exponent.
* Search ranking boosts only consider a dataset "ready" for the
  `aperiodic_spectral_parameterization` affordance based on **eligibility**
  (metadata-only), not on a specific QC outcome of a prior run — a `fail`
  on one run does not retroactively make a dataset ineligible, since QC
  failure can be a property of one signal/segment, not the dataset as a
  whole. Per-estimate QC should gate *interpretation*, not *discoverability*.
