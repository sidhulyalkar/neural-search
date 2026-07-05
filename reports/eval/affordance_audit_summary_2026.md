# Affordance Precision Audit Summary (2026-06-24)

## Headline finding: the documented affordance registry and the live corpus use two disjoint vocabularies

Before any precision question could be asked, sampling against the affordances named in
`PEER_VALIDATION_PROTOCOL.md` ("21 analysis affordance types... `neural_search/analysis_affordances.py`")
returned **zero matching datasets for all 8 sampled affordance types** (`q_learning`,
`choice_decoding`, `trial_aligned_neural`, `functional_connectivity`, `seizure_detection`,
`speech_decoding`, `delay_discounting_modeling`, `motor_decoding`).

Investigating why: `neural_search/analysis_affordances.py::AFFORDANCE_IDS` defines 18 values
(`event_aligned_activity`, `trial_averaged_response`, `choice_decoding`, `motor_decoding`,
`speech_decoding`, `q_learning_modeling`, `state_space_modeling`, `cross_modal_prediction`,
`brain_behavior_alignment`, `seizure_detection`, `sleep_stage_classification`, `fmri_glm_analysis`,
`functional_connectivity`, `representational_similarity_analysis`, `encoding_modeling`,
`bci_decoding`, `latent_dynamics_modeling`, `aperiodic_spectral_parameterization`) — already not
21, and already missing `delay_discounting_modeling` and `trial_aligned_neural` entirely (the
audit-sampler script's target list had drifted even from the registry it claimed to sample).

But the real gap is deeper. A direct comparison against every value actually present in
`data/corpus/normalized/combined_corpus.jsonl/full_corpus_v09.jsonl`'s `analysis_affordances` field
(990/7,171 records have a non-empty value, drawn from 19 distinct values: `population_coding`,
`stimulus_response`, `calcium_event_detection`, `cell_type_characterization`, `spike_sorting`,
`morphology_analysis`, `brain_image_analysis`, `microscopy_image_registration`,
`population_decoding`, `cross_lab_reproducibility`, `decision_variable_tracking`,
`neuron_model_optimization`, `electrophysiology_feature_extraction`, `ion_channel_analysis`,
`multi_area_coordination`, `cell_counting`, `brain_wide_mapping`, `morphological_feature_extraction`)
shows **zero overlap** with the registry's 18 IDs. Not partial overlap — zero.

**Conclusion: the rule-based affordance detector documented as the system's affordance source has
never populated this corpus snapshot, or its output was excluded when `full_corpus_v09.jsonl` was
built.** The 990 records that do carry affordances were tagged by some other, undocumented process
with a different vocabulary — appears to be morphology/imaging/electrophysiology-method oriented,
likely specific to the NeuroMorpho/Allen/BlueBrain/brain_image_library sources that dominate the
match counts. `neural_search/analysis_affordances.py` is wired into `enrich_corpus.py` per a grep of
call sites, so the detector exists and is connected to *something* — but not, apparently, to v09.

This is a more significant finding than a precision percentage: **the whitepaper's "21 analysis
affordance types" claim describes a system whose output is absent from the actual served corpus.**
Recommend checking whether a newer corpus build (post-v09) actually carries the registry's
affordances, or whether `enrich_corpus.py` needs to be re-run against v09 specifically.

## What was actually auditable: precision on the live vocabulary

Re-targeted the sampler at the 8 most frequent values that actually exist in the corpus
(`scripts/eval/sample_affordances_for_audit.py`, updated in place — old target list preserved in
the file's audit note for traceability). Sampled 15 datasets per affordance (120 rows total),
judged against each dataset's own modalities/tasks/species/title metadata (no file inspection).

| Affordance | n | Correct | Notes |
|---|---|---|---|
| `population_coding` | 15 | 15/15 | Allen 2P calcium imaging sessions — inherently multi-cell |
| `stimulus_response` | 15 | 15/15 | All explicitly tagged `visual_stimulation`/`change_detection` |
| `calcium_event_detection` | 15 | 15/15 | Standard analysis for the dF/F-style calcium traces these sessions provide |
| `cell_type_characterization` | 15 | 15/15 | Cre-driver line is itself a genetic cell-type label, named in the title |
| `spike_sorting` | 15 | 15/15 | Allen/IBL raw multi-channel Neuropixels traces — exactly spike-sorting's input |
| `morphology_analysis` | 15 | 15/15* | See caveat below |
| `brain_image_analysis` | 15 | 15/15 | Tomography, Nissl histology, MERSCOPE spatial imaging — all genuine brain imaging |
| `microscopy_image_registration` | 15 | 15/15 | Brain-wide tomography + an explicit "Reference Brain" atlas dataset |

**120/120 on the live vocabulary** — far above the 80% whitepaper-citation threshold. This is a
metadata-only judgment (no file inspection), consistent with the audit's own `support_type` caveat.

## One real caveat found inside the "live vocabulary" sample

12 of the 15 `morphology_analysis` rows (and most of `brain_image_analysis`/
`microscopy_image_registration`) are the same `brain_image_library` AAV-tracing dataset family,
repeated across many specimen IDs, with **modality tagged `calcium_imaging`** even though the
described technique — Tissuecyte 2-photon tomography for brain-wide axon/projection labeling — is
an anatomical tracing method, not functional calcium dynamics imaging. The `morphology_analysis`
affordance label itself is still defensible (brain-wide projection-pattern labeling is a
morphological/anatomical question), but the **modality field looks mislabeled** for this entire
dataset family. Worth checking the brain_image_library ingestion adapter's modality-inference logic.

## Sample diversity caveat

The corpus's 990 affordance-tagged records skew heavily toward Allen Brain Observatory (2P/Neuropixels)
and `brain_image_library` AAV-tracing specimens — the 15-per-affordance sample is consequently far
less diverse than the raw row count suggests (many rows are different specimens of the same parent
study, with near-identical titles). A future, larger audit pass should stratify by source as well
as affordance to avoid one collection dominating the judged sample.

## Caveat

LLM-judge pass (Claude Sonnet 4.6), metadata-only — no file inspection or linked-paper
verification, consistent with `support_type=metadata_only` in every row. The 8-affordance,
zero-overlap taxonomy mismatch is the most load-bearing finding here and doesn't depend on judgment
calls; the 120/120 precision figure is a secondary result on a sampler that had to be re-targeted
first.
