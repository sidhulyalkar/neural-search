"""Tests for typed finding field extractor (Task 8)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neural_search.literature.typed_finding_extractor import (
    _compiled,
    detect_negation,
    enrich_finding,
    extract_anatomical_directions,
    extract_behavioral_measures,
    extract_comparison_conditions,
    extract_computational_models,
    extract_conditions,
    extract_decoding_types,
    extract_developmental_stage,
    extract_dimensionality_reduction,
    extract_connectivity_type,
    extract_effect_scale,
    extract_frequency_bands,
    extract_genetic_tool,
    extract_injury_model,
    extract_metabolic_context,
    extract_molecular_marker,
    extract_network_coupling,
    extract_pharmacological_agent,
    extract_population_types,
    extract_sensory_stimulus,
    extract_signal_types,
    extract_sleep_stage,
    extract_social_affective,
    extract_spatial_frame,
    extract_statistical_relations,
    extract_synaptic_plasticity,
    extract_temporal_patterns,
    extract_typed_fields,
)


# ---------------------------------------------------------------------------
# Compiled-pattern cache (perf regression guard: 2026-06-24)
#
# enrich_finding() over the full ~190K-finding corpus took 30-45+ minutes
# before this cache existed, because _any()/_first() called re.search() with
# raw pattern strings -- recompiling on every call once this module's 500+
# distinct patterns exceeded CPython's shared, process-wide 512-entry regex
# cache. _compiled() is a dedicated, unbounded, module-scoped cache so each
# pattern compiles exactly once regardless of what else is sharing the
# global re cache.
# ---------------------------------------------------------------------------

class TestCompiledPatternCache:
    def test_same_pattern_returns_same_compiled_object(self):
        a = _compiled(r"\btheta\b")
        b = _compiled(r"\btheta\b")
        assert a is b

    def test_compiled_pattern_is_case_insensitive(self):
        pattern = _compiled(r"\btheta\b")
        assert pattern.search("THETA power increased")
        assert pattern.search("theta power increased")

    def test_different_patterns_compile_independently(self):
        a = _compiled(r"\btheta\b")
        b = _compiled(r"\bgamma\b")
        assert a is not b
        assert a.pattern != b.pattern


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

class TestNegation:
    def test_did_not_increase(self):
        assert detect_negation("Hippocampal theta did not increase during the task.") is True

    def test_no_significant_activation(self):
        assert detect_negation("There was no significant activation in the amygdala.") is True

    def test_failed_to_show(self):
        assert detect_negation("The drug failed to suppress gamma oscillations.") is True

    def test_counteractivation(self):
        assert detect_negation("PFC showed counteractivation relative to baseline.") is True

    def test_absence_of(self):
        assert detect_negation("Absence of theta was noted during sleep.") is True

    def test_unchanged(self):
        assert detect_negation("Beta power remained unchanged across conditions.") is True

    def test_inhibit(self):
        assert detect_negation("Optogenetic stimulation inhibited CA1 firing.") is True

    def test_positive_finding_not_negated(self):
        assert detect_negation("Theta power increased during spatial navigation.") is False

    def test_empty_string(self):
        assert detect_negation("") is False

    def test_no_significant_increase_is_negation(self):
        assert detect_negation("We found no significant increase in LFP power.") is True


# ---------------------------------------------------------------------------
# Frequency band
# ---------------------------------------------------------------------------

class TestFrequencyBand:
    def test_theta(self):
        assert "theta" in extract_frequency_bands("Theta oscillations in hippocampus")

    def test_gamma(self):
        assert "gamma" in extract_frequency_bands("Gamma-band synchrony during attention")

    def test_ripple_keyword(self):
        assert "ripple" in extract_frequency_bands("High-frequency ripple events during sleep")

    def test_sharp_wave(self):
        assert "ripple" in extract_frequency_bands("Sharp-wave ripples were detected in CA1")

    def test_multiple_bands(self):
        bands = extract_frequency_bands("Theta-gamma coupling in hippocampus")
        assert "theta" in bands
        assert "gamma" in bands

    def test_spindle_is_sigma(self):
        assert "sigma" in extract_frequency_bands("Sleep spindles were counted by sigma-band power")

    def test_no_match(self):
        assert extract_frequency_bands("The subject performed a navigation task") == []

    def test_cross_frequency(self):
        assert "cross_frequency" in extract_frequency_bands("Phase-amplitude coupling between theta and gamma")


# ---------------------------------------------------------------------------
# Temporal pattern
# ---------------------------------------------------------------------------

class TestTemporalPattern:
    def test_transient(self):
        assert "transient" in extract_temporal_patterns("A transient increase in spike rate was observed")

    def test_sustained(self):
        assert "sustained" in extract_temporal_patterns("Sustained activity in PFC during the delay period")

    def test_event_locked(self):
        assert "event_locked" in extract_temporal_patterns("Event-locked LFP responses were found")

    def test_phase_locked(self):
        assert "phase_locked" in extract_temporal_patterns("Spikes were phase-locked to theta rhythm")

    def test_oscillatory(self):
        assert "oscillatory" in extract_temporal_patterns("The circuit exhibited oscillatory dynamics")

    def test_delay_period(self):
        assert "delay_period" in extract_temporal_patterns("Neurons showed elevated activity during the delay period")

    def test_no_match(self):
        assert extract_temporal_patterns("Theta power increased.") == []

    def test_phasic(self):
        assert "transient" in extract_temporal_patterns("A phasic response was observed after stimulus")


# ---------------------------------------------------------------------------
# Spatial frame
# ---------------------------------------------------------------------------

class TestSpatialFrame:
    def test_inter_regional(self):
        frames = extract_spatial_frame("Coherence between hippocampus and PFC was enhanced")
        assert "inter_regional" in frames

    def test_laminar(self):
        frames = extract_spatial_frame("Layer 4 units showed enhanced firing rates")
        assert "laminar" in frames

    def test_whole_brain(self):
        frames = extract_spatial_frame("Whole-brain fMRI analysis revealed default-mode activation")
        assert "whole_brain" in frames

    def test_defaults_to_local(self):
        frames = extract_spatial_frame("CA1 theta power increased.")
        assert frames == ["local"]

    def test_columnar(self):
        frames = extract_spatial_frame("Columnar organization of orientation selectivity")
        assert "columnar" in frames


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------

class TestCondition:
    def test_resting_state(self):
        assert "resting_state" in extract_conditions("Default-mode activity during resting-state fMRI")

    def test_task(self):
        assert "task" in extract_conditions("During the choice trial, PFC neurons increased firing")

    def test_stimulation(self):
        assert "stimulation" in extract_conditions("Optogenetic stimulation of PFC disrupted memory")

    def test_lesion(self):
        assert "lesion" in extract_conditions("Hippocampal lesion impaired spatial memory")

    def test_pharmacological(self):
        assert "pharmacological" in extract_conditions("Drug infusion of muscimol blocked the effect")

    def test_multiple_conditions(self):
        conditions = extract_conditions("Drug injection during the task disrupted performance")
        assert "pharmacological" in conditions
        assert "task" in conditions

    def test_empty(self):
        assert extract_conditions("Neurons increased firing.") == []


# ---------------------------------------------------------------------------
# Signal type
# ---------------------------------------------------------------------------

class TestSignalType:
    def test_lfp(self):
        assert "lfp" in extract_signal_types("LFP recordings in hippocampus showed theta")

    def test_local_field_potential(self):
        assert "lfp" in extract_signal_types("Local field potential power increased in CA1")

    def test_spike(self):
        assert "spike" in extract_signal_types("Single-unit activity was recorded in PFC")

    def test_eeg(self):
        assert "eeg" in extract_signal_types("EEG coherence was measured between frontal and parietal electrodes")

    def test_bold(self):
        assert "bold" in extract_signal_types("BOLD signal in amygdala correlated with fear")

    def test_calcium(self):
        assert "calcium" in extract_signal_types("GCaMP calcium imaging in hippocampus")

    def test_ecog(self):
        assert "ecog" in extract_signal_types("ECoG recordings from temporal lobe")

    def test_neuropixels(self):
        assert "neuropixels" in extract_signal_types("Neuropixels probes recorded from 384 channels")

    def test_no_match(self):
        assert extract_signal_types("Subjects completed a spatial navigation task.") == []


# ---------------------------------------------------------------------------
# Statistical relation
# ---------------------------------------------------------------------------

class TestStatisticalRelation:
    def test_correlation(self):
        assert "correlation" in extract_statistical_relations("Theta power correlated with memory performance")

    def test_decoding(self):
        assert "decoding" in extract_statistical_relations("An SVM decoder classified choice direction with 85% accuracy")

    def test_anova(self):
        assert "anova" in extract_statistical_relations("A two-way ANOVA revealed a significant interaction")

    def test_regression(self):
        assert "regression" in extract_statistical_relations("Linear regression showed a significant beta coefficient")

    def test_granger(self):
        assert "granger_causality" in extract_statistical_relations("Granger causality analysis showed PFC driving hippocampus")

    def test_pca(self):
        assert "dimensionality_reduction" in extract_statistical_relations("PCA revealed a low-dimensional trajectory")

    def test_no_match(self):
        assert extract_statistical_relations("Theta increased during spatial navigation.") == []


# ---------------------------------------------------------------------------
# Top-level extract_typed_fields
# ---------------------------------------------------------------------------

class TestExtractTypedFields:
    _ALL_KEYS = (
        "negation", "frequency_band", "temporal_pattern", "spatial_frame",
        "condition", "signal_type", "statistical_relation",
        "behavioral_measure", "anatomical_direction", "effect_scale",
        "comparison_condition", "population_type",
        "synaptic_plasticity", "network_coupling", "decoding_type", "computational_model",
        "sensory_stimulus", "pharmacological_agent", "injury_model", "developmental_stage",
        "genetic_tool", "molecular_marker", "social_affective", "sleep_stage",
        "metabolic_context", "connectivity_type", "dimensionality_reduction",
    )

    def test_returns_all_keys(self):
        result = extract_typed_fields("Theta oscillations in hippocampus increased during reward learning.")
        for key in self._ALL_KEYS:
            assert key in result, f"Missing key: {key}"

    def test_negation_is_bool(self):
        result = extract_typed_fields("Firing did not increase.")
        assert isinstance(result["negation"], bool)
        assert result["negation"] is True

    def test_lists_are_lists(self):
        result = extract_typed_fields("Theta power correlated with LFP amplitude.")
        for key in ("frequency_band", "temporal_pattern", "spatial_frame", "condition", "signal_type", "statistical_relation"):
            assert isinstance(result[key], list), f"{key} should be a list"

    def test_empty_string_safe(self):
        result = extract_typed_fields("")
        assert result["negation"] is False
        assert result["frequency_band"] == []


# ---------------------------------------------------------------------------
# enrich_finding
# ---------------------------------------------------------------------------

class TestEnrichFinding:
    def test_does_not_mutate_input(self):
        record = {"finding_id": "f1", "finding_text": "Theta increased."}
        enriched = enrich_finding(record)
        assert "frequency_band" not in record  # original unchanged
        assert "frequency_band" in enriched

    def test_preserves_existing_fields(self):
        record = {"finding_id": "f1", "finding_text": "Theta increased.", "confidence": 0.9}
        enriched = enrich_finding(record)
        assert enriched["finding_id"] == "f1"
        assert enriched["confidence"] == 0.9
        assert "theta" in enriched["frequency_band"]

    def test_new_fields_all_present(self):
        result = extract_typed_fields("Parvalbumin interneurons showed LTP vs. baseline.")
        for key in ("behavioral_measure", "anatomical_direction", "effect_scale",
                    "comparison_condition", "population_type", "synaptic_plasticity",
                    "network_coupling", "decoding_type", "computational_model"):
            assert key in result, f"Missing new field: {key}"
            assert isinstance(result[key], list)

    def test_new_fields_empty_on_unrelated_text(self):
        result = extract_typed_fields("The quick brown fox jumped over the lazy dog.")
        for key in ("behavioral_measure", "anatomical_direction", "effect_scale",
                    "population_type", "synaptic_plasticity", "network_coupling",
                    "decoding_type", "computational_model"):
            assert result[key] == [], f"{key} should be empty for unrelated text"


# ===========================================================================
# New field tests
# ===========================================================================

class TestBehavioralMeasure:
    def test_accuracy(self):
        assert "accuracy" in extract_behavioral_measures("Accuracy improved after training in the task.")

    def test_reaction_time(self):
        assert "reaction_time" in extract_behavioral_measures("Reaction time was faster on rewarded trials.")

    def test_discriminability(self):
        assert "discriminability" in extract_behavioral_measures("d' increased significantly in the discrimination task.")

    def test_freezing(self):
        assert "freezing" in extract_behavioral_measures("Animals showed increased freezing during fear recall.")

    def test_locomotion(self):
        assert "locomotion" in extract_behavioral_measures("Running speed correlated with theta frequency.")

    def test_navigation(self):
        assert "navigation" in extract_behavioral_measures("Spatial navigation performance was impaired after lesion.")

    def test_choice(self):
        assert "choice" in extract_behavioral_measures("Choice accuracy varied with reward magnitude.")

    def test_licking(self):
        assert "licking" in extract_behavioral_measures("Licking responses were conditioned to tone onset.")

    def test_lever_press(self):
        assert "lever_press" in extract_behavioral_measures("Lever press rates increased under fixed-ratio schedule.")

    def test_no_match(self):
        assert extract_behavioral_measures("Theta power was elevated in hippocampus.") == []


class TestAnatomicalDirection:
    def test_dorsal(self):
        assert "dorsal" in extract_anatomical_directions("Dorsal hippocampus showed stronger theta oscillations.")

    def test_ventral(self):
        assert "ventral" in extract_anatomical_directions("Ventral CA1 activity correlated with anxiety.")

    def test_layer_2_3(self):
        assert "layer_2_3" in extract_anatomical_directions("L2/3 pyramidal cells showed burst firing.")

    def test_layer_5(self):
        assert "layer_5" in extract_anatomical_directions("Layer 5 output neurons projected to subcortical targets.")

    def test_ca1(self):
        assert "ca1" in extract_anatomical_directions("CA1 place cells fired reliably at specific locations.")

    def test_medial_lateral(self):
        dirs = extract_anatomical_directions("Medial entorhinal cortex, but not lateral, showed grid cell activity.")
        assert "medial" in dirs
        assert "lateral" in dirs

    def test_no_match(self):
        assert extract_anatomical_directions("Theta power increased during reward.") == []


class TestEffectScale:
    def test_strong(self):
        assert "strong" in extract_effect_scale("Robust increases in gamma power were observed.")

    def test_strong_significant(self):
        assert "strong" in extract_effect_scale("The effect was significantly larger in the drug condition.")

    def test_modest(self):
        assert "modest" in extract_effect_scale("A modest reduction in firing rate was detected.")

    def test_weak(self):
        assert "weak" in extract_effect_scale("Only a weak trend was observed in the data.")

    def test_trend(self):
        assert "trend" in extract_effect_scale("There was a trend toward significance (p=0.07).")

    def test_absent(self):
        assert "absent" in extract_effect_scale("No significant effect was found in the control condition.")

    def test_no_match(self):
        assert extract_effect_scale("Theta oscillations were present in CA1.") == []


class TestComparisonCondition:
    def test_vs_baseline(self):
        assert "vs_baseline" in extract_comparison_conditions("Activity increased vs. baseline during the task.")

    def test_vs_control(self):
        assert "vs_control" in extract_comparison_conditions("Drug group showed more errors compared to control.")

    def test_vs_sham(self):
        assert "vs_sham" in extract_comparison_conditions("LTP was impaired vs. sham-operated animals.")

    def test_within_subject(self):
        assert "within_subject" in extract_comparison_conditions("A within-subject crossover design was used.")

    def test_pre_post(self):
        assert "pre_post" in extract_comparison_conditions("We compared activity before and after training.")

    def test_between_group(self):
        assert "between_group" in extract_comparison_conditions("A between-group comparison revealed significant differences.")

    def test_vs_vehicle(self):
        assert "vs_vehicle" in extract_comparison_conditions("Muscimol-injected animals vs. vehicle-injected controls.")

    def test_no_match(self):
        assert extract_comparison_conditions("Theta power was elevated during navigation.") == []


class TestPopulationType:
    def test_pyramidal(self):
        assert "pyramidal" in extract_population_types("Pyramidal neurons in CA1 fired during place field traversal.")

    def test_parvalbumin(self):
        assert "parvalbumin" in extract_population_types("Parvalbumin-positive interneurons mediated gamma oscillations.")

    def test_pv_plus(self):
        assert "parvalbumin" in extract_population_types("PV+ cells showed fast-spiking activity.")

    def test_somatostatin(self):
        assert "somatostatin" in extract_population_types("SST+ interneurons preferentially targeted distal dendrites.")

    def test_place_cell(self):
        assert "place_cell" in extract_population_types("Place cells remapped in the novel environment.")

    def test_grid_cell(self):
        assert "grid_cell" in extract_population_types("Grid cells maintained their firing pattern across environments.")

    def test_dopaminergic(self):
        assert "dopaminergic" in extract_population_types("Dopaminergic neurons in VTA responded to reward prediction errors.")

    def test_interneuron_generic(self):
        assert "interneuron" in extract_population_types("GABAergic interneurons were recruited during gamma bursts.")

    def test_no_match(self):
        assert extract_population_types("Theta oscillations were recorded in hippocampus.") == []


class TestSynapticPlasticity:
    def test_ltp(self):
        assert "ltp" in extract_synaptic_plasticity("LTP was induced by high-frequency stimulation in CA1.")

    def test_long_term_potentiation(self):
        assert "ltp" in extract_synaptic_plasticity("Long-term potentiation was impaired in aged animals.")

    def test_ltd(self):
        assert "ltd" in extract_synaptic_plasticity("LTD was expressed at Schaffer-collateral synapses.")

    def test_stdp(self):
        assert "stdp" in extract_synaptic_plasticity("STDP was induced by pairing pre- and post-synaptic spikes.")

    def test_hebbian(self):
        assert "hebbian" in extract_synaptic_plasticity("Associative plasticity followed Hebbian learning rules.")

    def test_facilitation(self):
        assert "facilitation" in extract_synaptic_plasticity("Short-term facilitation was observed at the synapse.")

    def test_structural_plasticity(self):
        assert "structural_plasticity" in extract_synaptic_plasticity("Dendritic spine density increased after learning.")

    def test_no_match(self):
        assert extract_synaptic_plasticity("Theta power correlated with memory performance.") == []


class TestNetworkCoupling:
    def test_theta_gamma(self):
        assert "theta_gamma_coupling" in extract_network_coupling("Theta-gamma coupling was enhanced during working memory.")

    def test_replay(self):
        assert "replay" in extract_network_coupling("Hippocampal replay occurred during post-task sleep.")

    def test_reactivation(self):
        assert "replay" in extract_network_coupling("Sequential reactivation of place cells was observed during rest.")

    def test_entrainment(self):
        assert "entrainment" in extract_network_coupling("PFC neurons were entrained to hippocampal theta.")

    def test_up_down_state(self):
        assert "up_down_state" in extract_network_coupling("Up and down states alternated during slow-wave sleep.")

    def test_synchrony(self):
        assert "synchrony" in extract_network_coupling("Synchronized activity across areas increased during attention.")

    def test_spike_field_coherence(self):
        assert "spike_field_coherence" in extract_network_coupling("Spike-field coherence in the gamma band was elevated.")

    def test_no_match(self):
        assert extract_network_coupling("Accuracy improved after drug administration.") == []


class TestDecodingType:
    def test_svm(self):
        assert "svm" in extract_decoding_types("An SVM classifier decoded movement direction from PFC activity.")

    def test_lda(self):
        assert "lda" in extract_decoding_types("Linear discriminant analysis separated rewarded from unrewarded trials.")

    def test_bci(self):
        assert "bci" in extract_decoding_types("The BCI system decoded intended movement from motor cortex.")

    def test_population_decoder(self):
        assert "population_decoder" in extract_decoding_types("Population decoding of odor identity was highly accurate.")

    def test_cross_validated(self):
        assert "cross_validated" in extract_decoding_types("Accuracy was assessed by leave-one-out cross-validation.")

    def test_bayesian(self):
        assert "bayesian_decoder" in extract_decoding_types("A Bayesian decoder estimated position from place cell activity.")

    def test_information_transfer(self):
        assert "information_transfer_rate" in extract_decoding_types("The bit rate of the neural decoder reached 50 bits/second.")

    def test_no_match(self):
        assert extract_decoding_types("Theta power was elevated in hippocampus.") == []


class TestComputationalModel:
    def test_integrate_and_fire(self):
        assert "integrate_and_fire" in extract_computational_models("An integrate-and-fire model reproduced the observed firing patterns.")

    def test_hodgkin_huxley(self):
        assert "hodgkin_huxley" in extract_computational_models("Hodgkin-Huxley equations were used to model the action potential.")

    def test_rnn(self):
        assert "rnn" in extract_computational_models("A recurrent neural network was trained to perform the working memory task.")

    def test_attractor(self):
        assert "attractor" in extract_computational_models("Attractor dynamics in PFC maintained the working memory trace.")

    def test_drift_diffusion(self):
        assert "drift_diffusion" in extract_computational_models("A drift-diffusion model fit the reaction time distributions.")

    def test_rl_model(self):
        assert "rl_model" in extract_computational_models("Temporal difference learning captured the dopamine response.")

    def test_predictive_coding(self):
        assert "predictive_coding" in extract_computational_models("Predictive coding accounted for the neural prediction error signals.")

    def test_mean_field(self):
        assert "mean_field" in extract_computational_models("A mean-field approximation described the population dynamics.")

    def test_no_match(self):
        assert extract_computational_models("Theta power increased during memory encoding.") == []


# ===========================================================================
# New fields — experimental context / methodology / disease
# ===========================================================================

class TestSensoryStimulus:
    def test_visual_grating(self):
        assert "visual" in extract_sensory_stimulus("Neurons were tuned to orientation of drifting gratings.")

    def test_visual_evoked(self):
        assert "visual" in extract_sensory_stimulus("Visual evoked potentials were recorded in V1.")

    def test_auditory_tone(self):
        assert "auditory" in extract_sensory_stimulus("Pure tone stimuli at 8 kHz were delivered binaurally.")

    def test_auditory_noise(self):
        assert "auditory" in extract_sensory_stimulus("Noise burst responses were measured in auditory cortex.")

    def test_somatosensory_whisker(self):
        assert "somatosensory" in extract_sensory_stimulus("Whisker deflection evoked barrel cortex responses.")

    def test_somatosensory_tactile(self):
        assert "somatosensory" in extract_sensory_stimulus("Tactile stimulus was applied to the hindpaw.")

    def test_olfactory(self):
        assert "olfactory" in extract_sensory_stimulus("Odorant concentrations were varied in the olfactory stimulus set.")

    def test_multisensory(self):
        assert "multisensory" in extract_sensory_stimulus("Multisensory integration occurred in superior colliculus.")

    def test_nociceptive(self):
        assert "nociceptive" in extract_sensory_stimulus("Von Frey filaments measured mechanical nociceptive threshold.")

    def test_no_match(self):
        assert extract_sensory_stimulus("Theta power increased during working memory.") == []


class TestPharmacologicalAgent:
    def test_muscimol(self):
        assert "muscimol" in extract_pharmacological_agent("Muscimol infusion into PFC impaired working memory.")

    def test_ttx(self):
        assert "tetrodotoxin" in extract_pharmacological_agent("TTX was injected to block sodium channels locally.")

    def test_bicuculline(self):
        assert "bicuculline" in extract_pharmacological_agent("Bicuculline application increased hippocampal excitability.")

    def test_gabazine(self):
        assert "bicuculline" in extract_pharmacological_agent("Gabazine blocked GABA-A receptors in the slice.")

    def test_ap5(self):
        assert "ap5_nmda_antagonist" in extract_pharmacological_agent("AP5 bath application blocked LTP induction.")

    def test_ketamine(self):
        assert "ketamine" in extract_pharmacological_agent("Ketamine at sub-anesthetic dose altered gamma oscillations.")

    def test_scopolamine(self):
        assert "scopolamine" in extract_pharmacological_agent("Scopolamine injection impaired spatial memory consolidation.")

    def test_haloperidol(self):
        assert "haloperidol" in extract_pharmacological_agent("Haloperidol reduced stereotypy in dopamine-sensitized animals.")

    def test_amphetamine(self):
        assert "amphetamine" in extract_pharmacological_agent("Amphetamine increased locomotion and dopamine release.")

    def test_fluoxetine(self):
        assert "fluoxetine" in extract_pharmacological_agent("Chronic fluoxetine treatment restored BDNF levels.")

    def test_corticosterone(self):
        assert "corticosterone" in extract_pharmacological_agent("Elevated corticosterone impaired hippocampal LTP.")

    def test_kainic_acid(self):
        assert "kainic_acid" in extract_pharmacological_agent("Kainic acid injection induced status epilepticus.")

    def test_6ohda(self):
        assert "6ohda" in extract_pharmacological_agent("6-OHDA lesion of substantia nigra depleted striatal dopamine.")

    def test_no_match(self):
        assert extract_pharmacological_agent("Theta power increased during spatial navigation.") == []


class TestInjuryModel:
    def test_tbi_cci(self):
        assert "tbi_cci" in extract_injury_model("Controlled cortical impact produced focal TBI in rats.")

    def test_tbi_abbreviation(self):
        assert "tbi_general" in extract_injury_model("TBI caused persistent memory deficits.")

    def test_tbi_fpi(self):
        assert "tbi_fpi" in extract_injury_model("Fluid percussion injury produced diffuse axonal damage.")

    def test_stroke_mcao(self):
        assert "stroke_mcao" in extract_injury_model("MCAO for 90 minutes produced a reproducible cortical infarct.")

    def test_epilepsy_kainate(self):
        assert "epilepsy_kainate" in extract_injury_model("Kainic acid model produced chronic temporal lobe epilepsy.")

    def test_epilepsy_se(self):
        assert "epilepsy_se" in extract_injury_model("Status epilepticus was induced by systemic pilocarpine.")

    def test_parkinson_6ohda(self):
        assert "parkinson_6ohda" in extract_injury_model("6-OHDA lesion rats showed bradykinesia and tremor.")

    def test_parkinson_mptp(self):
        assert "parkinson_mptp" in extract_injury_model("MPTP model mice exhibited dopaminergic cell loss.")

    def test_alzheimer_app(self):
        assert "alzheimer_app" in extract_injury_model("5xFAD mice showed amyloid plaques by 2 months.")

    def test_depression_social_defeat(self):
        assert "depression_social_defeat" in extract_injury_model("Chronic social defeat stress produced anhedonia.")

    def test_no_match(self):
        assert extract_injury_model("Theta oscillations increased during spatial memory.") == []


class TestDevelopmentalStage:
    def test_neonatal(self):
        assert "neonatal" in extract_developmental_stage("Neonatal hippocampal lesion produced schizophrenia-like symptoms.")

    def test_postnatal_day(self):
        assert "neonatal" in extract_developmental_stage("Pups at postnatal day 7 showed altered theta coherence.")

    def test_juvenile(self):
        assert "juvenile" in extract_developmental_stage("Juvenile animals showed reduced fear extinction compared to adults.")

    def test_adolescent(self):
        assert "adolescent" in extract_developmental_stage("Adolescent stress altered adult HPA reactivity.")

    def test_aged(self):
        assert "aged" in extract_developmental_stage("Aged mice showed reduced place cell stability.")

    def test_aging(self):
        assert "aged" in extract_developmental_stage("Cognitive decline during aging correlated with CA3 hyperactivity.")

    def test_embryonic(self):
        assert "embryonic" in extract_developmental_stage("Embryonic cortical neurons expressed GABA depolarizing currents.")

    def test_no_match(self):
        assert extract_developmental_stage("Theta power increased during reward learning.") == []


class TestGeneticTool:
    def test_aav(self):
        assert "aav_vector" in extract_genetic_tool("AAV2/1-CaMKII-GFP was injected into the dorsal hippocampus.")

    def test_cre_lox(self):
        assert "cre_lox" in extract_genetic_tool("Floxed ChR2 was expressed in PV-Cre mice.")

    def test_dreadd(self):
        assert "dreadd" in extract_genetic_tool("hM4Di DREADD silenced CA1 neurons upon CNO injection.")

    def test_channelrhodopsin(self):
        assert "channelrhodopsin" in extract_genetic_tool("ChR2 activation drove theta entrainment in hippocampus.")

    def test_inhibitory_opsin(self):
        assert "inhibitory_opsin" in extract_genetic_tool("ArchT silenced pyramidal cells during the delay period.")

    def test_knockout(self):
        assert "knockout" in extract_genetic_tool("NR2B knockout mice showed impaired LTP at CA1 synapses.")

    def test_rnai(self):
        assert "rnai_knockdown" in extract_genetic_tool("siRNA knockdown of BDNF impaired hippocampal memory.")

    def test_crispr(self):
        assert "crispr" in extract_genetic_tool("CRISPR-Cas9 editing of the Arc gene eliminated IEG expression.")

    def test_fluorescent_reporter(self):
        assert "fluorescent_reporter" in extract_genetic_tool("GFP-expressing neurons were identified under two-photon imaging.")

    def test_no_match(self):
        assert extract_genetic_tool("Theta power correlated with working memory performance.") == []


class TestMolecularMarker:
    def test_cfos(self):
        assert "cfos_ieg" in extract_molecular_marker("c-Fos expression increased in amygdala after fear conditioning.")

    def test_arc(self):
        assert "arc" in extract_molecular_marker("Arc mRNA was elevated in hippocampus one hour after training.")

    def test_bdnf(self):
        assert "bdnf" in extract_molecular_marker("BDNF levels in hippocampus increased after exercise.")

    def test_camkii(self):
        assert "camkii" in extract_molecular_marker("CaMKII autophosphorylation was required for LTP maintenance.")

    def test_immunohistochemistry(self):
        assert "immunohistochemistry" in extract_molecular_marker("IHC confirmed co-localization of PV and ChR2.")

    def test_western_blot(self):
        assert "western_blot" in extract_molecular_marker("Western blot showed increased GluA1 after LTP induction.")

    def test_rna_seq(self):
        assert "rna_sequencing" in extract_molecular_marker("RNA-seq revealed transcriptomic changes in PFC after stress.")

    def test_single_cell_rna(self):
        assert "single_cell_rna" in extract_molecular_marker("scRNA-seq identified distinct interneuron subtypes in hippocampus.")

    def test_inflammatory_marker(self):
        assert "inflammatory_cytokine" in extract_molecular_marker("TNF-alpha and IL-1beta levels increased after TBI.")

    def test_no_match(self):
        assert extract_molecular_marker("Theta power increased during spatial memory.") == []


class TestSocialAffective:
    def test_restraint_stress(self):
        assert "restraint_stress" in extract_social_affective("Restraint stress for 2 hours elevated corticosterone.")

    def test_forced_swim_stress(self):
        assert "forced_swim_stress" in extract_social_affective("Forced swim stress activated the locus coeruleus.")

    def test_footshock(self):
        assert "footshock_stress" in extract_social_affective("Foot shock delivery triggered fear learning in the amygdala.")

    def test_social_defeat(self):
        assert "social_defeat" in extract_social_affective("Social defeat stress reduced hippocampal neurogenesis.")

    def test_elevated_plus_maze(self):
        assert "elevated_plus_maze" in extract_social_affective("Time in open arms of the elevated plus maze decreased after stress.")

    def test_open_field(self):
        assert "open_field" in extract_social_affective("Open field test revealed reduced exploration after ketamine.")

    def test_forced_swim_test(self):
        assert "forced_swim_test" in extract_social_affective("Immobility in the forced swim test was reduced by antidepressants.")

    def test_anhedonia(self):
        assert "anhedonia" in extract_social_affective("Sucrose preference test revealed anhedonia in stressed animals.")

    def test_social_recognition(self):
        assert "social_recognition" in extract_social_affective("Social recognition memory was impaired in OXT knockout mice.")

    def test_three_chamber(self):
        assert "three_chamber_social" in extract_social_affective("Three-chamber test revealed reduced sociability in Shank3 KO.")

    def test_no_match(self):
        assert extract_social_affective("Theta power increased during spatial navigation.") == []


class TestSleepStage:
    def test_nrem(self):
        assert "nrem" in extract_sleep_stage("Sharp-wave ripples occurred during NREM sleep.")

    def test_slow_wave_sleep(self):
        assert "nrem" in extract_sleep_stage("Memory reactivation was observed during slow-wave sleep.")

    def test_rem(self):
        assert "rem" in extract_sleep_stage("REM sleep was characterized by theta oscillations in hippocampus.")

    def test_sleep_consolidation(self):
        assert "sleep_consolidation" in extract_sleep_stage("Post-learning sleep consolidation required hippocampal replay.")

    def test_sleep_dependent(self):
        assert "sleep_dependent" in extract_sleep_stage("Sleep-dependent memory consolidation was impaired by REM deprivation.")

    def test_quiet_wake(self):
        assert "quiet_wake" in extract_sleep_stage("Ripples were detected during quiet waking as well as NREM sleep.")

    def test_sleep_wake_cycle(self):
        assert "sleep_wake_cycle" in extract_sleep_stage("Circadian regulation of hippocampal plasticity was demonstrated.")

    def test_no_match(self):
        assert extract_sleep_stage("PFC neurons fired during the delay period of a working memory task.") == []


class TestMetabolicContext:
    def test_neuroinflammation(self):
        assert "neuroinflammation" in extract_metabolic_context("Neuroinflammation was elevated in the hippocampus after TBI.")

    def test_microglial_activation(self):
        assert "microglial_activation" in extract_metabolic_context("Microglial activation was measured by Iba1 immunostaining.")

    def test_tnf_alpha(self):
        assert "tnf_alpha" in extract_metabolic_context("TNF-alpha was elevated in plasma after systemic LPS injection.")

    def test_il1_beta(self):
        assert "il1_beta" in extract_metabolic_context("IL-1beta infusion impaired hippocampal LTP.")

    def test_oxidative_stress(self):
        assert "oxidative_stress" in extract_metabolic_context("Oxidative stress markers increased in aged cortex.")

    def test_bbb(self):
        assert "blood_brain_barrier" in extract_metabolic_context("Blood-brain barrier disruption occurred after MCAO.")

    def test_mitochondria(self):
        assert "mitochondria" in extract_metabolic_context("Mitochondrial dysfunction reduced ATP production in neurons.")

    def test_apoptosis(self):
        assert "apoptosis" in extract_metabolic_context("Caspase-3 activation confirmed apoptosis in the lesioned region.")

    def test_no_match(self):
        assert extract_metabolic_context("Theta oscillations increased during spatial exploration.") == []


class TestConnectivityType:
    def test_functional_connectivity(self):
        assert "functional_connectivity" in extract_connectivity_type("Resting-state functional connectivity between PFC and hippocampus increased.")

    def test_effective_connectivity(self):
        assert "effective_connectivity" in extract_connectivity_type("Dynamic causal modelling revealed top-down effective connectivity from PFC.")

    def test_structural_connectivity(self):
        assert "structural_connectivity" in extract_connectivity_type("DTI tractography mapped structural connectivity in white matter tracts.")

    def test_corticostriatal(self):
        assert "corticostriatal" in extract_connectivity_type("Corticostriatal synchrony increased during reward learning.")

    def test_hippocampal_cortical(self):
        assert "hippocampal_cortical" in extract_connectivity_type("CA1-PFC coherence was elevated during working memory.")

    def test_thalamocortical(self):
        assert "thalamocortical" in extract_connectivity_type("Thalamocortical spindles synchronized cortical activity during NREM.")

    def test_top_down(self):
        assert "top_down" in extract_connectivity_type("Top-down feedback from PFC suppressed V1 responses during attention.")

    def test_feedforward(self):
        assert "feedforward" in extract_connectivity_type("Feedforward projections from V1 drove responses in V4.")

    def test_long_range(self):
        assert "long_range_projection" in extract_connectivity_type("Long-range connections from hippocampus to PFC were potentiated after learning.")

    def test_no_match(self):
        assert extract_connectivity_type("Theta power increased during spatial memory.") == []


class TestDimensionalityReduction:
    def test_pca(self):
        assert "pca" in extract_dimensionality_reduction("Principal component analysis revealed a low-dimensional trajectory.")

    def test_pca_abbreviation(self):
        assert "pca" in extract_dimensionality_reduction("PCA of the population activity showed clear task-related structure.")

    def test_ica(self):
        assert "ica" in extract_dimensionality_reduction("Independent component analysis separated EEG artifacts from signals.")

    def test_tsne(self):
        assert "tsne" in extract_dimensionality_reduction("t-SNE visualization revealed distinct cell type clusters.")

    def test_umap(self):
        assert "umap" in extract_dimensionality_reduction("UMAP embedding showed continuous manifold structure in the population code.")

    def test_dpca(self):
        assert "dpca" in extract_dimensionality_reduction("Demixed PCA (dPCA) separated task and choice components.")

    def test_factor_analysis(self):
        assert "factor_analysis" in extract_dimensionality_reduction("Factor analysis identified latent variables underlying population variance.")

    def test_manifold(self):
        assert "manifold_learning" in extract_dimensionality_reduction("Manifold learning revealed a ring-like geometry of head direction cells.")

    def test_state_space(self):
        assert "state_space" in extract_dimensionality_reduction("A state space model tracked the latent cognitive state during the task.")

    def test_no_match(self):
        assert extract_dimensionality_reduction("Theta power increased during spatial navigation.") == []
