"""Rule-based typed field extractor for neuroscience findings.

Extends the normalized FindingRecord with 27 typed fields:

  ORIGINAL (7):
    frequency_band      — theta / gamma / beta / alpha / delta / ripple / broadband
    temporal_pattern    — transient / sustained / oscillatory / event-locked / phase-locked
    negation            — True if the sentence explicitly negates the finding
    spatial_frame       — local / inter_regional / whole_brain / laminar / columnar
    condition           — resting_state / task / stimulation / lesion / pharmacological
    signal_type         — lfp / spike / eeg / bold / calcium / ecog / neuropixels
    statistical_relation — correlation / regression / anova / t_test / decoding

  BEHAVIORAL / ANATOMICAL (5):
    behavioral_measure  — accuracy / reaction_time / choice / licking / freezing / navigation_error
    anatomical_direction — dorsal / ventral / medial / lateral / anterior / posterior / layer_specific
    effect_scale        — strong / modest / weak / absent / trend
    comparison_condition — vs_baseline / vs_control / vs_sham / within_subject / pre_post
    population_type     — pyramidal / parvalbumin / interneuron / place_cell / grid_cell / dopaminergic

  NEURAL-NETWORK / COMPUTATIONAL (4):
    synaptic_plasticity — ltp / ltd / stdp / hebbian / facilitation / depression
    network_coupling    — theta_gamma / replay / entrainment / up_down_state / nested_oscillation
    decoding_type       — svm / lda / linear_classifier / population_decoder / bci
    computational_model — integrate_and_fire / hodgkin_huxley / rnn / attractor / drift_diffusion

  EXPERIMENTAL CONTEXT / METHODOLOGY / DISEASE (11):
    sensory_stimulus         — visual / auditory / somatosensory / olfactory / multisensory
    pharmacological_agent    — muscimol / TTX / AP5 / ketamine / 6-OHDA / other named agents
    injury_model             — TBI / stroke / epilepsy / Parkinson / Alzheimer / depression models
    developmental_stage      — embryonic / neonatal / juvenile / adolescent / young_adult / aged
    genetic_tool             — AAV / Cre-lox / DREADD / opsins / knockout / RNAi / CRISPR
    molecular_marker         — c-Fos / Arc / BDNF / CaMKII / IHC / RNA-seq / cytokines
    social_affective         — stress paradigms / anxiety tests / depression tests / social behavior
    sleep_stage              — NREM / REM / consolidation / quiet wake / active wake
    metabolic_context        — neuroinflammation / oxidative stress / BBB / apoptosis / metabolism
    connectivity_type        — functional / effective / structural / top-down / feedforward
    dimensionality_reduction — PCA / ICA / t-SNE / UMAP / dPCA / manifold / state-space

All extraction is purely rule-based (regex/substring matching) — no model calls.
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


def _first(pattern_value_pairs: list[tuple[str, str]], text: str) -> list[str]:
    """Return all values whose pattern matches (multiple may match). Preserves order, dedupes."""
    seen: set[str] = set()
    result = []
    for p, v in pattern_value_pairs:
        if re.search(p, text, re.IGNORECASE) and v not in seen:
            seen.add(v)
            result.append(v)
    return result


# ---------------------------------------------------------------------------
# Negation
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS: list[str] = [
    r"\bdid\s+not\b",
    r"\bdoes\s+not\b",
    r"\bdo\s+not\b",
    r"\bwas\s+not\b",
    r"\bwere\s+not\b",
    r"\bno\s+significant\b",
    r"\bno\s+effect\b",
    r"\bnot\s+significantly?\b",
    r"\bfailed\s+to\b",
    r"\babsence\s+of\b",
    r"\bnot\s+observed\b",
    r"\bnot\s+activated?\b",
    r"\bcounteractivat",
    r"\binhibit",
    r"\bsuppress",
    r"\bblocked?\b",
    r"\bablat",
    r"\bsilenced?\b",
    r"\bnullif",
    r"\bno\s+increase\b",
    r"\bno\s+decrease\b",
    r"\bunchanged\b",
]


def detect_negation(text: str) -> bool:
    """Return True if the finding explicitly negates or contradicts a result."""
    return _any(_NEGATION_PATTERNS, text)


# ---------------------------------------------------------------------------
# Frequency band
# ---------------------------------------------------------------------------

_FREQ_BAND_PAIRS: list[tuple[str, str]] = [
    (r"\btheta\b", "theta"),
    (r"\bgamma\b", "gamma"),
    (r"\bbeta\b", "beta"),
    (r"\balpha\b", "alpha"),
    (r"\bdelta\b", "delta"),
    (r"\bripple\b", "ripple"),
    (r"\bsharp.wave", "ripple"),
    (r"\bswr\b", "ripple"),
    (r"\bsigma\b", "sigma"),
    (r"\bspindle\b", "sigma"),
    (r"\bhigh.freq", "high_frequency"),
    (r"\bbroadband\b", "broadband"),
    (r"\blow.freq", "low_frequency"),
    (r"\binfraslow\b", "infraslow"),
    (r"\bcross.freq", "cross_frequency"),
    (r"\bphase.amplitude", "cross_frequency"),
]


def extract_frequency_bands(text: str) -> list[str]:
    return _first(_FREQ_BAND_PAIRS, text)


# ---------------------------------------------------------------------------
# Temporal pattern
# ---------------------------------------------------------------------------

_TEMPORAL_PAIRS: list[tuple[str, str]] = [
    (r"\btransient\b", "transient"),
    (r"\bbrief\b", "transient"),
    (r"\bphasic\b", "transient"),
    (r"\bsustained\b", "sustained"),
    (r"\bpersistent\b", "sustained"),
    (r"\btonic\b", "sustained"),
    (r"\bevent.locked\b", "event_locked"),
    (r"\btrial.locked\b", "event_locked"),
    (r"\bstimulus.locked\b", "event_locked"),
    (r"\bperistimulus\b", "event_locked"),
    (r"\bphase.lock", "phase_locked"),
    (r"\bphase.coupling\b", "phase_locked"),
    (r"\bphase.coher", "phase_locked"),
    (r"\boscillat", "oscillatory"),
    (r"\brhythm", "oscillatory"),
    (r"\bcyclic\b", "oscillatory"),
    (r"\bramp", "ramping"),
    (r"\bdelay.period\b", "delay_period"),
    (r"\bdelay\s+activity\b", "delay_period"),
    (r"\bworking.memory\s+period\b", "delay_period"),
]


def extract_temporal_patterns(text: str) -> list[str]:
    return _first(_TEMPORAL_PAIRS, text)


# ---------------------------------------------------------------------------
# Spatial frame
# ---------------------------------------------------------------------------

_SPATIAL_PAIRS: list[tuple[str, str]] = [
    (r"\bwhole.brain\b", "whole_brain"),
    (r"\bglobal\b", "whole_brain"),
    (r"\bdefault.mode\b", "whole_brain"),
    (r"\binter.region", "inter_regional"),
    (r"\bbetween.region", "inter_regional"),
    (r"\bsimultan", "inter_regional"),
    (r"\bcoher", "inter_regional"),
    (r"\blong.range", "inter_regional"),
    (r"\blaminar\b", "laminar"),
    (r"\blayer\s+[IVX1-6]", "laminar"),
    (r"\bcortical\s+layer", "laminar"),
    (r"\bcolumnar\b", "columnar"),
    (r"\bhypercolumn\b", "columnar"),
]


def extract_spatial_frame(text: str) -> list[str]:
    found = _first(_SPATIAL_PAIRS, text)
    return found if found else ["local"]


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------

_CONDITION_PAIRS: list[tuple[str, str]] = [
    (r"\bresting.state\b", "resting_state"),
    (r"\bresting\s+condition", "resting_state"),
    (r"\bbaseline\b", "resting_state"),
    (r"\bspontaneous\s+activity\b", "resting_state"),
    (r"\bstimulat", "stimulation"),
    (r"\boptogenet", "stimulation"),
    (r"\bchemogenet", "stimulation"),
    (r"\bdbs\b", "stimulation"),
    (r"\belectric\s+stimulat", "stimulation"),
    (r"\blesion\b", "lesion"),
    (r"\bablat", "lesion"),
    (r"\binactivat", "lesion"),
    (r"\bknockout\b", "lesion"),
    (r"\bpharmacol", "pharmacological"),
    (r"\bdrug\b", "pharmacological"),
    (r"\binfus", "pharmacological"),
    (r"\binjection\b", "pharmacological"),
    (r"\bantagonist\b", "pharmacological"),
    (r"\bagonist\b", "pharmacological"),
    (r"\btask\b", "task"),
    (r"\btrial\b", "task"),
    (r"\bchoice\b", "task"),
    (r"\bdecision\b", "task"),
    (r"\blearning\b", "task"),
]


def extract_conditions(text: str) -> list[str]:
    return _first(_CONDITION_PAIRS, text)


# ---------------------------------------------------------------------------
# Signal type
# ---------------------------------------------------------------------------

_SIGNAL_PAIRS: list[tuple[str, str]] = [
    (r"\blfp\b", "lfp"),
    (r"\blocal\s+field\s+potential", "lfp"),
    (r"\bfield\s+potential", "lfp"),
    (r"\bspike\b", "spike"),
    (r"\bunit\s+activity\b", "spike"),
    (r"\bmultiunit\b", "spike"),
    (r"\bsingle.unit\b", "spike"),
    (r"\baction\s+potential", "spike"),
    (r"\beeg\b", "eeg"),
    (r"\belectroencephalog", "eeg"),
    (r"\berp\b", "eeg"),
    (r"\bmeeg\b", "meg"),
    (r"\bmagnetoencephalog", "meg"),
    (r"\bbold\b", "bold"),
    (r"\bfmri\b", "bold"),
    (r"\bfunctional\s+mri\b", "bold"),
    (r"\bcalcium\s+imag", "calcium"),
    (r"\bca2?\+\s+imag", "calcium"),
    (r"\bgcamp", "calcium"),
    (r"\becog\b", "ecog"),
    (r"\belectrocorticog", "ecog"),
    (r"\bneuropixels\b", "neuropixels"),
    (r"\bsilicon\s+probe", "neuropixels"),
    (r"\btetrode\b", "tetrode"),
    (r"\bpatch.clamp\b", "patch_clamp"),
    (r"\bwhole.cell\b", "patch_clamp"),
    (r"\bvoltage.imag", "voltage_imaging"),
    (r"\bvsdi\b", "voltage_imaging"),
    (r"\bfluoresc", "fluorescence"),
    (r"\bmicropipette\b", "micropipette"),
]


def extract_signal_types(text: str) -> list[str]:
    return _first(_SIGNAL_PAIRS, text)


# ---------------------------------------------------------------------------
# Statistical relation
# ---------------------------------------------------------------------------

_STAT_PAIRS: list[tuple[str, str]] = [
    (r"\bcorrelat", "correlation"),
    (r"\bcovarianc", "correlation"),
    (r"\bassociat", "correlation"),
    (r"\bregress", "regression"),
    (r"\bglm\b", "regression"),
    (r"\banova\b", "anova"),
    (r"\banalysis\s+of\s+variance", "anova"),
    (r"\bt.test\b", "t_test"),
    (r"\bwilcoxon\b", "t_test"),
    (r"\bdecod", "decoding"),
    (r"\bclassif", "decoding"),
    (r"\bsvm\b", "decoding"),
    (r"\blda\b", "decoding"),
    (r"\bpca\b", "dimensionality_reduction"),
    (r"\bnmf\b", "dimensionality_reduction"),
    (r"\bdimensionality\s+reduc", "dimensionality_reduction"),
    (r"\bgranger\b", "granger_causality"),
    (r"\bcausali", "granger_causality"),
    (r"\bmutual\s+information\b", "information_theory"),
]


def extract_statistical_relations(text: str) -> list[str]:
    return _first(_STAT_PAIRS, text)


# ===========================================================================
# NEW FIELDS — BEHAVIORAL / ANATOMICAL (5)
# ===========================================================================

# ---------------------------------------------------------------------------
# Behavioral measure — what behavioral variable was measured
# ---------------------------------------------------------------------------

_BEHAVIORAL_PAIRS: list[tuple[str, str]] = [
    (r"\baccurac", "accuracy"),
    (r"\bcorrect\s+response", "accuracy"),
    (r"\bhit\s+rate", "accuracy"),
    (r"\bd'", "discriminability"),
    (r"\bd\s+prime\b", "discriminability"),
    (r"\bdiscriminab", "discriminability"),
    (r"\breaction\s+time\b", "reaction_time"),
    (r"\bresponse\s+time\b", "reaction_time"),
    (r"\b\bRT\b", "reaction_time"),
    (r"\blatency\b", "reaction_time"),
    (r"\bchoice\b", "choice"),
    (r"\bdecision\b", "choice"),
    (r"\bpreference\b", "choice"),
    (r"\blick", "licking"),
    (r"\bnose\s+pok", "licking"),
    (r"\bpoke\b", "licking"),
    (r"\bfreez", "freezing"),
    (r"\bimmobil", "freezing"),
    (r"\bfear\s+response", "freezing"),
    (r"\brunning\s+speed", "locomotion"),
    (r"\blocomotion\b", "locomotion"),
    (r"\bvelocity\b", "locomotion"),
    (r"\bmovement\b", "locomotion"),
    (r"\bnavigation\b", "navigation"),
    (r"\bpath\s+length", "navigation"),
    (r"\bplace\s+field", "navigation"),
    (r"\bspatial\s+error", "navigation"),
    (r"\blever\s+press", "lever_press"),
    (r"\boperant\b", "lever_press"),
    (r"\bpressing\b", "lever_press"),
    (r"\bsucrose\b", "reward_consumption"),
    (r"\bwater\s+reward", "reward_consumption"),
    (r"\breward\s+consumption", "reward_consumption"),
    (r"\bperformance\b", "performance"),
    (r"\berror\s+rate", "error_rate"),
    (r"\bmistake", "error_rate"),
]


def extract_behavioral_measures(text: str) -> list[str]:
    return _first(_BEHAVIORAL_PAIRS, text)


# ---------------------------------------------------------------------------
# Anatomical direction — spatial specificity within a region
# ---------------------------------------------------------------------------

_ANATOMICAL_DIR_PAIRS: list[tuple[str, str]] = [
    (r"\bdorsal\b", "dorsal"),
    (r"\bventral\b", "ventral"),
    (r"\bmedial\b", "medial"),
    (r"\blateral\b", "lateral"),
    (r"\banterior\b", "anterior"),
    (r"\bposterior\b", "posterior"),
    (r"\brostral\b", "anterior"),
    (r"\bcaudal\b", "posterior"),
    (r"\bsuperficial\b", "superficial"),
    (r"\bdeep\s+layer", "deep"),
    (r"\bL2/3\b", "layer_2_3"),
    (r"\blayer\s+2/?3\b", "layer_2_3"),
    (r"\bL4\b", "layer_4"),
    (r"\blayer\s+4\b", "layer_4"),
    (r"\bL5\b", "layer_5"),
    (r"\blayer\s+5\b", "layer_5"),
    (r"\bL6\b", "layer_6"),
    (r"\blayer\s+6\b", "layer_6"),
    (r"\bCA1\b", "ca1"),
    (r"\bCA3\b", "ca3"),
    (r"\bdentate\b", "dentate_gyrus"),
    (r"\bsubiculum\b", "subiculum"),
    (r"\bdorsomedial\b", "dorsomedial"),
    (r"\bdorsolateral\b", "dorsolateral"),
    (r"\bventromedial\b", "ventromedial"),
    (r"\bventrolateral\b", "ventrolateral"),
]


def extract_anatomical_directions(text: str) -> list[str]:
    return _first(_ANATOMICAL_DIR_PAIRS, text)


# ---------------------------------------------------------------------------
# Effect scale — qualitative magnitude of the finding
# ---------------------------------------------------------------------------

_EFFECT_SCALE_PAIRS: list[tuple[str, str]] = [
    (r"\brobust\b", "strong"),
    (r"\bprofound\b", "strong"),
    (r"\bdramatic\b", "strong"),
    (r"\bsubstantial\b", "strong"),
    (r"\blarge\s+effect", "strong"),
    (r"\bstrongly\b", "strong"),
    (r"\bmarkedly\b", "strong"),
    (r"\bstrikingly\b", "strong"),
    (r"\bsignificantly\b", "strong"),
    (r"\bmodest\b", "modest"),
    (r"\bmoderate\b", "modest"),
    (r"\bpartial\b", "modest"),
    (r"\bsome\s+effect\b", "modest"),
    (r"\bsmall\s+effect", "modest"),
    (r"\bweak\b", "weak"),
    (r"\bsmall\s+but\b", "weak"),
    (r"\bsubtle\b", "weak"),
    (r"\bmarginal\b", "weak"),
    (r"\btrend\b", "trend"),
    (r"\btendency\b", "trend"),
    (r"\bapproach(?:es|ing)?\s+significance\b", "trend"),
    (r"\bnot\s+significant\b", "absent"),
    (r"\bno\s+significant\b", "absent"),
    (r"\bno\s+effect\b", "absent"),
    (r"\bnull\s+result\b", "absent"),
]


def extract_effect_scale(text: str) -> list[str]:
    return _first(_EFFECT_SCALE_PAIRS, text)


# ---------------------------------------------------------------------------
# Comparison condition — what the finding is relative to
# ---------------------------------------------------------------------------

_COMPARISON_PAIRS: list[tuple[str, str]] = [
    (r"\bvs\.?\s+baseline\b", "vs_baseline"),
    (r"\brelative\s+to\s+baseline\b", "vs_baseline"),
    (r"\bcompared\s+to\s+baseline\b", "vs_baseline"),
    (r"\bvs\.?\s+rest\b", "vs_baseline"),
    (r"\bvs\.?\s+control\b", "vs_control"),
    (r"\brelative\s+to\s+control\b", "vs_control"),
    (r"\bcompared\s+to\s+control\b", "vs_control"),
    (r"\bvs\.?\s+sham\b", "vs_sham"),
    (r"\bsham.operated\b", "vs_sham"),
    (r"\bsham\s+group\b", "vs_shamy"),
    (r"\bwithin.subject\b", "within_subject"),
    (r"\bwithin\s+subject", "within_subject"),
    (r"\bpaired\b", "within_subject"),
    (r"\bbefore\s+and\s+after\b", "pre_post"),
    (r"\bpre.?\s+and\s+post\b", "pre_post"),
    (r"\bpre.post\b", "pre_post"),
    (r"\btraining\s+vs\.?\s+test\b", "pre_post"),
    (r"\bbetween.group\b", "between_group"),
    (r"\bbetween\s+group", "between_group"),
    (r"\bgroup\s+comparison\b", "between_group"),
    (r"\bvs\.?\s+vehicle\b", "vs_vehicle"),
    (r"\bvehicle.inject", "vs_vehicle"),
    (r"\bvs\.?\s+saline\b", "vs_vehicle"),
    (r"\bcontra(?:lateral)?\.?\s+hemisphere", "contralateral"),
    (r"\bipsi(?:lateral)?\.?\s+vs\.?\s+contra", "contralateral"),
]


def extract_comparison_conditions(text: str) -> list[str]:
    return _first(_COMPARISON_PAIRS, text)


# ---------------------------------------------------------------------------
# Population type — cell type or functional class of neurons
# ---------------------------------------------------------------------------

_POPULATION_PAIRS: list[tuple[str, str]] = [
    (r"\bpyramidal\b", "pyramidal"),
    (r"\bprincipal\s+cell", "pyramidal"),
    (r"\bprincipal\s+neuron", "pyramidal"),
    (r"\bexcitatory\s+neuron", "pyramidal"),
    (r"\bparvalbumin\b", "parvalbumin"),
    (r"\bPV\+", "parvalbumin"),
    (r"\bPV\s+interneuron", "parvalbumin"),
    (r"\bfast.spiking\b", "parvalbumin"),
    (r"\bsomatostatin\b", "somatostatin"),
    (r"\bSST\+", "somatostatin"),
    (r"\bSST\s+interneuron", "somatostatin"),
    (r"\bVIP\+?\b", "vip_interneuron"),
    (r"\bVIP\s+interneuron", "vip_interneuron"),
    (r"\binterneuron\b", "interneuron"),
    (r"\binhibitory\s+neuron", "interneuron"),
    (r"\bGABAergic\b", "interneuron"),
    (r"\bplace\s+cell", "place_cell"),
    (r"\bplace.modulated\b", "place_cell"),
    (r"\bgrid\s+cell", "grid_cell"),
    (r"\bhead.direction\s+cell", "head_direction_cell"),
    (r"\bborder\s+cell", "border_cell"),
    (r"\bspeed\s+cell", "speed_cell"),
    (r"\bdopaminergic\b", "dopaminergic"),
    (r"\bdopamine\s+neuron", "dopaminergic"),
    (r"\bDA\s+neuron", "dopaminergic"),
    (r"\bserotonergic\b", "serotonergic"),
    (r"\b5.HT\s+neuron", "serotonergic"),
    (r"\bcholinergic\b", "cholinergic"),
    (r"\bnoradrenergic\b", "noradrenergic"),
    (r"\bpurkinje\b", "purkinje_cell"),
    (r"\bgranule\s+cell", "granule_cell"),
    (r"\bastrocyte\b", "astrocyte"),
    (r"\bmicroglia\b", "microglia"),
    (r"\boligodendrocyte\b", "oligodendrocyte"),
]


def extract_population_types(text: str) -> list[str]:
    return _first(_POPULATION_PAIRS, text)


# ===========================================================================
# NEW FIELDS — NEURAL NETWORK / COMPUTATIONAL (4)
# ===========================================================================

# ---------------------------------------------------------------------------
# Synaptic plasticity — plasticity type
# ---------------------------------------------------------------------------

_PLASTICITY_PAIRS: list[tuple[str, str]] = [
    (r"\bLTP\b", "ltp"),
    (r"\blong.term\s+potentiation\b", "ltp"),
    (r"\bsynaptic\s+potentiation\b", "ltp"),
    (r"\bpotentiated\b", "ltp"),
    (r"\bLTD\b", "ltd"),
    (r"\blong.term\s+depression\b", "ltd"),
    (r"\bsynaptic\s+depression\b", "ltd"),
    (r"\bdepressed\b", "ltd"),
    (r"\bSTDP\b", "stdp"),
    (r"\bspike.timing.dependent\s+plasticity\b", "stdp"),
    (r"\bHebbian\b", "hebbian"),
    (r"\bassociative\s+plasticity\b", "hebbian"),
    (r"\bsynaptic\s+facilitation\b", "facilitation"),
    (r"\bfacilitated\b", "facilitation"),
    (r"\bshort.term\s+facilitation\b", "facilitation"),
    (r"\bSTP\b", "short_term_plasticity"),
    (r"\bshort.term\s+plasticity\b", "short_term_plasticity"),
    (r"\bshort.term\s+depression\b", "short_term_plasticity"),
    (r"\bmetaplasticity\b", "metaplasticity"),
    (r"\bhomeostatic\s+plasticity\b", "homeostatic"),
    (r"\bsynaptic\s+scaling\b", "homeostatic"),
    (r"\bspine\s+densit", "structural_plasticity"),
    (r"\bdendritic\s+spine", "structural_plasticity"),
    (r"\baxonal\s+sprout", "structural_plasticity"),
    (r"\bneurogenesis\b", "neurogenesis"),
]


def extract_synaptic_plasticity(text: str) -> list[str]:
    return _first(_PLASTICITY_PAIRS, text)


# ---------------------------------------------------------------------------
# Network coupling — oscillatory coupling and replay phenomena
# ---------------------------------------------------------------------------

_COUPLING_PAIRS: list[tuple[str, str]] = [
    (r"\btheta.gamma\b", "theta_gamma_coupling"),
    (r"\bgamma\s+nested\s+in\s+theta\b", "theta_gamma_coupling"),
    (r"\bphase.amplitude\s+coupling\b", "theta_gamma_coupling"),
    (r"\bcoupling\b", "oscillatory_coupling"),
    (r"\bentr[ai]nment\b", "entrainment"),
    (r"\bentrained\b", "entrainment"),
    (r"\breplay\b", "replay"),
    (r"\breactivat", "replay"),
    (r"\bsequential\s+reactiv", "replay"),
    (r"\bsequence\s+replay", "replay"),
    (r"\bup\s+states?\b", "up_down_state"),
    (r"\bdown\s+states?\b", "up_down_state"),
    (r"\bup.down\s+states?", "up_down_state"),
    (r"\bup\s+and\s+down\s+states?", "up_down_state"),
    (r"\bslow\s+oscillation", "up_down_state"),
    (r"\bnested\s+oscillat", "nested_oscillation"),
    (r"\bcross.frequency\s+coupling\b", "cross_frequency_coupling"),
    (r"\bspike.field\s+coheren", "spike_field_coherence"),
    (r"\bLFP.spike\s+coheren", "spike_field_coherence"),
    (r"\bspike.phase\b", "spike_field_coherence"),
    (r"\bsynchron", "synchrony"),
    (r"\bdesynchron", "desynchrony"),
    (r"\bcoordinated\s+activit", "synchrony"),
]


def extract_network_coupling(text: str) -> list[str]:
    return _first(_COUPLING_PAIRS, text)


# ---------------------------------------------------------------------------
# Decoding type — classifier / readout method
# ---------------------------------------------------------------------------

_DECODING_PAIRS: list[tuple[str, str]] = [
    (r"\bSVM\b", "svm"),
    (r"\bsupport\s+vector\b", "svm"),
    (r"\bLDA\b", "lda"),
    (r"\blinear\s+discriminant\b", "lda"),
    (r"\blinear\s+classifier\b", "linear_classifier"),
    (r"\blogistic\s+regress", "linear_classifier"),
    (r"\bneural\s+network\s+classifier\b", "neural_network_classifier"),
    (r"\bdeep\s+learning\b", "neural_network_classifier"),
    (r"\bCNN\b", "neural_network_classifier"),
    (r"\bBCI\b", "bci"),
    (r"\bbrain.computer\s+interface\b", "bci"),
    (r"\bneuroprostheti", "bci"),
    (r"\bpopulation\s+decod", "population_decoder"),
    (r"\bensemble\s+decod", "population_decoder"),
    (r"\btemplate\s+match", "template_matching"),
    (r"\bbayesian\s+decod", "bayesian_decoder"),
    (r"\bmaximum\s+likelihood", "bayesian_decoder"),
    (r"\bcross.validat", "cross_validated"),
    (r"\bleave.one.out", "cross_validated"),
    (r"\bk.fold\b", "cross_validated"),
    (r"\bbit\s+rate\b", "information_transfer_rate"),
    (r"\bbits\s+per\s+second\b", "information_transfer_rate"),
]


def extract_decoding_types(text: str) -> list[str]:
    return _first(_DECODING_PAIRS, text)


# ---------------------------------------------------------------------------
# Computational model — theoretical/modeling framework
# ---------------------------------------------------------------------------

_MODEL_PAIRS: list[tuple[str, str]] = [
    (r"\bintegrate.and.fire\b", "integrate_and_fire"),
    (r"\bIAF\s+model\b", "integrate_and_fire"),
    (r"\bHodgkin.Huxley\b", "hodgkin_huxley"),
    (r"\bconductance.based\b", "hodgkin_huxley"),
    (r"\brecurrent\s+neural\s+network\b", "rnn"),
    (r"\bRNN\b", "rnn"),
    (r"\bLSTM\b", "rnn"),
    (r"\breservoir\s+computing\b", "reservoir"),
    (r"\becho\s+state\b", "reservoir"),
    (r"\battractor\s+network\b", "attractor"),
    (r"\battractor\s+dynamics\b", "attractor"),
    (r"\bHopfield\b", "attractor"),
    (r"\bdrift.diffusion\b", "drift_diffusion"),
    (r"\bDDM\b", "drift_diffusion"),
    (r"\baccumulator\s+model\b", "drift_diffusion"),
    (r"\bTrace\s+model\b", "drift_diffusion"),
    (r"\bBayesian\s+model\b", "bayesian"),
    (r"\bBayesian\s+brain\b", "bayesian"),
    (r"\bpredictive\s+cod", "predictive_coding"),
    (r"\bprecision.weighted\b", "predictive_coding"),
    (r"\bfree\s+energy\s+principle\b", "predictive_coding"),
    (r"\breinforcement\s+learning\s+model\b", "rl_model"),
    (r"\btemporal\s+difference\b", "rl_model"),
    (r"\bTD\s+(?:error|model|learning)\b", "rl_model"),
    (r"\bQ.learning\b", "rl_model"),
    (r"\bmean[\s-]+field\b", "mean_field"),
    (r"\brate\s+model\b", "mean_field"),
    (r"\bneural\s+mass\b", "mean_field"),
    (r"\bWilson.Cowan\b", "mean_field"),
    (r"\bcomputational\s+model\b", "generic_computational"),
    (r"\bspiking\s+network\s+model\b", "spiking_network"),
]


def extract_computational_models(text: str) -> list[str]:
    return _first(_MODEL_PAIRS, text)


# ===========================================================================
# NEW FIELDS — EXPERIMENTAL CONTEXT / METHODOLOGY / DISEASE (11)
# ===========================================================================

# ---------------------------------------------------------------------------
# Sensory stimulus — what type of sensory input was delivered
# ---------------------------------------------------------------------------

_SENSORY_PAIRS: list[tuple[str, str]] = [
    # visual — stimulus descriptions and tuning properties
    (r"\bvisual\s+(?:stimulus|stimuli|input|evoked|response|cortex\s+response)\b", "visual"),
    (r"\bgratings?\b", "visual"),
    (r"\bdrift\w*\s+(?:bar|grat\w+)\b", "visual"),
    (r"\bnatural\s+(?:image|scene)\b", "visual"),
    (r"\borientation\s+(?:tun\w+|select\w+|column)\b", "visual"),
    (r"\bdirection\s+select\w+\b", "visual"),
    (r"\breceptive\s+field\b", "visual"),
    (r"\bcontrast\s+sensiti\w+\b", "visual"),
    (r"\bvisual\s+evoked\b", "visual"),
    (r"\bvisual\s+response\b", "visual"),
    (r"\bvisual\s+tuning\b", "visual"),
    (r"\bphotic\b", "visual"),
    (r"\blight\s+flash\b", "visual"),
    (r"\bspatial\s+frequency\b", "visual"),
    (r"\btemporal\s+frequency\b", "visual"),
    (r"\bV1\s+(?:neuron|response|cell)\b", "visual"),
    (r"\bprimary\s+visual\s+cortex\b", "visual"),
    # auditory
    (r"\bauditory\s+(?:stimulus|stimuli|input|evoked|response|cortex\s+response)\b", "auditory"),
    (r"\bpure\s+tone\b", "auditory"),
    (r"\bnoise\s+burst\b", "auditory"),
    (r"\bclick\s+(?:train|stimulus)\b", "auditory"),
    (r"\bauditory\s+evoked\b", "auditory"),
    (r"\bsound\s+(?:stimulus|stimuli|response)\b", "auditory"),
    (r"\btone\s+(?:pip|response|onset|offset)\b", "auditory"),
    (r"\bfrequency.tuned\b", "auditory"),
    (r"\btonotopic\b", "auditory"),
    (r"\bcharacteristic\s+frequency\b", "auditory"),
    (r"\bauditory\s+cortex\b", "auditory"),
    (r"\bA1\s+(?:neuron|response|cell)\b", "auditory"),
    # somatosensory / tactile
    (r"\bwhisker\s+(?:deflect\w+|stimulat\w+|response)\b", "somatosensory"),
    (r"\bbarrel\s+cortex\b", "somatosensory"),
    (r"\btactile\s+(?:stimulus|stimuli|input|response)\b", "somatosensory"),
    (r"\bsomatosensory\s+(?:stimulus|response|cortex)\b", "somatosensory"),
    (r"\bair\s+puff\b", "somatosensory"),
    (r"\bvibrotactile\b", "somatosensory"),
    (r"\bS1\s+(?:neuron|response|cell)\b", "somatosensory"),
    # olfactory
    (r"\bodor\s+(?:stimulus|stimuli|response|presentat\w+|discriminat\w+)\b", "olfactory"),
    (r"\bolfactory\s+(?:stimulus|response|bulb\s+response)\b", "olfactory"),
    (r"\bvolatile\s+compound\b", "olfactory"),
    (r"\bodorant\b", "olfactory"),
    (r"\bolfactory\s+cortex\b", "olfactory"),
    # multisensory
    (r"\bmultisensory\b", "multisensory"),
    (r"\bcrossmodal\b", "multisensory"),
    (r"\baudiovisual\b", "multisensory"),
    (r"\bbimodal\s+(?:stimulus|input)\b", "multisensory"),
    (r"\bsuperior\s+colliculus\s+integrat\w+\b", "multisensory"),
    # nociceptive / pain
    (r"\bnociceptive\s+(?:stimulus|response|signal)\b", "nociceptive"),
    (r"\bpain\s+(?:stimulus|response|threshold)\b", "nociceptive"),
    (r"\bhot\s+plate\s+test\b", "nociceptive"),
    (r"\bformalin\s+test\b", "nociceptive"),
    (r"\bvon\s+Frey\b", "nociceptive"),
    (r"\bhyperalgesia\b", "nociceptive"),
    (r"\ballodynia\b", "nociceptive"),
    # vestibular
    (r"\bvestibular\s+(?:stimulus|response|input)\b", "vestibular"),
    (r"\brotation\s+stimul\w+\b", "vestibular"),
    (r"\bself.motion\s+signal\b", "vestibular"),
]


def extract_sensory_stimulus(text: str) -> list[str]:
    return _first(_SENSORY_PAIRS, text)


# ---------------------------------------------------------------------------
# Pharmacological agent — specific named drug / compound
# ---------------------------------------------------------------------------

_PHARM_PAIRS: list[tuple[str, str]] = [
    # Silencing / GABA
    (r"\bmuscimol\b", "muscimol"),
    (r"\bbicuculline\b", "bicuculline"),
    (r"\bgabazine\b", "bicuculline"),
    (r"\bSR95531\b", "bicuculline"),
    (r"\bdiazepam\b", "benzodiazepine"),
    (r"\bbenzodiazepine\b", "benzodiazepine"),
    (r"\bmidazolam\b", "benzodiazepine"),
    (r"\balprazolam\b", "benzodiazepine"),
    # Glutamate / NMDA antagonism
    (r"\bAP5\b|\bAP-5\b|\bAPV\b", "ap5_nmda_antagonist"),
    (r"\bNBQX\b", "nbqx_ampa_antagonist"),
    (r"\bMK.801\b|\bMK801\b", "mk801_nmda_antagonist"),
    (r"\bketamine\b", "ketamine"),
    (r"\bmemantine\b", "memantine"),
    # Sodium channel
    (r"\btetrodotoxin\b|\bTTX\b", "tetrodotoxin"),
    (r"\blidocaine\b", "lidocaine"),
    # Dopamine system
    (r"\braclopride\b", "raclopride"),
    (r"\bSCH23390\b|\bSCH-23390\b", "sch23390_d1_antagonist"),
    (r"\beticlopride\b", "eticlopride_d2_antagonist"),
    (r"\bhaloperidol\b", "haloperidol"),
    (r"\bclozapine\b", "clozapine"),
    (r"\bamphetamine\b|\bAMPH\b", "amphetamine"),
    (r"\bcocaine\b", "cocaine"),
    (r"\bL.DOPA\b|\blevodopa\b", "levodopa"),
    (r"\bapomorphine\b", "apomorphine"),
    # Cholinergic
    (r"\bscopolamine\b", "scopolamine"),
    (r"\batropine\b", "atropine"),
    (r"\bnicotine\b", "nicotine"),
    (r"\bphysostigmine\b", "physostigmine"),
    (r"\bcarbachol\b", "carbachol"),
    (r"\bdonepezil\b", "donepezil"),
    # Serotonin system
    (r"\bfluoxetine\b", "fluoxetine"),
    (r"\bsertraline\b", "fluoxetine"),
    (r"\bparoxetine\b", "fluoxetine"),
    (r"\b5.7.DHT\b|\bSSRI\b", "ssri_serotonin"),
    # Noradrenergic
    (r"\bpropranolol\b", "propranolol"),
    (r"\bclonidine\b", "clonidine"),
    (r"\bprazosin\b", "prazosin"),
    (r"\bDSP.4\b", "dsp4_noradrenaline_lesion"),
    # Opioid
    (r"\bmorphine\b", "morphine"),
    (r"\bfentanyl\b", "fentanyl"),
    (r"\bnaloxone\b|\bnaltrexone\b", "opioid_antagonist"),
    (r"\bheroin\b", "heroin"),
    # Stress hormones / HPA
    (r"\bcorticosterone\b", "corticosterone"),
    (r"\bdexamethasone\b", "dexamethasone"),
    (r"\bCRH\b|\bcorticotropin.releasing\b", "crh"),
    (r"\bmetyrapone\b", "metyrapone"),
    # Cannabinoid
    (r"\bTHC\b|\btetrahydrocannabinol\b", "thc"),
    (r"\bCBD\b|\bcannabidiol\b", "cbd"),
    (r"\bCB1\s+antagonist\b|\bSR141716\b|\bAM251\b", "cb1_antagonist"),
    # Antibiotics / tools
    (r"\bcycloheximide\b", "cycloheximide"),
    (r"\banisomycin\b", "anisomycin"),
    # Epilepsy inducers
    (r"\bkainic\s+acid\b|\bkainate\b", "kainic_acid"),
    (r"\bpilocarpine\b", "pilocarpine"),
    # Parkinson toxins
    (r"\bMPTP\b", "mptp"),
    (r"\b6.OHDA\b|\b6\s+OHDA\b", "6ohda"),
    # Anesthetics used as pharmacological tools
    (r"\burethane\b", "urethane"),
    (r"\bisoflurane\b", "isoflurane"),
    (r"\bpentobarbital\b", "pentobarbital"),
]


def extract_pharmacological_agent(text: str) -> list[str]:
    return _first(_PHARM_PAIRS, text)


# ---------------------------------------------------------------------------
# Injury / disease model — experimental model of pathology
# ---------------------------------------------------------------------------

_INJURY_PAIRS: list[tuple[str, str]] = [
    # TBI
    (r"\bcontrolled\s+cortical\s+impact\b|\bCCI\b", "tbi_cci"),
    (r"\bfluid\s+percussion\s+injur\w+\b|\bFPI\b", "tbi_fpi"),
    (r"\bblast\s+injur\w+\b|\bblast\s+TBI\b", "tbi_blast"),
    (r"\btraumatic\s+brain\s+injur\w+\b|\bTBI\b", "tbi_general"),
    (r"\bconcussion\b", "tbi_concussion"),
    # Stroke / ischemia
    (r"\bmiddle\s+cerebral\s+artery\s+occlusion\b|\bMCAO\b", "stroke_mcao"),
    (r"\bcerebral\s+ischemia\b|\bfocal\s+ischemia\b", "stroke_ischemia"),
    (r"\bischemic\s+stroke\b", "stroke_ischemia"),
    (r"\bphotothrombosis\b", "stroke_photothrombosis"),
    # Epilepsy models
    (r"\bkainic\s+acid\s+model\b|\bkainate\s+model\b", "epilepsy_kainate"),
    (r"\bpilocarpine\s+model\b", "epilepsy_pilocarpine"),
    (r"\bstatus\s+epilepticus\b|\bSE\b", "epilepsy_se"),
    (r"\btemporal\s+lobe\s+epilepsy\b|\bTLE\b", "epilepsy_tle"),
    (r"\bkindling\b", "epilepsy_kindling"),
    # Parkinson models
    (r"\b6.OHDA\s+(?:lesion|model|rat|mouse)\b", "parkinson_6ohda"),
    (r"\bMPTP\s+(?:model|mouse|treatment)\b|\bMPTP.induced\b", "parkinson_mptp"),
    (r"\balpha.synuclein\s+(?:model|transgenic)\b|\bPINK1\b|\bParkin\s+knockout\b", "parkinson_genetic"),
    # Alzheimer models
    (r"\bAPP\s+(?:mouse|transgenic|model)\b|\bAPP/PS1\b|\b5xFAD\b|\bhAPP\b", "alzheimer_app"),
    (r"\btau\s+(?:model|transgenic|patholog\w+)\b|\bP301L\b|\bPS19\b|\btauopathy\b", "alzheimer_tau"),
    (r"\bamyloid\s+(?:plaque|model|patholog\w+)\b|\bAbeta\b|\bA.beta\b", "alzheimer_amyloid"),
    # Spinal cord
    (r"\bspinal\s+cord\s+injur\w+\b|\bSCI\b", "spinal_cord_injury"),
    (r"\bcontusion\s+model\b", "spinal_cord_injury"),
    # Peripheral nerve
    (r"\bsciatic\s+nerve\b|\bCCI\s+model\b|\bSNI\b|\bSNL\b", "peripheral_nerve_injury"),
    (r"\bneuropathic\s+pain\b", "neuropathic_pain"),
    # Depression / stress models
    (r"\bchronic\s+(?:mild|unpredictable)\s+stress\b|\bCUMS\b|\bCMS\b", "depression_stress_model"),
    (r"\bsocial\s+defeat\b|\bchronic\s+social\s+defeat\b", "depression_social_defeat"),
    (r"\blearned\s+helplessness\b", "depression_learned_helplessness"),
    # Autism models
    (r"\bVPA\s+model\b|\bvalproic\s+acid\s+model\b", "autism_vpa"),
    (r"\bFMR1\s+knockout\b|\bFMRP\b|\bfragile\s+X\b", "autism_fmr1"),
    (r"\bShank3\b|\bShank\s+knockout\b", "autism_shank"),
    # Schizophrenia models
    (r"\bneonatal\s+ventral\s+hippocampal\s+lesion\b|\bNVHL\b", "schizophrenia_nvhl"),
    (r"\bdiscontinued\s+pre.pulse\b|\bPPI\s+deficit\b", "schizophrenia_ppi"),
]


def extract_injury_model(text: str) -> list[str]:
    return _first(_INJURY_PAIRS, text)


# ---------------------------------------------------------------------------
# Developmental stage — age / developmental window of subjects
# ---------------------------------------------------------------------------

_DEVELOPMENTAL_PAIRS: list[tuple[str, str]] = [
    (r"\bembryonic\b|\bE[0-9]{1,2}\b|\bfetal\b|\bembry\w+\b", "embryonic"),
    (r"\bneonatal\b|\bP[0-9]\b(?!\d)|\bP1[0-4]\b|\bpostnatal\s+day\s+[0-9]{1,2}\b|\bnewborn\b|\bpup\b", "neonatal"),
    (r"\bjuvenile\b|\bP1[5-9]\b|\bP2[0-9]\b|\bpre.adolescent\b|\bpreadolescent\b", "juvenile"),
    (r"\badolescent\b|\bP[3-5][0-9]\b|\bpuberty\b|\bpubertal\b|\bperi.adolescent\b", "adolescent"),
    (r"\byoung\s+adult\b|\bP6[0-9]\b|\bP[7-9][0-9]\b", "young_adult"),
    (r"\baged\b|\bold\s+(?:mice|rats|animals|animals)\b|\baging\b|\bsenescent\b|\bgeriatric\b|\b(?:18|20|22|24).month.old\b", "aged"),
    (r"\bpostnatal\b(?!\s+day)", "postnatal"),
]


def extract_developmental_stage(text: str) -> list[str]:
    return _first(_DEVELOPMENTAL_PAIRS, text)


# ---------------------------------------------------------------------------
# Genetic tool — molecular/genetic manipulation type
# ---------------------------------------------------------------------------

_GENETIC_TOOL_PAIRS: list[tuple[str, str]] = [
    (r"\bAAV\w*\b|\badeno.associated\s+virus\b", "aav_vector"),
    (r"\blentivirus\b|\blentiviral\s+vector\b", "lentiviral_vector"),
    (r"\bCre.dependent\b|\bCre.inducible\b|\bflox\w+\b|\blox.stop.lox\b|\bLSL\b", "cre_lox"),
    (r"\bDREADD\b|\bhM3Dq\b|\bhM4Di\b|\bCNO\b", "dreadd"),
    (r"\bchannelrhodopsin\b|\bChR2\b|\bChRmine\b|\beSFO\b|\bC1V1\b", "channelrhodopsin"),
    (r"\bArchT\b|\bNpHR\b|\bhalorhodopsin\b|\biC\+\+\b|\beArchT\b", "inhibitory_opsin"),
    (r"\bknockout\b|\bKO\s+(?:mice|mouse|rat|animal)\b|\bnull\s+mutant\b", "knockout"),
    (r"\bknock.in\b|\breporter\s+mouse\b|\bKI\s+(?:mice|mouse)\b", "knockin"),
    (r"\btransgenic\b|\boverexpression\b|\bconditional\s+overexpression\b", "transgenic_overexpression"),
    (r"\bsiRNA\b|\bshRNA\b|\bRNAi\b|\bknockdown\b", "rnai_knockdown"),
    (r"\bCRISPR\b|\bCas9\b|\bbase\s+editing\b|\bprime\s+editing\b", "crispr"),
    (r"\bGFP\b|\beYFP\b|\btdTomato\b|\bmCherry\b|\bfluorescent\s+protein\b|\bfluorescent\s+reporter\b", "fluorescent_reporter"),
    (r"\bCamKII.Cre\b|\bPV.Cre\b|\bSST.Cre\b|\bVIP.Cre\b|\bDAT.Cre\b|\bTH.Cre\b", "cre_driver_line"),
]


def extract_genetic_tool(text: str) -> list[str]:
    return _first(_GENETIC_TOOL_PAIRS, text)


# ---------------------------------------------------------------------------
# Molecular marker — protein/RNA marker or assay type used
# ---------------------------------------------------------------------------

_MOLECULAR_MARKER_PAIRS: list[tuple[str, str]] = [
    # Immediate-early genes
    (r"\bc.Fos\b|\bcFos\b|\bFos\b|\bimmediate.early\s+gene\b|\bIEG\b", "cfos_ieg"),
    (r"\bArc\b|\bArc\s+mRNA\b|\bActivity.regulated\b", "arc"),
    (r"\bEgr1\b|\bzif268\b|\bKrox.24\b", "egr1"),
    # Neurotrophins
    (r"\bBDNF\b|\bbrain.derived\s+neurotrophic\b|\bTrkB\b", "bdnf"),
    (r"\bNGF\b|\bnerve\s+growth\s+factor\b|\bTrkA\b", "ngf"),
    (r"\bNT.3\b|\bneurotrophin.3\b", "nt3"),
    # Kinases / signaling
    (r"\bCaMKII\b|\bcalcium.calmodulin.dependent\s+kinase\b", "camkii"),
    (r"\bpCREB\b|\bCREB\s+phosphorylation\b|\bphospho.CREB\b", "creb_phosphorylation"),
    (r"\bERK\b|\bpERK\b|\bMAPK\b|\bMEK\b", "erk_mapk"),
    (r"\bPKA\b|\bprotein\s+kinase\s+A\b|\bcAMP.PKA\b", "pka"),
    (r"\bPI3K\b|\bAkt\b|\bmTOR\b|\brapamycin\b", "pi3k_akt_mtor"),
    # Receptors
    (r"\bGluA1\b|\bGluN2B\b|\bNR2B\b|\bAMPA\s+receptor\s+subunit\b", "glutamate_receptor_subunit"),
    (r"\bD1R\b|\bD1\s+receptor\b|\bDrd1\b", "d1_receptor"),
    (r"\bD2R\b|\bD2\s+receptor\b|\bDrd2\b", "d2_receptor"),
    (r"\bNMDA\s+receptor\b|\bNR1\b|\bGluN1\b", "nmda_receptor"),
    # Assays
    (r"\bimmunohistochem\w+\b|\bIHC\b|\bimmunostain\w+\b|\bimmunofluo\w+\b", "immunohistochemistry"),
    (r"\bwestern\s+blot\b|\bprotein\s+expression\b|\bwestern\s+analysis\b", "western_blot"),
    (r"\bPCR\b|\bqPCR\b|\bRT.PCR\b|\bgene\s+expression\b", "pcr"),
    (r"\bRNA.seq\b|\btranscriptom\w+\b|\bgene\s+expression\s+profil\w+\b", "rna_sequencing"),
    (r"\bsingle.cell\s+RNA\b|\bscRNA.seq\b|\bsnRNA.seq\b", "single_cell_rna"),
    (r"\bin\s+situ\s+hybridization\b|\bFISH\b|\bRNAscope\b", "in_situ_hybridization"),
    (r"\bflow\s+cytometry\b|\bFACS\b", "flow_cytometry"),
    # Inflammatory markers
    (r"\bIba1\b|\bCD68\b|\bcr3\b", "microglia_marker"),
    (r"\bGFAP\b|\bS100B\b", "astrocyte_marker"),
    (r"\bTNF.alpha\b|\bIL.1beta\b|\bIL.6\b|\bcytokine\b|\bneuroinflammation\b", "inflammatory_cytokine"),
]


def extract_molecular_marker(text: str) -> list[str]:
    return _first(_MOLECULAR_MARKER_PAIRS, text)


# ---------------------------------------------------------------------------
# Social / affective context — stress, emotion, social paradigm
# ---------------------------------------------------------------------------

_SOCIAL_AFFECTIVE_PAIRS: list[tuple[str, str]] = [
    # Stress
    (r"\brestraint\s+stress\b", "restraint_stress"),
    (r"\bforced\s+swim\b|\bforcibly\s+submerged\b", "forced_swim_stress"),
    (r"\bfoot\s*shock\b|\belectric\s+shock\b|\bunavoidable\s+shock\b", "footshock_stress"),
    (r"\bchronic\s+(?:mild|unpredictable)\s+stress\b|\bCUMS\b|\bCMS\b", "chronic_stress"),
    (r"\bsocial\s+defeat\b|\bdefeat\s+stress\b", "social_defeat"),
    (r"\bpredator\s+stress\b|\bpredator\s+odor\b", "predator_stress"),
    (r"\bstress\b", "stress_general"),
    # Anxiety-related
    (r"\belevated\s+plus\s+maze\b|\bEPM\b", "elevated_plus_maze"),
    (r"\bopen\s+field\s+test\b|\bOFT\b", "open_field"),
    (r"\blight.dark\s+box\b|\blight.dark\s+transition\b", "light_dark_test"),
    (r"\bnovelty\s+suppressed\s+feeding\b|\bNSF\b", "novelty_suppressed_feeding"),
    (r"\banziolytic\b|\banxiogenic\b|\banxiety.like\b", "anxiety_behavior"),
    # Depression-related
    (r"\bforced\s+swim\s+test\b|\bFST\b|\bimmobilit\w+\s+FST\b", "forced_swim_test"),
    (r"\btail\s+suspension\s+test\b|\bTST\b", "tail_suspension_test"),
    (r"\bsucrose\s+preference\b|\banhedonia\b", "anhedonia"),
    (r"\blearned\s+helplessness\b", "learned_helplessness"),
    # Social behavior
    (r"\bsocial\s+recogniti\w+\b", "social_recognition"),
    (r"\bsocial\s+interact\w+\b", "social_interaction"),
    (r"\bsocial\s+memory\b", "social_memory"),
    (r"\bthree.chamber\s+test\b|\b3.chamber\b", "three_chamber_social"),
    (r"\baggression\b|\bterritorial\b|\bfighting\b", "aggression"),
    (r"\bmaternal\s+behavior\b|\bpup\s+retrieval\b|\bnest\s+building\b", "maternal_behavior"),
    # Affective states
    (r"\bfear\s+expression\b|\bconditioned\s+fear\b|\bfear\s+generalization\b", "fear_expression"),
    (r"\bcorticosterone\s+level\b|\bHPA\s+axis\b|\bcortisol\b", "hpa_axis_measure"),
]


def extract_social_affective(text: str) -> list[str]:
    return _first(_SOCIAL_AFFECTIVE_PAIRS, text)


# ---------------------------------------------------------------------------
# Sleep stage — sleep/wake state during recording
# ---------------------------------------------------------------------------

_SLEEP_STAGE_PAIRS: list[tuple[str, str]] = [
    (r"\bNREM\b|\bnon.REM\b|\bslow.wave\s+sleep\b|\bSWS\b|\bN[23]\s+sleep\b", "nrem"),
    (r"\bREM\s+sleep\b|\brapid\s+eye\s+movement\s+sleep\b|\bparadoxical\s+sleep\b", "rem"),
    (r"\blight\s+NREM\b|\bN1\s+sleep\b|\bN2\s+sleep\b", "light_nrem"),
    (r"\bsleep\s+onset\b|\bsleep.wake\s+transition\b|\bdrifting\s+off\b", "sleep_onset"),
    (r"\bpost.learning\s+sleep\b|\bsleep\s+consolidat\w+\b|\boffline\s+consolidat\w+\b", "sleep_consolidation"),
    (r"\bsleep.dependent\b|\bduring\s+sleep\b|\bpost.task\s+sleep\b", "sleep_dependent"),
    (r"\bquiet\s+waking\b|\bpassive\s+waking\b|\bawake\s+rest\w+\b", "quiet_wake"),
    (r"\bactive\s+waking\b|\bactive\s+explorat\w+\b|\btheta\s+state\b", "active_wake"),
    (r"\bsleep.wake\s+cycle\b|\bcircadian\b|\bultradian\b", "sleep_wake_cycle"),
    (r"\bpre.sleep\b|\bbefore\s+sleep\b|\bpre.rest\b", "pre_sleep"),
]


def extract_sleep_stage(text: str) -> list[str]:
    return _first(_SLEEP_STAGE_PAIRS, text)


# ---------------------------------------------------------------------------
# Metabolic / inflammatory context
# ---------------------------------------------------------------------------

_METABOLIC_PAIRS: list[tuple[str, str]] = [
    # Neuroinflammation
    (r"\bneuroinflammation\b|\bneuro.inflammation\b", "neuroinflammation"),
    (r"\bmicroglia\s+activat\w+\b|\bmicroglial\s+activat\w+\b|\bIba1\b", "microglial_activation"),
    (r"\bTNF.alpha\b|\bTNF\b", "tnf_alpha"),
    (r"\bIL.1.?beta\b|\binterleukin.1\b", "il1_beta"),
    (r"\bIL.6\b|\binterleukin.6\b", "il6"),
    (r"\bIL.10\b|\binterleukin.10\b", "il10"),
    (r"\bcytokine\b|\bpro.inflammatory\b", "cytokine_general"),
    (r"\bNFkB\b|\bNF.kappaB\b|\bnuclear\s+factor\b", "nfkb"),
    # Oxidative stress
    (r"\boxidative\s+stress\b|\breactive\s+oxygen\s+species\b|\bROS\b", "oxidative_stress"),
    (r"\bantioxidant\b|\bsuperoxide\b|\bSOD\b|\bcatalase\b", "antioxidant"),
    (r"\blipid\s+peroxidation\b|\bMDA\b", "lipid_peroxidation"),
    # Blood-brain barrier
    (r"\bblood.brain\s+barrier\b|\bBBB\b|\bvascular\s+permeability\b", "blood_brain_barrier"),
    (r"\bendothelial\s+cell\b|\bblood\s+vessel\b|\bangiogenesis\b", "vascular"),
    # Metabolism
    (r"\bglucose\s+(?:uptake|metabolism|utilization)\b|\b18F.FDG\b|\bPET\b", "glucose_metabolism"),
    (r"\bATP\b|\bmitochondri\w+\b|\bOXPHOS\b|\belectron\s+transport\b", "mitochondria"),
    (r"\bautophagy\b|\blysosome\b|\bproteasome\b|\bubiquitin\b", "autophagy_proteasome"),
    # Cell death
    (r"\bapoptosis\b|\bprogrammed\s+cell\s+death\b|\bcaspase\b", "apoptosis"),
    (r"\bnecrosis\b|\bnecroptosis\b|\bpyroptosis\b", "necrosis"),
]


def extract_metabolic_context(text: str) -> list[str]:
    return _first(_METABOLIC_PAIRS, text)


# ---------------------------------------------------------------------------
# Connectivity type — how brain regions are connected / interacting
# ---------------------------------------------------------------------------

_CONNECTIVITY_PAIRS: list[tuple[str, str]] = [
    (r"\bfunctional\s+connectivity\b|\bFC\b|\bresting.state\s+FC\b|\brsFC\b", "functional_connectivity"),
    (r"\beffective\s+connectivity\b|\bDCM\b|\bdynamic\s+causal\s+model\b", "effective_connectivity"),
    (r"\bstructural\s+connectivity\b|\bDTI\b|\bdiffusion\s+tensor\b|\btractography\b|\bwhite\s+matter\s+tract\b", "structural_connectivity"),
    (r"\bcorticostriatal\b|\bcortex.striatum\b|\bfrontostriatal\b", "corticostriatal"),
    (r"\bhippocampal.cortical\b|\bCA1.PFC\b|\bHC.mPFC\b|\bhippocampo.prefrontal\b", "hippocampal_cortical"),
    (r"\bthalamocortical\b|\bthalamo.cortical\b|\bTC\s+loop\b", "thalamocortical"),
    (r"\bcerebellar.cortical\b|\bcerebellocortical\b|\bcerbellar\s+output\b", "cerebellar_cortical"),
    (r"\bamygdalo.hippocampal\b|\bamygdala.hippocampus\s+connect\w+\b", "amygdala_hippocampal"),
    (r"\bbasal\s+ganglia.cortex\b|\bBG.cortex\b|\bstriato.cortical\b", "basal_ganglia_cortical"),
    (r"\bcalossal\b|\binterhemispheric\b|\bcorpus\s+callosum\b", "interhemispheric"),
    (r"\blong.range\s+connect\w+\b|\bdistal\s+projections\b|\baxonal\s+projections\b", "long_range_projection"),
    (r"\brecurrent\s+connect\w+\b|\brecurrent\s+circuit\b|\bfeedback\s+connect\w+\b", "recurrent_circuit"),
    (r"\bfeedforward\b|\bfeed.forward\b|\bbottom.up\b", "feedforward"),
    (r"\btop.down\b|\bfeedback\s+projections\b|\bdescending\s+modulation\b", "top_down"),
    (r"\bsparse\s+coding\b|\bsparse\s+representation\b|\bpopulation\s+sparseness\b", "population_coding"),
]


def extract_connectivity_type(text: str) -> list[str]:
    return _first(_CONNECTIVITY_PAIRS, text)


# ---------------------------------------------------------------------------
# Dimensionality reduction / analysis method
# ---------------------------------------------------------------------------

_DIM_REDUCTION_PAIRS: list[tuple[str, str]] = [
    (r"\bprincipal\s+component\s+analysis\b|\bPCA\b", "pca"),
    (r"\bindependent\s+component\s+analysis\b|\bICA\b", "ica"),
    (r"\bnon.negative\s+matrix\s+factorization\b|\bNMF\b", "nmf"),
    (r"\bdemixed\s+PCA\b|\bdPCA\b", "dpca"),
    (r"\bfactor\s+analysis\b|\blatent\s+factor\b", "factor_analysis"),
    (r"\bt.SNE\b|\bt-stochastic\b|\bt.distributed\b", "tsne"),
    (r"\bUMAP\b|\buniform\s+manifold\b", "umap"),
    (r"\bmanifold\s+(?:learning|embedding|analysis)\b|\blatent\s+manifold\b", "manifold_learning"),
    (r"\blinear\s+dynamical\s+system\b|\bLDS\b|\bKalman\s+filter\b", "linear_dynamical"),
    (r"\bgaussian\s+process\b|\bGPFA\b|\bGaussian\s+process\s+factor\b", "gaussian_process"),
    (r"\bstate\s+space\s+model\b|\bSSM\b|\blatent\s+state\b", "state_space"),
    (r"\bjoint\s+PSTH\b|\bJPSTH\b|\bnoise\s+correlation\b|\bpairwise\s+correlation\b", "pairwise_correlation"),
]


def extract_dimensionality_reduction(text: str) -> list[str]:
    return _first(_DIM_REDUCTION_PAIRS, text)


# ===========================================================================
# Top-level API
# ===========================================================================

def extract_typed_fields(finding_text: str) -> dict:
    """Return all 27 typed extension fields for a single finding text.

    All fields are lists (possibly empty) except `negation` (bool).
    Callers should merge this dict into the existing finding record.
    """
    t = finding_text or ""
    return {
        # original 7
        "negation": detect_negation(t),
        "frequency_band": extract_frequency_bands(t),
        "temporal_pattern": extract_temporal_patterns(t),
        "spatial_frame": extract_spatial_frame(t),
        "condition": extract_conditions(t),
        "signal_type": extract_signal_types(t),
        "statistical_relation": extract_statistical_relations(t),
        # behavioral / anatomical 5
        "behavioral_measure": extract_behavioral_measures(t),
        "anatomical_direction": extract_anatomical_directions(t),
        "effect_scale": extract_effect_scale(t),
        "comparison_condition": extract_comparison_conditions(t),
        "population_type": extract_population_types(t),
        # neural-network / computational 4
        "synaptic_plasticity": extract_synaptic_plasticity(t),
        "network_coupling": extract_network_coupling(t),
        "decoding_type": extract_decoding_types(t),
        "computational_model": extract_computational_models(t),
        # experimental context / methodology / disease 11
        "sensory_stimulus": extract_sensory_stimulus(t),
        "pharmacological_agent": extract_pharmacological_agent(t),
        "injury_model": extract_injury_model(t),
        "developmental_stage": extract_developmental_stage(t),
        "genetic_tool": extract_genetic_tool(t),
        "molecular_marker": extract_molecular_marker(t),
        "social_affective": extract_social_affective(t),
        "sleep_stage": extract_sleep_stage(t),
        "metabolic_context": extract_metabolic_context(t),
        "connectivity_type": extract_connectivity_type(t),
        "dimensionality_reduction": extract_dimensionality_reduction(t),
    }


def enrich_finding(record: dict) -> dict:
    """Return a new record dict with all typed extension fields merged in.

    Does not mutate the input.
    """
    extras = extract_typed_fields(record.get("finding_text") or "")
    return {**record, **extras}
