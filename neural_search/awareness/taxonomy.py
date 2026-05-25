"""Deterministic taxonomy for broad neuroscience data-form awareness."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


@dataclass(frozen=True)
class DataForm:
    """A broad kind of neuroscience data the search engine should recognize."""

    id: str
    label: str
    family: str
    scale: str
    aliases: tuple[str, ...] = ()
    modalities: tuple[str, ...] = ()
    standards: tuple[str, ...] = ()
    analysis_families: tuple[str, ...] = ()
    complementary_forms: tuple[str, ...] = ()
    required_signals: tuple[str, ...] = ()

    def match_terms(self) -> tuple[str, ...]:
        return (self.id, self.label, *self.aliases, *self.modalities)


@dataclass(frozen=True)
class QueryAwareness:
    """Data-form and analysis intent inferred from a query."""

    query: str
    requested_data_forms: tuple[str, ...] = ()
    analysis_families: tuple[str, ...] = ()
    scale_terms: tuple[str, ...] = ()
    species_terms: tuple[str, ...] = ()
    required_signals: tuple[str, ...] = ()
    excluded_data_forms: tuple[str, ...] = ()

    def model_dump(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "requested_data_forms": list(self.requested_data_forms),
            "analysis_families": list(self.analysis_families),
            "scale_terms": list(self.scale_terms),
            "species_terms": list(self.species_terms),
            "required_signals": list(self.required_signals),
            "excluded_data_forms": list(self.excluded_data_forms),
        }


DATA_FORMS: dict[str, DataForm] = {
    "extracellular_ephys": DataForm(
        id="extracellular_ephys",
        label="Extracellular electrophysiology",
        family="electrophysiology",
        scale="cell_population",
        aliases=("spikes", "single unit", "multi unit", "silicon probe", "tetrode"),
        modalities=("neuropixels", "extracellular_ephys", "lfp"),
        standards=("nwb", "dandi"),
        analysis_families=("spike_train_analysis", "decoding", "event_aligned_analysis"),
        complementary_forms=("behavior", "optical_imaging", "clinical_ephys"),
        required_signals=("units", "spike_times", "events"),
    ),
    "intracellular_ephys": DataForm(
        id="intracellular_ephys",
        label="Intracellular electrophysiology",
        family="electrophysiology",
        scale="cellular",
        aliases=("patch clamp", "whole cell", "current clamp", "voltage clamp"),
        modalities=("patch_clamp", "intracellular_ephys"),
        standards=("nwb",),
        analysis_families=("cellular_physiology", "excitability_analysis"),
        required_signals=("sweeps", "membrane_voltage"),
    ),
    "optical_imaging": DataForm(
        id="optical_imaging",
        label="Optical neural imaging",
        family="imaging",
        scale="cell_population",
        aliases=("calcium imaging", "two photon", "2p", "gcamp", "miniscope"),
        modalities=("calcium_imaging", "two_photon", "widefield_imaging"),
        standards=("nwb", "dandi"),
        analysis_families=("event_aligned_analysis", "population_dynamics", "decoding"),
        complementary_forms=("behavior", "extracellular_ephys"),
        required_signals=("fluorescence", "roi_masks", "events"),
    ),
    "fiber_photometry": DataForm(
        id="fiber_photometry",
        label="Fiber photometry",
        family="imaging",
        scale="population_signal",
        aliases=("photometry", "fiber optic", "dopamine photometry"),
        modalities=("fiber_photometry",),
        standards=("nwb", "dandi"),
        analysis_families=("event_aligned_analysis", "neuromodulator_analysis"),
        complementary_forms=("behavior", "extracellular_ephys"),
        required_signals=("fluorescence", "events"),
    ),
    "eeg_meg": DataForm(
        id="eeg_meg",
        label="EEG and MEG",
        family="human_noninvasive",
        scale="whole_brain",
        aliases=("eeg", "meg", "scalp eeg", "electroencephalography", "magnetoencephalography"),
        modalities=("eeg", "meg", "polysomnography"),
        standards=("bids", "openneuro", "edf"),
        analysis_families=("time_frequency", "connectivity", "clinical_prediction", "bci_decoding"),
        complementary_forms=("behavior", "clinical"),
        required_signals=("channels", "events", "sampling_rate"),
    ),
    "intracranial_human_ephys": DataForm(
        id="intracranial_human_ephys",
        label="Human intracranial electrophysiology",
        family="clinical_ephys",
        scale="mesoscale",
        aliases=("ecog", "ieeg", "seeg", "depth electrode", "subdural grid"),
        modalities=("ecog", "ieeg", "seeg"),
        standards=("bids", "nwb", "openneuro"),
        analysis_families=("speech_decoding", "memory_analysis", "clinical_prediction"),
        complementary_forms=("behavior", "clinical", "mri"),
        required_signals=("channels", "events", "electrodes"),
    ),
    "mri": DataForm(
        id="mri",
        label="MRI and fMRI",
        family="human_noninvasive",
        scale="whole_brain",
        aliases=("fmri", "bold", "functional mri", "structural mri", "diffusion mri", "dti"),
        modalities=("fmri", "mri", "diffusion_mri"),
        standards=("bids", "openneuro", "nifti"),
        analysis_families=("connectivity", "encoding_modeling", "clinical_prediction"),
        complementary_forms=("behavior", "clinical", "intracranial_human_ephys"),
        required_signals=("images", "events", "participants"),
    ),
    "behavior": DataForm(
        id="behavior",
        label="Behavior and task events",
        family="behavior",
        scale="organism",
        aliases=("behavior", "behaviour", "pose", "tracking", "kinematics", "trials", "events"),
        modalities=("behavior_video", "pose_tracking", "running_wheel", "events"),
        standards=("bids", "nwb"),
        analysis_families=("behavioral_modeling", "reinforcement_learning", "kinematics"),
        complementary_forms=("extracellular_ephys", "optical_imaging", "eeg_meg", "mri"),
        required_signals=("events", "trials"),
    ),
    "clinical": DataForm(
        id="clinical",
        label="Clinical neuroscience",
        family="clinical",
        scale="participant",
        aliases=("patient", "clinical", "diagnosis", "seizure", "sleep", "symptom"),
        modalities=("clinical", "polysomnography", "eeg", "mri"),
        standards=("bids", "edf", "dicom"),
        analysis_families=("clinical_prediction", "phenotyping", "biomarker_discovery"),
        complementary_forms=("eeg_meg", "mri", "intracranial_human_ephys"),
        required_signals=("participants", "diagnosis", "sessions"),
    ),
    "connectomics": DataForm(
        id="connectomics",
        label="Connectomics and morphology",
        family="structure",
        scale="circuit",
        aliases=("connectome", "connectomics", "em reconstruction", "morphology", "tracing"),
        modalities=("electron_microscopy", "morphology", "tracing"),
        standards=("swc", "zarr", "ngff"),
        analysis_families=("circuit_mapping", "connectivity", "morphology_analysis"),
        complementary_forms=("extracellular_ephys", "optical_imaging", "molecular"),
        required_signals=("cells", "edges", "morphology"),
    ),
    "molecular": DataForm(
        id="molecular",
        label="Molecular and single-cell neuroscience",
        family="molecular",
        scale="cellular",
        aliases=("single cell", "scrna", "transcriptomics", "spatial transcriptomics", "proteomics"),
        modalities=("single_cell_rna", "transcriptomics", "proteomics"),
        standards=("h5ad", "zarr", "loom"),
        analysis_families=("cell_type_mapping", "differential_expression", "spatial_mapping"),
        complementary_forms=("connectomics", "mri", "clinical"),
        required_signals=("cells", "genes", "metadata"),
    ),
    "computational_model": DataForm(
        id="computational_model",
        label="Computational model or simulation",
        family="modeling",
        scale="multi_scale",
        aliases=("simulation", "model", "computational", "neural network model", "biophysical model"),
        modalities=("model_output", "simulation"),
        standards=("json", "hdf5"),
        analysis_families=("model_comparison", "parameter_inference", "mechanistic_modeling"),
        complementary_forms=("extracellular_ephys", "behavior", "molecular"),
        required_signals=("parameters", "outputs"),
    ),
}

ANALYSIS_ALIASES: dict[str, tuple[str, ...]] = {
    "decoding": ("decode", "decoding", "classifier", "classification", "predict choice"),
    "event_aligned_analysis": ("event aligned", "peri event", "psth", "trial aligned"),
    "connectivity": ("connectivity", "functional connectivity", "network", "connectome"),
    "behavioral_modeling": ("behavioral model", "q learning", "reinforcement learning", "rl model"),
    "clinical_prediction": ("clinical prediction", "diagnosis", "biomarker", "seizure detection"),
    "molecular_profiling": ("cell type", "gene expression", "transcriptomics", "differential expression"),
    "encoding_modeling": ("encoding model", "stimulus encoding", "receptive field"),
    "bci_decoding": ("bci", "brain computer interface", "motor imagery"),
}

SCALE_ALIASES: dict[str, tuple[str, ...]] = {
    "cellular": ("cellular", "single cell", "patch clamp"),
    "cell_population": ("population", "neurons", "units", "spikes"),
    "circuit": ("circuit", "network", "connectome"),
    "whole_brain": ("whole brain", "brain wide", "fmri", "eeg", "meg"),
    "organism": ("behavior", "task", "animal", "participant"),
}

SPECIES_ALIASES: dict[str, tuple[str, ...]] = {
    "mouse": ("mouse", "mice", "mus musculus"),
    "rat": ("rat", "rats"),
    "human": ("human", "participant", "patient", "patients"),
    "nonhuman_primate": ("macaque", "monkey", "nonhuman primate", "nhp"),
    "zebrafish": ("zebrafish", "danio"),
    "drosophila": ("drosophila", "fly"),
}

NEGATIVE_MARKERS = ("without", "excluding", "exclude", "not", "no", "but not", "NOT")


def _contains_term(text: str, term: str) -> bool:
    normalized_text = _norm(text)
    normalized_term = _norm(term)
    if not normalized_term:
        return False
    return re.search(rf"(?<!\w){re.escape(normalized_term)}(?!\w)", normalized_text) is not None


def detect_data_forms(text_or_values: str | list[str] | tuple[str, ...]) -> list[str]:
    """Detect broad data forms in free text or metadata values."""

    text = " ".join(text_or_values) if not isinstance(text_or_values, str) else text_or_values
    matches: list[str] = []
    for form_id, form in DATA_FORMS.items():
        if any(_contains_term(text, term) for term in form.match_terms()):
            matches.append(form_id)
    return sorted(dict.fromkeys(matches))


def _detect_negative_data_forms(query: str) -> list[str]:
    excluded: list[str] = []
    normalized = _norm(query)
    for marker in NEGATIVE_MARKERS:
        marker_norm = _norm(marker)
        if marker_norm not in normalized:
            continue
        trailing = normalized.split(marker_norm, 1)[1]
        for form_id, form in DATA_FORMS.items():
            if any(_contains_term(trailing, term) for term in form.match_terms()):
                excluded.append(form_id)
    return sorted(dict.fromkeys(excluded))


def infer_query_awareness(query: str) -> QueryAwareness:
    """Infer broad neuroscience data needs from a free-text query."""

    requested = detect_data_forms(query)
    excluded = _detect_negative_data_forms(query)
    requested = [form for form in requested if form not in excluded]
    analysis: list[str] = []
    for family, aliases in ANALYSIS_ALIASES.items():
        if any(_contains_term(query, alias) for alias in aliases):
            analysis.append(family)

    scale_terms: list[str] = []
    for scale, aliases in SCALE_ALIASES.items():
        if any(_contains_term(query, alias) for alias in aliases):
            scale_terms.append(scale)

    species_terms: list[str] = []
    for species, aliases in SPECIES_ALIASES.items():
        if any(_contains_term(query, alias) for alias in aliases):
            species_terms.append(species)

    required_signals: list[str] = []
    for form_id in requested:
        required_signals.extend(DATA_FORMS[form_id].required_signals)
    for family in analysis:
        if family == "decoding":
            required_signals.extend(["labels", "neural_signal"])
        elif family == "connectivity":
            required_signals.extend(["regions", "time_series"])
        elif family == "behavioral_modeling":
            required_signals.extend(["choices", "rewards", "trials"])

    return QueryAwareness(
        query=query,
        requested_data_forms=tuple(sorted(dict.fromkeys(requested))),
        analysis_families=tuple(sorted(dict.fromkeys(analysis))),
        scale_terms=tuple(sorted(dict.fromkeys(scale_terms))),
        species_terms=tuple(sorted(dict.fromkeys(species_terms))),
        required_signals=tuple(sorted(dict.fromkeys(required_signals))),
        excluded_data_forms=tuple(excluded),
    )
