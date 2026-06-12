"""Conservative label normalizers for species, modality, brain region, and data signals.

Rules:
- prefer structured metadata fields over free-text inference
- mark text-inferred fields as inferred=True
- keep unknown as None; do not guess
- do not infer raw data availability from archive or standard alone
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Species normalization
# ---------------------------------------------------------------------------

_SPECIES_MAP: dict[str, str] = {
    # mouse
    "mouse": "mouse",
    "mice": "mouse",
    "mus musculus": "mouse",
    "murine": "mouse",
    # rat
    "rat": "rat",
    "rattus norvegicus": "rat",
    # human
    "human": "human",
    "humans": "human",
    "homo sapiens": "human",
    "participants": "human",
    "subjects": "human",
    "patients": "human",
    "healthy volunteers": "human",
    "healthy adults": "human",
    # macaque / NHP
    "macaque": "macaque",
    "macaques": "macaque",
    "monkey": "macaque",
    "monkeys": "macaque",
    "rhesus": "macaque",
    "rhesus macaque": "macaque",
    "non-human primate": "non_human_primate",
    "non human primate": "non_human_primate",
    "nhp": "non_human_primate",
    # zebrafish
    "zebrafish": "zebrafish",
    "danio rerio": "zebrafish",
    # drosophila
    "drosophila": "drosophila",
    "fruit fly": "drosophila",
    "drosophila melanogaster": "drosophila",
}


def normalize_species(raw: str) -> str | None:
    """Return a canonical species key or None if unrecognized."""
    key = raw.strip().lower()
    return _SPECIES_MAP.get(key)


def extract_species_from_text(text: str) -> list[tuple[str, str]]:
    """Return [(canonical, matched_phrase), ...] found in free text."""
    results: list[tuple[str, str]] = []
    lowered = text.lower()
    for phrase, canonical in sorted(_SPECIES_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in lowered:
            results.append((canonical, phrase))
    seen: set[str] = set()
    deduped = []
    for canon, phrase in results:
        if canon not in seen:
            seen.add(canon)
            deduped.append((canon, phrase))
    return deduped


# ---------------------------------------------------------------------------
# Modality normalization
# ---------------------------------------------------------------------------

_MODALITY_MAP: dict[str, str] = {
    # extracellular electrophysiology
    "neuropixels": "extracellular_ephys",
    "neuropixels probe": "extracellular_ephys",
    "ap band": "extracellular_ephys",
    "ap-band": "extracellular_ephys",
    "spike sorting": "extracellular_ephys",
    "spike train": "extracellular_ephys",
    "spike trains": "extracellular_ephys",
    "single unit": "extracellular_ephys",
    "multi-unit activity": "extracellular_ephys",
    "multiunit": "extracellular_ephys",
    "tetrode": "extracellular_ephys",
    "silicon probe": "extracellular_ephys",
    "electrode array": "extracellular_ephys",
    "utah array": "extracellular_ephys",
    "extracellular electrophysiology": "extracellular_ephys",
    "extracellular recording": "extracellular_ephys",
    "extracellular ephys": "extracellular_ephys",
    # lfp
    "lfp": "lfp",
    "local field potential": "lfp",
    "lf-band": "lfp",
    "lf band": "lfp",
    # calcium imaging
    "two-photon": "calcium_imaging",
    "two photon": "calcium_imaging",
    "2p imaging": "calcium_imaging",
    "2-photon": "calcium_imaging",
    "2p": "calcium_imaging",
    "calcium imaging": "calcium_imaging",
    "calcium imag": "calcium_imaging",
    "gcamp": "calcium_imaging",
    "gcamp6": "calcium_imaging",
    "geci": "calcium_imaging",
    "fluorescence imaging": "calcium_imaging",
    "widefield imaging": "widefield",
    "widefield calcium imaging": "widefield",
    # fmri
    "fmri": "fmri",
    "bold": "fmri",
    "bold fmri": "fmri",
    "functional mri": "fmri",
    "functional magnetic resonance imaging": "fmri",
    # eeg
    "eeg": "eeg",
    "electroencephalography": "eeg",
    "electroencephalogram": "eeg",
    # meg
    "meg": "meg",
    "magnetoencephalography": "meg",
    # ecog
    "ecog": "ecog",
    "electrocorticography": "ecog",
    # intracellular
    "patch clamp": "intracellular_ephys",
    "whole cell patch clamp": "intracellular_ephys",
    "whole-cell patch": "intracellular_ephys",
    "intracellular recording": "intracellular_ephys",
    "intracellular electrophysiology": "intracellular_ephys",
    "sharp electrode": "intracellular_ephys",
}


def normalize_modality(raw: str) -> str | None:
    """Return a canonical modality key or None if unrecognized."""
    key = raw.strip().lower()
    return _MODALITY_MAP.get(key)


def extract_modalities_from_text(text: str) -> list[tuple[str, str]]:
    """Return [(canonical, matched_phrase), ...] found in free text."""
    lowered = text.lower()
    results: list[tuple[str, str]] = []
    for phrase, canonical in sorted(_MODALITY_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in lowered:
            results.append((canonical, phrase))
    seen: set[str] = set()
    deduped = []
    for canon, phrase in results:
        if canon not in seen:
            seen.add(canon)
            deduped.append((canon, phrase))
    return deduped


# ---------------------------------------------------------------------------
# Brain region normalization
# ---------------------------------------------------------------------------

_REGION_MAP: dict[str, str] = {
    # hippocampal
    "ca1": "hippocampus_ca1",
    "ca3": "hippocampus_ca3",
    "dg": "dentate_gyrus",
    "dentate gyrus": "dentate_gyrus",
    "hippocampus": "hippocampus",
    "hippocampal": "hippocampus",
    # entorhinal
    "mec": "entorhinal_cortex",
    "medial entorhinal cortex": "entorhinal_cortex",
    "lec": "entorhinal_cortex",
    "lateral entorhinal cortex": "entorhinal_cortex",
    "entorhinal cortex": "entorhinal_cortex",
    # visual
    "visp": "visual_cortex",
    "v1": "visual_cortex",
    "primary visual cortex": "visual_cortex",
    "visual cortex": "visual_cortex",
    "mt": "mt_v5",
    "v5": "mt_v5",
    "mt/v5": "mt_v5",
    "area mt": "mt_v5",
    "middle temporal area": "mt_v5",
    # parietal
    "lip": "lateral_intraparietal",
    "lateral intraparietal": "lateral_intraparietal",
    "lateral intraparietal area": "lateral_intraparietal",
    # prefrontal
    "pfc": "prefrontal_cortex",
    "mpfc": "prefrontal_cortex",
    "dlpfc": "prefrontal_cortex",
    "prefrontal cortex": "prefrontal_cortex",
    "medial prefrontal cortex": "prefrontal_cortex",
    "dorsolateral prefrontal cortex": "prefrontal_cortex",
    "orbitofrontal cortex": "orbitofrontal_cortex",
    "ofc": "orbitofrontal_cortex",
    # striatum
    "striatum": "striatum",
    "caudate": "striatum",
    "caudate nucleus": "striatum",
    "putamen": "striatum",
    "nucleus accumbens": "striatum",
    # cerebellum
    "cerebellum": "cerebellum",
    # thalamus
    "thalamus": "thalamus",
    # amygdala
    "amygdala": "amygdala",
}


def normalize_brain_region(raw: str) -> str | None:
    """Return a canonical brain region key or None if unrecognized."""
    key = raw.strip().lower()
    return _REGION_MAP.get(key)


def extract_regions_from_text(text: str) -> list[tuple[str, str]]:
    """Return [(canonical, matched_phrase), ...] found in free text."""
    lowered = text.lower()
    results: list[tuple[str, str]] = []
    for phrase, canonical in sorted(_REGION_MAP.items(), key=lambda x: -len(x[0])):
        if phrase in lowered:
            results.append((canonical, phrase))
    seen: set[str] = set()
    deduped = []
    for canon, phrase in results:
        if canon not in seen:
            seen.add(canon)
            deduped.append((canon, phrase))
    return deduped


# ---------------------------------------------------------------------------
# Raw vs processed data signal extraction
#
# Rules:
# - do NOT infer raw data availability from BIDS/NWB/ALF standard alone
# - calcium imaging does NOT satisfy extracellular spike affordances
# - fMRI does NOT satisfy spike/LFP affordances
# - only positive keyword evidence triggers raw_available=True
# ---------------------------------------------------------------------------

_RAW_DATA_KEYWORDS: frozenset[str] = frozenset({
    "raw ap",
    "ap-band",
    "ap band",
    "lf-band",
    "lf band",
    "raw voltage",
    "continuous data",
    "raw imaging",
    "raw imaging movie",
    "tiff stack",
    "tiff stacks",
    "acquisition data",
    ".ap.bin",
    ".lf.bin",
    "raw recording",
    "raw neural",
    "continuous neural",
    "raw electrophysiology",
    "continuous electrophysiology",
})

_PROCESSED_ONLY_KEYWORDS: frozenset[str] = frozenset({
    "spike times only",
    "clusters table",
    "alf processed",
    "roi traces only",
    "derived features",
    "summary statistics",
    "derivatives only",
    "preprocessed",
    "processed traces",
    "spike sorted only",
    "sorted spikes",
    "sorted units",
    "processed data only",
    "analysis ready",
})

_NEGATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bno\s+raw\b"),
    re.compile(r"\bnot\s+raw\b"),
    re.compile(r"\bwithout\s+raw\b"),
    re.compile(r"\bno\s+continuous\b"),
]


def extract_raw_data_evidence(text: str) -> tuple[bool | None, list[str]]:
    """Return (raw_available, matched_phrases).

    Returns None for raw_available when evidence is insufficient.
    Never infers from archive or data standard name alone.
    """
    lowered = text.lower()
    matched: list[str] = []

    for kw in _RAW_DATA_KEYWORDS:
        if kw in lowered:
            matched.append(kw)

    # Check for negations that override positive keywords
    for pat in _NEGATION_PATTERNS:
        if pat.search(lowered):
            return False, [pat.pattern]

    if matched:
        return True, matched
    return None, []


def extract_processed_data_evidence(text: str) -> tuple[bool | None, list[str]]:
    """Return (processed_available, matched_phrases)."""
    lowered = text.lower()
    matched: list[str] = []
    for kw in _PROCESSED_ONLY_KEYWORDS:
        if kw in lowered:
            matched.append(kw)
    if matched:
        return True, matched
    return None, []


# ---------------------------------------------------------------------------
# Cross-modality guardrails
# ---------------------------------------------------------------------------

_MODALITY_AFFORDANCE_CONSTRAINTS: dict[str, frozenset[str]] = {
    "calcium_imaging": frozenset({"extracellular_ephys", "lfp", "spike_sorting"}),
    "widefield": frozenset({"extracellular_ephys", "lfp", "spike_sorting"}),
    "fmri": frozenset({"extracellular_ephys", "lfp", "spike_sorting", "calcium_imaging"}),
}

_SPECIES_INTERCHANGEABILITY_VIOLATIONS: frozenset[frozenset[str]] = frozenset({
    frozenset({"human", "mouse"}),
    frozenset({"human", "rat"}),
    frozenset({"human", "macaque"}),
    frozenset({"human", "non_human_primate"}),
})

_REGION_EQUIVALENCE_VIOLATIONS: frozenset[frozenset[str]] = frozenset({
    frozenset({"mt_v5", "lateral_intraparietal"}),
    frozenset({"prefrontal_cortex", "lateral_intraparietal"}),
    frozenset({"entorhinal_cortex", "hippocampus"}),
})


def modality_satisfies_affordance(modality: str, affordance: str) -> bool:
    """Return False when modality cannot satisfy the affordance (hard rule)."""
    blocked = _MODALITY_AFFORDANCE_CONSTRAINTS.get(modality, frozenset())
    return affordance.lower() not in blocked


def species_are_interchangeable(a: str, b: str) -> bool:
    """Return False when species are definitively non-interchangeable."""
    pair = frozenset({a.lower(), b.lower()})
    return pair not in _SPECIES_INTERCHANGEABILITY_VIOLATIONS


def regions_are_equivalent(a: str, b: str) -> bool:
    """Return False when regions are definitively non-equivalent."""
    pair = frozenset({a.lower(), b.lower()})
    return pair not in _REGION_EQUIVALENCE_VIOLATIONS
