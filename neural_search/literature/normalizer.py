"""Post-processing normalizer for extracted FindingRecord JSONL files.

Applies five normalization passes in order:
  1. Species canonical forms  (humans→human, mice→mouse, …)
  2. Region string cleanup    (underscores, abbreviations, generic terms)
  3. Task canonical forms     (consolidate closely related task strings)
  4. Finding deduplication    (exact finding_text dedup within a paper)
  5. Quality tagging          (adds a `quality_flags` list to each record)

All passes are non-destructive: the original fields are kept, and an
`_normalized` dict is added for any field that was changed.  A
`quality_flags` list is appended so downstream consumers can filter by
flag without re-running normalization.
"""

from __future__ import annotations

import re
from typing import Any

from neural_search.literature.typed_finding_extractor import enrich_finding

# ---------------------------------------------------------------------------
# Species
# ---------------------------------------------------------------------------

# Map every surface form → canonical string
_SPECIES_CANONICAL: dict[str, str] = {
    # Homo sapiens
    "humans": "human",
    "human subjects": "human",
    "human participants": "human",
    "people": "human",
    "person": "human",
    "patients": "human",
    "healthy adults": "human",
    "adults": "human",
    # Mus musculus
    "mice": "mouse",
    "mus musculus": "mouse",
    "c57bl/6": "mouse",
    "balb/c": "mouse",
    # Rattus norvegicus
    "rats": "rat",
    "rattus norvegicus": "rat",
    "sprague-dawley": "rat",
    "wistar": "rat",
    # Non-human primates
    "monkeys": "monkey",
    "macaques": "macaque",
    "macaca mulatta": "macaque",
    "rhesus": "macaque",
    "rhesus macaque": "macaque",
    "marmosets": "marmoset",
    # Other
    "cats": "cat",
    "ferrets": "ferret",
    "rabbits": "rabbit",
    "zebrafish": "zebrafish",
    "drosophila": "drosophila",
    "c. elegans": "c. elegans",
    # Non-specific
    "mammals": "mammal",
    "animals": "animal",
    "rodents": "rodent",
    "primates": "primate",
}


def normalize_species(species_list: list[str]) -> tuple[list[str], bool]:
    """Return (normalized_list, changed).

    Applies case-insensitive lookup against _SPECIES_CANONICAL; keeps
    unrecognised entries unchanged.
    """
    result = []
    changed = False
    seen: set[str] = set()
    for s in species_list:
        canonical = _SPECIES_CANONICAL.get(s.lower().strip(), s.lower().strip())
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
        if canonical != s:
            changed = True
    return result, changed


# ---------------------------------------------------------------------------
# Regions
# ---------------------------------------------------------------------------

# Too generic to be useful for KG linking — flagged, not removed
_GENERIC_REGIONS: set[str] = {
    "brain",
    "cortex",
    "neocortex",
    "cerebral cortex",
    "cns",
    "central nervous system",
    "nervous system",
    "neural",
    "neuronal",
    "neurons",
    "neuron",
    "nerve",
    "network",
    "cortical",
    "gray matter",
    "grey matter",
    "white matter",
    "neuropil",
}

# Abbreviation / alias → canonical long form
_REGION_ALIASES: dict[str, str] = {
    # Primary areas
    "v1": "primary visual cortex",
    "v2": "secondary visual cortex",
    "v4": "visual area v4",
    "mt": "middle temporal area",
    "m1": "primary motor cortex",
    "s1": "primary somatosensory cortex",
    "a1": "primary auditory cortex",
    # Prefrontal
    "pfc": "prefrontal cortex",
    "dlpfc": "dorsolateral prefrontal cortex",
    "ofc": "orbitofrontal cortex",
    "mpfc": "medial prefrontal cortex",
    "vmPFC": "ventromedial prefrontal cortex",
    # Hippocampal / medial temporal
    "mtl": "medial temporal lobe",
    "ca1": "CA1",
    "ca3": "CA3",
    "dg": "dentate gyrus",
    # Subcortical
    "nac": "nucleus accumbens",
    "acc": "anterior cingulate cortex",
    "vlpfc": "ventrolateral prefrontal cortex",
    "bla": "basolateral amygdala",
    "cea": "central amygdala",
    "sn": "substantia nigra",
    "snc": "substantia nigra pars compacta",
    "snr": "substantia nigra pars reticulata",
    "gp": "globus pallidus",
    "lc": "locus coeruleus",
    "dr": "dorsal raphe",
    "vta": "ventral tegmental area",
    "pvn": "paraventricular nucleus",
    "stn": "subthalamic nucleus",
    # Cerebellum
    "cb": "cerebellum",
    # Brain stem
    "pag": "periaqueductal gray",
    # Spinal
    "sc": "spinal cord",
    # Peripheral
    "dgn": "dorsal root ganglion",
    "drg": "dorsal root ganglion",
}

# Fuzzy corrections for underscore-joined strings and common typos
_REGION_CLEANUPS: list[tuple[str, str]] = [
    (r"_", " "),         # prefrontal_cortex → prefrontal cortex
    (r"\bctx\b", "cortex"),
    (r"\bregion\b", ""),
    (r"\barea\b", ""),
]


def _clean_region_string(r: str) -> str:
    r = r.strip().lower()
    for pattern, replacement in _REGION_CLEANUPS:
        r = re.sub(pattern, replacement, r)
    r = re.sub(r"\s+", " ", r).strip()
    return r


def normalize_regions(
    regions: list[str],
) -> tuple[list[str], list[str], bool]:
    """Return (normalized_regions, generic_regions_flagged, changed).

    * Applies underscore cleanup and alias expansion.
    * Separates generic regions into a second list for quality flagging.
    """
    normalized: list[str] = []
    generic_flagged: list[str] = []
    changed = False
    seen: set[str] = set()

    for r in regions:
        cleaned = _clean_region_string(r)
        canonical = _REGION_ALIASES.get(cleaned, cleaned)
        if canonical != r:
            changed = True
        if canonical in _GENERIC_REGIONS:
            generic_flagged.append(canonical)
            changed = True
            continue
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)

    return normalized, generic_flagged, changed


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

_TASK_CANONICAL: dict[str, str] = {
    # Memory variants
    "working memory task": "working memory",
    "spatial working memory": "working memory",
    "verbal working memory": "working memory",
    "visual working memory": "working memory",
    "short-term memory": "working memory",
    "long-term memory": "memory",
    "memory encoding": "memory",
    "memory retrieval": "memory",
    "episodic memory": "memory",
    "recognition memory": "memory",
    # Spatial
    "spatial navigation": "navigation",
    "place learning": "navigation",
    "maze learning": "navigation",
    "spatial learning": "navigation",
    # Decision making
    "decision-making": "decision making",
    "value-based decision making": "decision making",
    "economic decision making": "decision making",
    "risky decision making": "decision making",
    # Visual
    "visual perception": "visual perception",
    "visual processing": "visual perception",
    "visual discrimination": "visual discrimination",
    # Motor
    "motor learning": "motor learning",
    "motor control": "motor control",
    "motor task": "motor task",
    # Language
    "language comprehension": "language",
    "reading comprehension": "reading",
    "speech production": "speech",
    "speech recognition": "speech perception",
    "auditory processing": "auditory processing",
    # Reward
    "reward processing": "reward",
    "reward learning": "reward",
    "reinforcement learning task": "reinforcement learning",
    # Attention
    "selective attention": "attention",
    "sustained attention": "attention",
    "attentional control": "attention",
    # Conditioning
    "fear conditioning": "fear conditioning",
    "fear extinction": "fear extinction",
    "classical conditioning": "conditioning",
    "pavlovian conditioning": "conditioning",
}


# ---------------------------------------------------------------------------
# Cell types
# ---------------------------------------------------------------------------

_CELL_TYPE_CANONICAL: dict[str, str] = {
    # Neurons (generic)
    "neurons": "neuron",
    "neurones": "neuron",
    "nerve cells": "neuron",
    # Pyramidal
    "pyramidal neurons": "pyramidal cell",
    "pyramidal neuron": "pyramidal cell",
    "pyramidal cells": "pyramidal cell",
    # Interneurons
    "interneurons": "interneuron",
    "inhibitory neurons": "interneuron",
    "gaba neurons": "GABAergic interneuron",
    "gabaergic neurons": "GABAergic interneuron",
    "parvalbumin neurons": "parvalbumin interneuron",
    "pv neurons": "parvalbumin interneuron",
    "pv interneurons": "parvalbumin interneuron",
    "somatostatin neurons": "somatostatin interneuron",
    "sst neurons": "somatostatin interneuron",
    # Dopaminergic
    "dopaminergic neurons": "dopaminergic neuron",
    "dopamine neurons": "dopaminergic neuron",
    "da neurons": "dopaminergic neuron",
    # Cholinergic
    "cholinergic neurons": "cholinergic neuron",
    # Serotonergic
    "serotonergic neurons": "serotonergic neuron",
    "5-ht neurons": "serotonergic neuron",
    # Noradrenergic
    "noradrenergic neurons": "noradrenergic neuron",
    "norepinephrine neurons": "noradrenergic neuron",
    # Specific types
    "place cells": "place cell",
    "grid cells": "grid cell",
    "border cells": "border cell",
    "head direction cells": "head direction cell",
    "purkinje cells": "Purkinje cell",
    "granule cells": "granule cell",
    "motor neurons": "motor neuron",
    "motoneurons": "motor neuron",
    "sensory neurons": "sensory neuron",
    "retinal ganglion cells": "retinal ganglion cell",
    "ganglion cells": "retinal ganglion cell",
    "neural stem cells": "neural stem cell",
    # Glia
    "microglial cells": "microglia",
    "microglial": "microglia",
    "astrocytes": "astrocyte",
    "oligodendrocytes": "oligodendrocyte",
    "glial cells": "glia",
    "endothelial cells": "endothelial cell",
    "macrophages": "macrophage",
}


def normalize_cell_types(cell_types: list[str]) -> tuple[list[str], bool]:
    result = []
    changed = False
    seen: set[str] = set()
    for ct in cell_types:
        canonical = _CELL_TYPE_CANONICAL.get(ct.lower().strip(), ct.strip())
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
        if canonical != ct:
            changed = True
    return result, changed


# ---------------------------------------------------------------------------
# Molecules
# ---------------------------------------------------------------------------

_MOLECULE_CANONICAL: dict[str, str] = {
    # Neurotransmitters
    "dopamine": "dopamine",
    "da": "dopamine",
    "serotonin": "serotonin",
    "5-ht": "serotonin",
    "5ht": "serotonin",
    "norepinephrine": "norepinephrine",
    "noradrenaline": "norepinephrine",
    "ne": "norepinephrine",
    "glutamate": "glutamate",
    "gaba": "GABA",
    "gamma-aminobutyric acid": "GABA",
    "acetylcholine": "acetylcholine",
    "ach": "acetylcholine",
    # Receptors
    "nmda receptor": "NMDA receptor",
    "nmda receptors": "NMDA receptor",
    "ampa receptor": "AMPA receptor",
    "ampa receptors": "AMPA receptor",
    "d1 receptor": "D1 receptor",
    "d2 receptor": "D2 receptor",
    "gaba-a receptor": "GABA-A receptor",
    "gaba-b receptor": "GABA-B receptor",
    # Neuropeptides
    "bdnf": "BDNF",
    "brain-derived neurotrophic factor": "BDNF",
    "ngf": "NGF",
    "nerve growth factor": "NGF",
    "oxytocin": "oxytocin",
    "vasopressin": "vasopressin",
    "substance p": "substance P",
    # Calcium
    "ca2+": "calcium",
    "ca2": "calcium",
    "calcium ions": "calcium",
    # Disease proteins
    "α-synuclein": "alpha-synuclein",
    "alpha-synuclein": "alpha-synuclein",
    "tau protein": "tau",
    "amyloid beta": "amyloid-beta",
    "aβ": "amyloid-beta",
    "abeta": "amyloid-beta",
    "tdp-43": "TDP-43",
    # Kinases
    "erk": "ERK",
    "erk1/2": "ERK",
    "mapk": "MAPK",
    "camkii": "CaMKII",
    "pkc": "PKC",
    "pka": "PKA",
    # Inflammatory
    "il-6": "IL-6",
    "il-1β": "IL-1beta",
    "tnf-α": "TNF-alpha",
    "tnf-alpha": "TNF-alpha",
}


def normalize_molecules(molecules: list[str]) -> tuple[list[str], bool]:
    result = []
    changed = False
    seen: set[str] = set()
    for m in molecules:
        canonical = _MOLECULE_CANONICAL.get(m.lower().strip(), m.strip())
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
        if canonical != m:
            changed = True
    return result, changed


def normalize_tasks(tasks: list[str]) -> tuple[list[str], bool]:
    result = []
    changed = False
    seen: set[str] = set()
    for t in tasks:
        canonical = _TASK_CANONICAL.get(t.lower().strip(), t.lower().strip())
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
        if canonical != t:
            changed = True
    return result, changed


# ---------------------------------------------------------------------------
# Quality flags
# ---------------------------------------------------------------------------

_META_LANGUAGE_PREFIXES = (
    "this study",
    "in this study",
    "in this paper",
    "in this work",
    "we show",
    "we demonstrate",
    "we report",
    "we present",
    "we found",
    "our results",
    "our study",
    "results show",
    "results indicate",
    "results suggest",
    "the results",
    "the study",
    "the paper",
)


def compute_quality_flags(
    finding: dict[str, Any],
    generic_regions: list[str],
) -> list[str]:
    """Return a list of quality flag strings for downstream filtering.

    Flags (non-exhaustive):
      generic_region_only   — all regions stripped as too broad
      no_region             — regions list is empty after normalization
      no_species            — species list is empty
      low_confidence        — confidence < 0.7
      meta_language         — finding_text starts with boilerplate
      very_short_finding    — finding_text < 30 chars
    """
    flags: list[str] = []
    text = finding.get("finding_text", "")
    regions = finding.get("regions", [])
    species = finding.get("species", [])
    confidence = finding.get("confidence", 1.0)

    if generic_regions and not regions:
        flags.append("generic_region_only")
    elif not regions:
        flags.append("no_region")

    if not species:
        flags.append("no_species")

    if confidence < 0.7:
        flags.append("low_confidence")

    lower_text = text.lower()
    if any(lower_text.startswith(p) for p in _META_LANGUAGE_PREFIXES):
        flags.append("meta_language")

    if len(text) < 30:
        flags.append("very_short_finding")

    return flags


# ---------------------------------------------------------------------------
# Main normalization pass
# ---------------------------------------------------------------------------


def normalize_finding(record: dict[str, Any]) -> dict[str, Any]:
    """Apply all normalization passes to a single finding record dict.

    Returns a new dict (the original is never mutated).  Changes are
    recorded in `_normalized` and quality issues in `quality_flags`.
    """
    out = enrich_finding(record)  # merges the 27 typed extension fields in
    changes: dict[str, Any] = {}

    # --- species ---
    norm_species, sp_changed = normalize_species(list(out.get("species", [])))
    if sp_changed:
        changes["species_original"] = out["species"]
        out["species"] = norm_species

    # --- regions ---
    norm_regions, generic, reg_changed = normalize_regions(list(out.get("regions", [])))
    if reg_changed:
        changes["regions_original"] = out["regions"]
        out["regions"] = norm_regions
        if generic:
            changes["generic_regions"] = generic
    else:
        generic = []

    # --- tasks ---
    norm_tasks, task_changed = normalize_tasks(list(out.get("tasks", [])))
    if task_changed:
        changes["tasks_original"] = out["tasks"]
        out["tasks"] = norm_tasks

    # --- cell types ---
    norm_cells, cell_changed = normalize_cell_types(list(out.get("cell_types", [])))
    if cell_changed:
        changes["cell_types_original"] = out["cell_types"]
        out["cell_types"] = norm_cells

    # --- molecules ---
    norm_mols, mol_changed = normalize_molecules(list(out.get("molecules", [])))
    if mol_changed:
        changes["molecules_original"] = out["molecules"]
        out["molecules"] = norm_mols

    # --- quality flags ---
    flags = compute_quality_flags(out, generic)

    if changes:
        out["_normalized"] = changes
    if flags:
        out["quality_flags"] = flags

    return out


def deduplicate_findings(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Remove exact-duplicate finding_text entries within each paper.

    Returns (deduplicated_records, n_removed).
    """
    seen: dict[str, set[str]] = {}  # paper_id → set of finding_texts
    result: list[dict[str, Any]] = []
    removed = 0

    for r in records:
        pid = r.get("paper_id", "")
        text = r.get("finding_text", "")
        if pid not in seen:
            seen[pid] = set()
        key = text.strip().lower()
        if key in seen[pid]:
            removed += 1
            continue
        seen[pid].add(key)
        result.append(r)

    return result, removed
