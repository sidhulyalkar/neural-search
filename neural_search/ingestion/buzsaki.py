"""Buzsaki Lab dataset adapter.

The Buzsaki Lab (NYU) publicly hosts ~36 curated rodent ephys datasets at
https://buzsakilab.nyumc.org/datasets/ — mostly hippocampal and cortical
in-vivo recordings from freely moving rats and mice.

Many datasets are also mirrored on CRCNS (hc-3, hc-11, etc.) or DANDI.
This adapter adds them as a dedicated source with Buzsaki-specific metadata.
"""
from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from neural_search.ingestion.registry import register

logger = logging.getLogger(__name__)

BUZSAKI_BASE = "https://buzsakilab.nyumc.org/datasets"

# Curated metadata for each PI's dataset folder.
# Derived from lab publications and dataset READMEs.
_DATASET_META: dict[str, dict[str, Any]] = {
    "BerenyiT": {
        "title": "Berenyi: Closed-loop neuromodulation of hippocampal-cortical oscillations",
        "brain_regions": ["hippocampus", "prefrontal_cortex"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Multi-site silicon probe recordings from hippocampus (CA1) and mPFC "
            "during open-field and sleep. Closed-loop theta-phase targeted stimulation."
        ),
    },
    "CsicsvariJ": {
        "title": "Csicsvari: Hippocampal unit and LFP recordings during sharp-wave ripples",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Single-unit and LFP recordings from hippocampal CA1 and CA3 in freely "
            "moving rats. Focus on sharp-wave ripples and neuronal firing patterns."
        ),
    },
    "DibaK": {
        "title": "Diba: Hippocampal sequence reactivation and replay during sleep",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Silicon probe recordings from hippocampal CA1 and CA3 in rats during "
            "maze running and subsequent sleep. Forward and reverse replay events."
        ),
    },
    "EnglishD": {
        "title": "English: Excitatory-inhibitory balance during hippocampal oscillations",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["mouse"],
        "description": (
            "Dense multi-tetrode recordings from mouse hippocampal CA1. "
            "Excitatory and inhibitory unit classification during theta and SWR states."
        ),
    },
    "FernandezRuiz_Oliva": {
        "title": "Fernandez-Ruiz & Oliva: Entorhinal-hippocampal coordination",
        "brain_regions": ["hippocampus", "entorhinal_cortex"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Silicon probe recordings spanning entorhinal cortex and hippocampal CA1 "
            "in rats. Nested gamma oscillations and EC-CA1 coupling."
        ),
    },
    "FujisawaS": {
        "title": "Fujisawa: Hippocampal-prefrontal interaction during spatial learning",
        "brain_regions": ["hippocampus", "prefrontal_cortex"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Simultaneous multi-electrode recordings from hippocampus (CA1) and "
            "prefrontal cortex (mPFC) in rats performing a spatial alternation task."
        ),
    },
    "GelinasJ": {
        "title": "Gelinas: Human intracranial EEG hippocampal recordings",
        "brain_regions": ["hippocampus", "entorhinal_cortex"],
        "modalities": ["lfp", "ecog"],
        "species": ["human"],
        "description": (
            "Intracranial EEG recordings from patients with medial temporal lobe "
            "epilepsy. Theta and ripple oscillations in hippocampus and EC."
        ),
    },
    "GirardeauG": {
        "title": "Girardeau: Sharp-wave ripple disruption impairs spatial memory",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Tetrode recordings from hippocampal CA1 with targeted ripple disruption. "
            "Demonstrates role of SPW-Rs in memory consolidation."
        ),
    },
    "GrosmarkAD": {
        "title": "Grosmark: Sleep-dependent synaptic renormalization in hippocampus",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Silicon probe recordings from hippocampal CA1 (Achilles, Buddy, Cicero, "
            "Gatsby) during wake and sleep. Synaptic downscaling across sleep."
        ),
    },
    "HainmuellerT": {
        "title": "Hainmueller: Parallel map discovery in hippocampal area CA2",
        "brain_regions": ["hippocampus"],
        "modalities": ["calcium_imaging"],
        "species": ["mouse"],
        "description": (
            "Two-photon calcium imaging of hippocampal CA2 place cells in mice "
            "navigating virtual linear tracks."
        ),
    },
    "HeynoldAE": {
        "title": "Heynold: Hippocampal sharp-wave ripple dynamics across sleep",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Silicon probe recordings from hippocampal CA1 tracking sharp-wave ripple "
            "dynamics across sleep episodes in freely moving rats."
        ),
    },
    "HuszarR": {
        "title": "Huszar: Preconfigured dynamics in hippocampal population activity",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys"],
        "species": ["rat"],
        "description": (
            "High-density silicon probe recordings from hippocampal CA1 and CA3. "
            "Preconfigured attractor dynamics before spatial experience."
        ),
    },
    "LongJ": {
        "title": "Long: Hippocampal replay and consolidation across behavioral sessions",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Tetrode recordings from hippocampal CA1 across multiple behavioral "
            "sessions and intervening sleep episodes."
        ),
    },
    "MaslarovaA": {
        "title": "Maslarova: Entorhinal-hippocampal propagation of sharp-wave ripples",
        "brain_regions": ["hippocampus", "entorhinal_cortex"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Multi-site recordings spanning entorhinal cortex layer II/III and "
            "hippocampal CA1 during sleep and quiet wakefulness."
        ),
    },
    "McKenzieS": {
        "title": "McKenzie: Hippocampal representation of social information",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["mouse"],
        "description": (
            "Silicon probe recordings from hippocampal CA1 in mice during social "
            "interaction. Social and non-social place-field encoding."
        ),
    },
    "MizusekiK": {
        "title": "Mizuseki: Hippocampal theta oscillations and unit activity",
        "brain_regions": ["hippocampus", "entorhinal_cortex"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Multi-tetrode recordings from hippocampal CA1, CA3, and entorhinal "
            "cortex during theta oscillations in freely moving rats on a linear track."
        ),
    },
    "MontgomeryS": {
        "title": "Montgomery: Hippocampal-cortical temporal coding during sleep",
        "brain_regions": ["hippocampus", "prefrontal_cortex"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Multi-site recordings from hippocampus and prefrontal cortex during "
            "sleep. Temporal coordination of ripples and spindles."
        ),
    },
    "PastalkovaE": {
        "title": "Pastalkova: Internally generated hippocampal sequences",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Multi-tetrode recordings from hippocampal CA1 while rats ran on a "
            "running wheel between goal-directed runs. Time-cell sequences."
        ),
    },
    "PatelJ": {
        "title": "Patel: Hippocampal speed and direction coding",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Tetrode recordings from hippocampal CA1 and CA3 in rats running on "
            "treadmill and open field. Speed-modulated firing."
        ),
    },
    "PetersenP": {
        "title": "Petersen: Hippocampal theta and gamma during open-field exploration",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Silicon probe recordings from hippocampal CA1 during open-field "
            "exploration. CellExplorer dataset — theta/gamma coupling."
        ),
    },
    "PeyracheA": {
        "title": "Peyrache: Prefrontal cortex replay during sleep",
        "brain_regions": ["prefrontal_cortex", "hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Multi-tetrode recordings from medial prefrontal cortex and hippocampus "
            "during rule-based learning and post-task sleep. Sleep spindle replay."
        ),
    },
    "RoyerS": {
        "title": "Royer: Dissociable representations in hippocampal CA1 and CA3",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["mouse"],
        "description": (
            "Silicon probe recordings from hippocampal CA1 and CA3 in mice running "
            "on a linear track. Pattern completion vs. pattern separation."
        ),
    },
    "SenzaiY": {
        "title": "Senzai: Dentate gyrus cell type diversity and spatial coding",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["mouse"],
        "description": (
            "Silicon probe recordings from hippocampal dentate gyrus (granule cells, "
            "mossy cells, interneurons) in mice during spatial navigation."
        ),
    },
    "SiegleJ": {
        "title": "Siegle: Neuropixels 2.0 survey across mouse brain regions",
        "brain_regions": ["visual_cortex", "hippocampus", "thalamus", "striatum"],
        "modalities": ["extracellular_ephys", "neuropixels"],
        "species": ["mouse"],
        "description": (
            "Multi-region Neuropixels 2.0 recordings from mouse visual cortex (V1), "
            "hippocampus, thalamus, and striatum during passive visual stimulation."
        ),
    },
    "SoulaM": {
        "title": "Soula: Hippocampal place cell remapping and ensemble dynamics",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["mouse"],
        "description": (
            "Silicon probe recordings from hippocampal CA1 in mice across familiar "
            "and novel environments. Place cell remapping and ensemble transitions."
        ),
    },
    "StarkE": {
        "title": "Stark: Hippocampal interneuron control of sharp-wave ripples",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["mouse"],
        "description": (
            "Silicon probe recordings from hippocampal CA1 with optogenetic "
            "interneuron control. Mechanisms of ripple generation and termination."
        ),
    },
    "SteinmetzN": {
        "title": "Steinmetz: Neuropixels survey of mouse brain during visual tasks",
        "brain_regions": ["visual_cortex", "hippocampus", "thalamus", "striatum",
                          "motor_cortex", "prefrontal_cortex"],
        "modalities": ["extracellular_ephys", "neuropixels"],
        "species": ["mouse"],
        "description": (
            "Multi-region Neuropixels recordings from 42 brain areas in 10 mice "
            "performing a visual go/no-go decision task. Steinmetz et al. 2019."
        ),
    },
    "TingleyD": {
        "title": "Tingley: Hippocampal-striatal coordination during goal-directed behavior",
        "brain_regions": ["hippocampus", "striatum"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Multi-site recordings from hippocampus and dorsal striatum in rats "
            "performing reward-based navigation tasks."
        ),
    },
    "ValeroM": {
        "title": "Valero: Inhibitory-dominant hippocampal CA1 ripple assemblies",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Dense silicon probe recordings from hippocampal CA1 during sharp-wave "
            "ripples. Pyramidal cell and interneuron firing dynamics."
        ),
    },
    "VargaV": {
        "title": "Varga: Septohippocampal modulation of hippocampal oscillations",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Recordings from hippocampal CA1 and medial septum in rats. "
            "Septal pacemaking of hippocampal theta rhythm."
        ),
    },
    "VoroslakosM": {
        "title": "Voroslakos: Transcranial electric stimulation of hippocampal activity",
        "brain_regions": ["hippocampus", "prefrontal_cortex"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Silicon probe recordings from hippocampus and prefrontal cortex during "
            "transcranial alternating current stimulation (tACS)."
        ),
    },
    "WatsonBO": {
        "title": "Watson: Network homeostasis and state dynamics during sleep",
        "brain_regions": ["hippocampus", "prefrontal_cortex"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Multi-site recordings from hippocampus and prefrontal cortex across "
            "sleep-wake cycles. Homeostatic regulation of neural activity."
        ),
    },
    "YaghmazadehO": {
        "title": "Yaghmazadeh: Hippocampal interneuron diversity and network coordination",
        "brain_regions": ["hippocampus"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["mouse"],
        "description": (
            "Dense silicon probe recordings from hippocampal CA1. Classification "
            "of interneuron subtypes and their network function."
        ),
    },
    "ZhangY": {
        "title": "Zhang: Hippocampal-neocortical dialogue during memory consolidation",
        "brain_regions": ["hippocampus", "somatosensory_cortex"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Multi-site recordings from hippocampus and neocortex during sleep. "
            "Down-up state coordination and ripple-spindle coupling."
        ),
    },
    "ZutshiI": {
        "title": "Zutshi: Rapid place field remapping and memory encoding in hippocampus",
        "brain_regions": ["hippocampus", "entorhinal_cortex"],
        "modalities": ["extracellular_ephys", "lfp"],
        "species": ["rat"],
        "description": (
            "Silicon probe recordings from hippocampal CA1 and medial entorhinal "
            "cortex. Rapid acquisition of new spatial maps."
        ),
    },
}

_PROVENANCE = "buzsaki_lab_curated_silver_not_human_gold"


def _build_record(slug: str) -> dict[str, Any]:
    meta = _DATASET_META.get(slug, {})
    url = f"{BUZSAKI_BASE}/{slug}/"
    title = meta.get("title") or f"Buzsaki Lab: {slug}"
    description = meta.get("description") or ""
    brain_regions = meta.get("brain_regions", [])
    modalities = meta.get("modalities", ["extracellular_ephys"])
    species = meta.get("species", [])

    return {
        "source": "buzsaki",
        "source_id": slug,
        "title": title,
        "description": description,
        "url": url,
        "license": "CC-BY 4.0",
        "species": species,
        "modalities": modalities,
        "brain_regions": brain_regions,
        "tasks": [],
        "behaviors": [],
        "data_standards": [],
        "has_behavior": True,
        "has_trials": True,
        "has_raw_data": True,
        "has_processed_data": True,
        "metadata_json": {
            "raw_source": "buzsaki_lab",
            "pi": slug,
            "data_server": "buzsakilab.nyumc.org",
            "provenance": _PROVENANCE,
        },
    }


def _list_slugs(client: httpx.Client) -> list[str]:
    """Scrape the Buzsaki Lab dataset listing for folder names."""
    try:
        resp = client.get(f"{BUZSAKI_BASE}/", timeout=15)
        resp.raise_for_status()
        return re.findall(r'href="/datasets/([A-Z][^/"]+)/"', resp.text)
    except Exception as exc:
        logger.warning("Buzsaki listing failed: %s — using known list", exc)
        return list(_DATASET_META.keys())


@register("buzsaki")
def fetch_buzsaki_records(limit: int = 50) -> list[dict[str, Any]]:
    """Return corpus records for Buzsaki Lab datasets."""
    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        slugs = _list_slugs(client)

    # NWB folder is a format folder, not a dataset
    slugs = [s for s in slugs if s != "NWB"]

    records: list[dict[str, Any]] = []
    for slug in slugs[:limit]:
        if slug in _DATASET_META:
            records.append(_build_record(slug))
        else:
            logger.debug("No curated metadata for Buzsaki/%s — skipping", slug)

    logger.info("Buzsaki Lab: %d records", len(records))
    return records
