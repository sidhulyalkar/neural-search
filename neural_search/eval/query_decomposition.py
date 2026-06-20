"""Parse benchmark query records into structured QuerySpec objects.

Uses keyword matching against modality/species ontology terms.
Does not call any external service — pure deterministic parsing.
"""
from __future__ import annotations

import json
from pathlib import Path

from neural_search.eval.evidence import QuerySpec

_MODALITY_KW: dict[str, str] = {
    "fmri": "fmri",
    "functional mri": "fmri",
    "functional magnetic": "fmri",
    "bold": "fmri",
    "calcium imaging": "calcium_imaging",
    "two-photon": "calcium_imaging",
    "2-photon": "calcium_imaging",
    "gcamp": "calcium_imaging",
    "neuropixels": "neuropixels",
    "extracellular electrophysiology": "extracellular_ephys",
    "extracellular ephys": "extracellular_ephys",
    "electrophysiology": "extracellular_ephys",
    "single unit": "extracellular_ephys",
    "multi-unit": "extracellular_ephys",
    "spike sorting": "extracellular_ephys",
    "spike-sorting": "extracellular_ephys",
    "eeg": "eeg",
    "electroencephalography": "eeg",
    "meg": "meg",
    "magnetoencephalography": "meg",
    "patch clamp": "intracellular_ephys",
    "intracellular": "intracellular_ephys",
    "dti": "dti",
    "diffusion mri": "dti",
    "diffusion tensor": "dti",
    "lfp": "extracellular_ephys",
    "local field potential": "extracellular_ephys",
}

_SPECIES_KW: dict[str, str] = {
    "human": "human",
    "humans": "human",
    "patient": "human",
    "patients": "human",
    "homo sapiens": "human",
    "mouse": "mouse",
    "mice": "mouse",
    "murine": "mouse",
    "mus musculus": "mouse",
    "rat": "rat",
    "rats": "rat",
    "rodent": "rodent",
    "rodents": "rodent",
    "macaque": "macaque",
    "macaques": "macaque",
    "rhesus": "macaque",
    "primate": "macaque",
    "non-human primate": "macaque",
    "nhp": "macaque",
    "marmoset": "marmoset",
    "zebrafish": "zebrafish",
    "drosophila": "drosophila",
    "fly": "drosophila",
}

_DATA_LEVEL_KW: dict[str, str] = {
    "raw data": "raw",
    "raw recordings": "raw",
    "preprocessed": "preprocessed",
    "processed": "processed",
    "spike times": "processed",
    "spike-sorted": "processed",
}


def _extract_keywords(text: str, kw_map: dict[str, str]) -> list[str]:
    """Return canonical ids for all keywords found in text (no duplicates, order preserved)."""
    text_lower = text.lower()
    seen: set[str] = set()
    results: list[str] = []
    for kw in sorted(kw_map, key=len, reverse=True):
        if kw in text_lower:
            canonical = kw_map[kw]
            if canonical not in seen:
                seen.add(canonical)
                results.append(canonical)
    return results


def decompose_query(record: dict) -> QuerySpec:
    """Parse a raw benchmark query dict into a structured QuerySpec."""
    query_text = record.get("query", "")
    nice_to_have: list[str] = record.get("nice_to_have", []) or []
    nice_text = " ".join(str(n) for n in nice_to_have)

    return QuerySpec(
        query_id=record["query_id"],
        query_text=query_text,
        intent=record.get("intent", ""),
        scientific_goal=record.get("scientific_goal", ""),
        required_modalities=_extract_keywords(query_text, _MODALITY_KW),
        preferred_modalities=_extract_keywords(nice_text, _MODALITY_KW),
        required_species=_extract_keywords(query_text, _SPECIES_KW),
        preferred_species=_extract_keywords(nice_text, _SPECIES_KW),
        brain_regions=[],
        task_constraints=[],
        data_level_requirements=_extract_keywords(query_text, _DATA_LEVEL_KW),
        hard_negatives=list(record.get("known_failure_modes", []) or []),
        analysis_affordances=nice_to_have,
    )


def load_query_specs(queries_path: Path) -> list[QuerySpec]:
    """Load all QuerySpecs from a JSONL benchmark queries file."""
    specs: list[QuerySpec] = []
    with queries_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            specs.append(decompose_query(json.loads(line)))
    return specs
