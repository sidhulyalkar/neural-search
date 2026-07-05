"""Maps analysis_family/method requirements to concrete file-validation checks.

Deliberately modest: only covers the requirement checks actually exercised by
real validation runs so far, extended as new analysis_families get validated
against real data rather than speculatively covering all 27 up front (same
reasoning as `data/methods/method_registry.yaml`'s partial coverage).
"""

from __future__ import annotations

from neural_search.graph.dandi_nwb_validator import DandiAssetValidation
from neural_search.graph.openneuro_bids_validator import OpenNeuroValidation


def dandi_confirms_requirement(analysis_family: str, result: DandiAssetValidation) -> bool:
    """Whether a DANDI NWB header inspection confirms this analysis family's
    core requirement. Conservative: only confirms, never denies (a False
    return means "not confirmed by this check", not "refuted")."""

    if result.error:
        return False
    checks = {
        "spike_train_analysis": lambda r: r.has_units and (r.n_units or 0) > 0,
        "population_dynamics": lambda r: (r.has_units and (r.n_units or 0) >= 5) or r.has_imaging,
        "time_frequency": lambda r: r.has_electrodes and (r.n_electrodes or 0) > 0,
        "connectivity": lambda r: r.has_electrodes and (r.n_electrodes or 0) > 1,
        "event_aligned_analysis": lambda r: r.has_trials and (r.n_trials or 0) > 0,
        "decoding": lambda r: r.has_trials and (r.n_trials or 0) > 0,
        "encoding_modeling": lambda r: r.has_electrodes or r.has_imaging,
    }
    check = checks.get(analysis_family)
    return bool(check and check(result))


def openneuro_confirms_requirement(analysis_family: str, result: OpenNeuroValidation) -> bool:
    """Whether an OpenNeuro BIDS summary confirms this analysis family's core
    requirement. Conservative in the same sense as dandi_confirms_requirement."""

    if result.error:
        return False
    modalities = {m.lower() for m in result.modalities}
    checks = {
        "time_frequency": lambda r: bool({"eeg", "meg"} & modalities),
        "connectivity": lambda r: bool({"eeg", "meg", "mri"} & modalities),
        "clinical_prediction": lambda r: bool(modalities) and r.n_subjects > 0,
        "event_aligned_analysis": lambda r: len(r.tasks) > 0,
        "decoding": lambda r: len(r.tasks) > 0,
        "encoding_modeling": lambda r: "mri" in modalities,
    }
    check = checks.get(analysis_family)
    return bool(check and check(result))
