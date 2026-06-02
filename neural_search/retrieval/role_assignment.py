"""Dataset role assignment for multi-dataset result sets.

Assigns each result dataset exactly one role based on its relationship to
the query and the anchor dataset. A dataset with no assignable role is
excluded from the final demonstration result set.

Roles (in priority order):
  1. ANCHOR                   — highest usefulness, matches ≥3 sub-queries
  2. REPLICATION              — same task + species as anchor, different modality
  3. CROSS_SPECIES_COMPARATOR — same task, different species from anchor
  4. METHODOLOGICAL_COMPLEMENT — different region, shares ≥2 affordances
  5. PERTURBATION_CAUSAL      — has optogenetic/pharmacological manipulation
  6. BEHAVIOR_RICH            — rich trial-by-trial events, minimal neural
  7. POPULATION_DYNAMICS      — large cell count (≥100), dimensionality reduction
  8. IMAGING_EPHYS_BRIDGE     — both fMRI/EEG and electrophysiology
  9. UNASSIGNABLE             — no role matches; dataset excluded
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DatasetRole(StrEnum):
    ANCHOR = "anchor"
    REPLICATION = "replication"
    CROSS_SPECIES_COMPARATOR = "cross_species_comparator"
    METHODOLOGICAL_COMPLEMENT = "methodological_complement"
    PERTURBATION_CAUSAL = "perturbation_causal"
    BEHAVIOR_RICH = "behavior_rich"
    POPULATION_DYNAMICS = "population_dynamics"
    IMAGING_EPHYS_BRIDGE = "imaging_ephys_bridge"
    UNASSIGNABLE = "unassignable"


@dataclass
class RoleAssignment:
    dataset_id: str
    role: DatasetRole
    evidence: str = ""


def _labels(items: list, *, id_key: str = "label") -> set[str]:
    """Extract lowercase label strings from a list of dicts or strings."""
    result = set()
    for item in items:
        if isinstance(item, dict):
            v = item.get(id_key) or item.get("id") or item.get("affordance_id") or ""
        else:
            v = str(item)
        if v:
            result.add(v.lower())
    return result


def _find_anchor(datasets: list[dict]) -> dict | None:
    if not datasets:
        return None
    return max(
        datasets,
        key=lambda d: (d.get("sub_query_matches", 0), d.get("usefulness_score", 0.0)),
    )


_PERTURBATION_KEYWORDS = frozenset({
    "optogenetic", "opto", "chemogenetic", "dreadd", "pharmacological",
    "lesion", "inactivation", "muscimol", "silencing",
})

_EPHYS_MODALITIES = frozenset({
    "neuropixels", "extracellular_ephys", "ecog", "ieeg",
    "fiber_photometry", "patch_clamp",
})
_IMAGING_MODALITIES = frozenset({"fmri", "eeg", "meg"})


def assign_role(
    dataset: dict[str, Any],
    all_datasets: list[dict[str, Any]],
    anchor_id: str | None = None,
) -> RoleAssignment:
    """Assign a role to a single dataset in the context of a result set."""
    did = str(dataset.get("dataset_id") or dataset.get("source_id") or "unknown")
    mods = _labels(dataset.get("modalities", []))
    species = _labels(dataset.get("species", []))
    tasks = _labels(dataset.get("tasks", []))
    regions = _labels(dataset.get("brain_regions", []))
    affordances = _labels(dataset.get("affordances", []), id_key="affordance_id")
    sq_matches = dataset.get("sub_query_matches", 0)
    u_score = float(dataset.get("usefulness_score", 0.0))
    description = str(dataset.get("description") or "").lower()

    anchor = next(
        (d for d in all_datasets if str(d.get("dataset_id") or "") == anchor_id),
        None,
    )
    if anchor is None:
        anchor = _find_anchor(all_datasets)

    anchor_tasks = _labels(anchor.get("tasks", [])) if anchor else set()
    anchor_species = _labels(anchor.get("species", [])) if anchor else set()
    anchor_mods = _labels(anchor.get("modalities", [])) if anchor else set()
    anchor_affordances = _labels(anchor.get("affordances", []), id_key="affordance_id") if anchor else set()
    anchor_regions = _labels(anchor.get("brain_regions", [])) if anchor else set()
    anchor_did = str(anchor.get("dataset_id") or "") if anchor else ""

    # 1. ANCHOR
    if did == anchor_did or (sq_matches >= 3 and u_score >= 0.5):
        return RoleAssignment(did, DatasetRole.ANCHOR, f"matches {sq_matches} sub-queries, score={u_score:.2f}")

    # 2. REPLICATION: same task + species as anchor, different modality
    if tasks & anchor_tasks and species & anchor_species and mods and not (mods & anchor_mods):
        return RoleAssignment(
            did, DatasetRole.REPLICATION,
            f"same tasks/species as anchor, different modality {mods}",
        )

    # 3. CROSS-SPECIES COMPARATOR: same task, different species
    if tasks & anchor_tasks and species and not (species & anchor_species):
        return RoleAssignment(
            did, DatasetRole.CROSS_SPECIES_COMPARATOR,
            f"same tasks {tasks & anchor_tasks}, different species {species}",
        )

    # 4. METHODOLOGICAL COMPLEMENT: different region, shares ≥2 affordances
    shared_aff = affordances & anchor_affordances
    if len(shared_aff) >= 2 and regions and not (regions & anchor_regions):
        return RoleAssignment(
            did, DatasetRole.METHODOLOGICAL_COMPLEMENT,
            f"shares affordances {shared_aff}, different region {regions}",
        )

    # 5. PERTURBATION/CAUSAL: manipulation keyword in description or tasks
    if any(kw in description or kw in tasks for kw in _PERTURBATION_KEYWORDS):
        return RoleAssignment(did, DatasetRole.PERTURBATION_CAUSAL, "contains perturbation/causal keyword")

    # 6. BEHAVIOR-RICH: minimal neural, rich behavioral events
    has_behavior = dataset.get("has_behavior") or "behavior" in description
    if has_behavior and not (mods & (_EPHYS_MODALITIES | _IMAGING_MODALITIES)) and dataset.get("has_trials"):
        return RoleAssignment(did, DatasetRole.BEHAVIOR_RICH, "rich behavioral events, no neural recording")

    # 7. POPULATION DYNAMICS: large cell/subject count or dimensionality reduction
    pop_dyn_affs = {"population_dynamics", "dimensionality_reduction", "neural_manifold"}
    subject_count = int(dataset.get("subject_count") or dataset.get("session_count") or 0)
    if (pop_dyn_affs & affordances) or subject_count >= 100:
        return RoleAssignment(
            did, DatasetRole.POPULATION_DYNAMICS,
            f"population dynamics affordances or large count ({subject_count})",
        )

    # 8. IMAGING-EPHYS BRIDGE: has both fMRI/EEG and electrophysiology
    if mods & _IMAGING_MODALITIES and mods & _EPHYS_MODALITIES:
        return RoleAssignment(
            did, DatasetRole.IMAGING_EPHYS_BRIDGE,
            f"combined imaging {mods & _IMAGING_MODALITIES} + ephys {mods & _EPHYS_MODALITIES}",
        )

    return RoleAssignment(did, DatasetRole.UNASSIGNABLE, "no matching role criteria")
