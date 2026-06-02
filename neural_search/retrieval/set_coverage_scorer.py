"""Set-coverage scorer for multi-dataset result sets.

Scores a set of datasets D against a query goal, rewarding:
  - Individual usefulness quality (mean usefulness_score)
  - Modality, species, region diversity across the set
  - Complementary affordances (unique across set)
  - Provenance quality (DOI + license + complete metadata)

And penalizing:
  - Near-duplicate datasets (redundancy)
  - Missing required metadata fields
  - Hard-negative constraint violations

Formula:
    score(D) = mean_usefulness
             + α * coverage_bonus
             + β * complementarity_bonus
             + γ * provenance_bonus
             - δ * redundancy_penalty
             - ε * missing_metadata_penalty
             - ζ * hard_negative_penalty

Default weights: α=β=γ=0.1, δ=0.15, ε=0.05, ζ=0.30
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SetConstraints:
    """Query-level constraints for set-coverage scoring."""
    required_modalities: list[str] = field(default_factory=list)
    required_species: list[str] = field(default_factory=list)
    required_regions: list[str] = field(default_factory=list)
    required_affordances: list[str] = field(default_factory=list)
    hard_negative_modalities: list[str] = field(default_factory=list)
    hard_negative_species: list[str] = field(default_factory=list)


@dataclass
class SetCoverageResult:
    """Result of scoring a dataset set."""
    total_score: float
    mean_usefulness: float
    coverage_bonus: float
    complementarity_bonus: float
    provenance_bonus: float
    redundancy_penalty: float
    missing_metadata_penalty: float
    hard_negative_penalty: float
    hard_negative_violations: list[str] = field(default_factory=list)
    dataset_count: int = 0


class SetCoverageScorer:
    """Score a set of datasets for collective usefulness."""

    def __init__(
        self,
        alpha: float = 0.1,    # coverage bonus weight
        beta: float = 0.1,     # complementarity bonus weight
        gamma: float = 0.1,    # provenance bonus weight
        delta: float = 0.15,   # redundancy penalty weight
        epsilon: float = 0.05, # missing metadata penalty weight
        zeta: float = 0.30,    # hard-negative penalty weight (high — constraints matter)
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.zeta = zeta

    def score_set(
        self,
        datasets: list[dict[str, Any]],
        constraints: SetConstraints,
    ) -> SetCoverageResult:
        """Score a set of datasets collectively."""
        if not datasets:
            return SetCoverageResult(
                total_score=0.0, mean_usefulness=0.0,
                coverage_bonus=0.0, complementarity_bonus=0.0,
                provenance_bonus=0.0, redundancy_penalty=0.0,
                missing_metadata_penalty=0.0, hard_negative_penalty=0.0,
            )

        usefulness_scores = [float(d.get("usefulness_score", 0.0)) for d in datasets]
        mean_usefulness = sum(usefulness_scores) / len(usefulness_scores)

        def _labels(items: list) -> set[str]:
            result = set()
            for item in items:
                v = (item.get("label") if isinstance(item, dict) else item) or ""
                if v:
                    result.add(str(v).lower())
            return result

        all_modalities = {m for d in datasets for m in _labels(d.get("modalities", []))}
        all_species = {s for d in datasets for s in _labels(d.get("species", []))}

        coverage_bonus = 0.0
        n_dims = 0
        if constraints.required_modalities:
            req = {m.lower() for m in constraints.required_modalities}
            coverage_bonus += len(req & all_modalities) / len(req)
            n_dims += 1
        if constraints.required_species:
            req = {s.lower() for s in constraints.required_species}
            coverage_bonus += len(req & all_species) / len(req)
            n_dims += 1
        if not n_dims:
            coverage_bonus = (
                min(1.0, len(all_modalities) / 4.0) * 0.5
                + min(1.0, len(all_species) / 3.0) * 0.5
            )
        else:
            coverage_bonus /= n_dims

        all_affordances: list[set[str]] = []
        for d in datasets:
            affs: set[str] = set()
            for a in d.get("affordances", []):
                if isinstance(a, dict):
                    v = a.get("affordance_id") or a.get("label") or ""
                else:
                    v = str(a)
                if v:
                    affs.add(v.lower())
            all_affordances.append(affs)
        total_aff_union = set().union(*all_affordances) if all_affordances else set()
        unique_per = [
            len(affs - set().union(*(all_affordances[j] for j in range(len(all_affordances)) if j != i)))
            for i, affs in enumerate(all_affordances)
        ]
        complementarity_bonus = (
            min(1.0, sum(unique_per) / max(1, len(total_aff_union)))
            if total_aff_union else 0.0
        )

        n_with_id = sum(
            1 for d in datasets
            if d.get("doi") or d.get("source_id") or d.get("dataset_id")
        )
        provenance_bonus = n_with_id / len(datasets)

        signatures: dict[tuple, int] = {}
        for d in datasets:
            sig = tuple(sorted(
                list(_labels(d.get("modalities", []))) + list(_labels(d.get("species", [])))
            ))
            signatures[sig] = signatures.get(sig, 0) + 1
        redundant = sum(max(0, v - 1) for v in signatures.values())
        redundancy_penalty = min(1.0, redundant / len(datasets))

        n_no_modality = sum(1 for d in datasets if not d.get("modalities"))
        missing_metadata_penalty = n_no_modality / len(datasets)

        violations: list[str] = []
        for d in datasets:
            did = str(d.get("dataset_id") or d.get("source_id") or "?")
            mods = _labels(d.get("modalities", []))
            specs = _labels(d.get("species", []))
            violated = False
            for hn in constraints.hard_negative_modalities:
                if hn.lower() in mods:
                    violated = True
                    break
            if not violated:
                for hn in constraints.hard_negative_species:
                    if hn.lower() in specs:
                        violated = True
                        break
            if violated:
                violations.append(did)
        hard_negative_penalty = min(1.0, len(violations) / len(datasets))

        total = (
            mean_usefulness
            + self.alpha * coverage_bonus
            + self.beta * complementarity_bonus
            + self.gamma * provenance_bonus
            - self.delta * redundancy_penalty
            - self.epsilon * missing_metadata_penalty
            - self.zeta * hard_negative_penalty
        )

        return SetCoverageResult(
            total_score=max(0.0, min(1.0, total)),
            mean_usefulness=round(mean_usefulness, 4),
            coverage_bonus=round(coverage_bonus, 4),
            complementarity_bonus=round(complementarity_bonus, 4),
            provenance_bonus=round(provenance_bonus, 4),
            redundancy_penalty=round(redundancy_penalty, 4),
            missing_metadata_penalty=round(missing_metadata_penalty, 4),
            hard_negative_penalty=round(hard_negative_penalty, 4),
            hard_negative_violations=violations,
            dataset_count=len(datasets),
        )
