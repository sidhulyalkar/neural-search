"""Dataset compatibility scoring for scientific context pairing.

This module implements compatibility scoring between datasets based on their
experimental context slots. It goes beyond simple similarity to identify:
- Equivalent contexts (suitable for replication/meta-analysis)
- Complementary contexts (cross-modal, cross-species comparisons)
- Translational contexts (mapping across species/paradigms)
- Analysis-compatible contexts (shared analysis affordances)
- Contrastive contexts (controlled comparisons)

Mathematical formalization:
    Each dataset d is represented as a slot vector:
    E(d) = {organism, taxon_group, preparation, task_family, task_variant,
            event_schema, modality_family, device, signal_type, brain_region,
            region_hierarchy, species_homolog, data_standard, file_format,
            sampling_resolution, analysis_affordances, readiness, provenance}

    compatibility(d_i, d_j | intent) = sum_k w_k(intent) * match_k(E_k(d_i), E_k(d_j))
                                     + graph_path_score(d_i, d_j)
                                     + complementary_analysis_score(d_i, d_j)
                                     - contradiction_penalty(d_i, d_j)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from neural_search.ontology import normalize_text


class CompatibilityType(StrEnum):
    """Types of dataset compatibility relationships."""

    EQUIVALENT = "equivalent"  # Same experimental context, good for replication
    COMPLEMENTARY = "complementary"  # Same task/different modality or vice versa
    TRANSLATIONAL = "translational"  # Cross-species/paradigm mapping
    ANALYSIS_COMPATIBLE = "analysis_compatible"  # Shared analysis affordances
    CONTRASTIVE = "contrastive"  # Intentional controlled differences


@dataclass
class SlotMatch:
    """Result of matching a single context slot."""

    slot_name: str
    values_i: set[str]
    values_j: set[str]
    overlap: set[str]
    match_score: float  # 0-1, how well slots match
    match_type: str  # "exact", "partial", "hierarchical", "none"


@dataclass
class CompatibilityScore:
    """Detailed compatibility score between two datasets."""

    dataset_i_id: str
    dataset_j_id: str
    compatibility_type: CompatibilityType
    overall_score: float  # 0-1 combined score
    slot_matches: dict[str, SlotMatch]
    slot_contributions: dict[str, float]
    explanation: str
    supporting_evidence: list[str]
    contradictions: list[str]


@dataclass
class CompatibilityResult:
    """Complete compatibility analysis result."""

    query_dataset_id: str
    candidates: list[CompatibilityScore]
    best_equivalent: CompatibilityScore | None
    best_complementary: CompatibilityScore | None
    best_translational: CompatibilityScore | None
    best_analysis_compatible: CompatibilityScore | None


@dataclass
class CompatibilityConfig:
    """Configuration for compatibility scoring."""

    # Slot weights for different compatibility types
    equivalent_weights: dict[str, float] = field(default_factory=lambda: {
        "species": 0.20,
        "task": 0.20,
        "modality": 0.18,
        "brain_region": 0.12,
        "behavioral_event": 0.10,
        "data_standard": 0.08,
        "analysis_affordance": 0.07,
        "readiness": 0.05,
    })

    complementary_weights: dict[str, float] = field(default_factory=lambda: {
        "task": 0.30,  # Same task is important
        "species": 0.10,  # Can differ
        "modality": 0.05,  # Should differ
        "brain_region": 0.20,  # Same region preferred
        "behavioral_event": 0.15,
        "analysis_affordance": 0.15,
        "readiness": 0.05,
    })

    translational_weights: dict[str, float] = field(default_factory=lambda: {
        "task": 0.25,  # Similar task
        "species": 0.05,  # Different species expected
        "taxon_group": 0.15,  # Related taxonomic groups
        "modality": 0.15,
        "brain_region": 0.15,  # Homologous regions
        "behavioral_event": 0.15,
        "analysis_affordance": 0.10,
    })

    analysis_compatible_weights: dict[str, float] = field(default_factory=lambda: {
        "analysis_affordance": 0.40,
        "modality": 0.20,
        "behavioral_event": 0.20,
        "task": 0.10,
        "readiness": 0.10,
    })

    # Thresholds for classification
    equivalent_threshold: float = 0.75
    complementary_threshold: float = 0.60
    translational_threshold: float = 0.50
    analysis_compatible_threshold: float = 0.65

    # Penalty for contradictions
    contradiction_penalty: float = 0.3


# Species homology mappings for translational comparisons
SPECIES_HOMOLOGS: dict[str, set[str]] = {
    "mus_musculus": {"rattus_norvegicus", "homo_sapiens"},
    "rattus_norvegicus": {"mus_musculus", "homo_sapiens"},
    "homo_sapiens": {"macaca_mulatta", "mus_musculus", "rattus_norvegicus"},
    "macaca_mulatta": {"homo_sapiens", "macaca_fascicularis"},
    "macaca_fascicularis": {"homo_sapiens", "macaca_mulatta"},
    "danio_rerio": {"drosophila_melanogaster"},
    "drosophila_melanogaster": {"danio_rerio", "caenorhabditis_elegans"},
}

# Taxon group mappings
TAXON_GROUPS: dict[str, str] = {
    "mus_musculus": "rodent",
    "rattus_norvegicus": "rodent",
    "homo_sapiens": "primate",
    "macaca_mulatta": "primate",
    "macaca_fascicularis": "primate",
    "danio_rerio": "fish",
    "drosophila_melanogaster": "invertebrate",
    "caenorhabditis_elegans": "invertebrate",
}

# Modality family mappings
MODALITY_FAMILIES: dict[str, str] = {
    "neuropixels": "electrophysiology",
    "extracellular_ephys": "electrophysiology",
    "patch_clamp": "electrophysiology",
    "eeg": "electrophysiology",
    "ecog": "electrophysiology",
    "ieeg": "electrophysiology",
    "calcium_imaging": "optical",
    "two_photon": "optical",
    "fiber_photometry": "optical",
    "fmri": "imaging",
    "bold": "imaging",
    "meg": "magnetic",
    "single_cell_rnaseq": "transcriptomics",
    "single_nucleus_rnaseq": "transcriptomics",
    "spatial_transcriptomics": "transcriptomics",
    "behavior_video": "behavioral",
    "pose_tracking": "behavioral",
}


def _normalize_set(values: Any) -> set[str]:
    """Normalize a collection of values to a set of lowercase strings."""
    if not values:
        return set()
    if isinstance(values, str):
        return {normalize_text(values)}
    if isinstance(values, (list, tuple, set)):
        result = set()
        for v in values:
            if isinstance(v, str):
                result.add(normalize_text(v))
            elif isinstance(v, Mapping):
                # Handle dicts with 'id', 'label', or 'analysis_id' keys
                for key in ("id", "label", "analysis_id"):
                    if key in v and v[key]:
                        result.add(normalize_text(str(v[key])))
                        break
        return result
    return set()


def _get_dataset_slots(dataset: Mapping[str, Any]) -> dict[str, set[str]]:
    """Extract all context slots from a dataset record."""
    slots: dict[str, set[str]] = {}

    # Direct field mappings
    field_mapping = {
        "species": "species",
        "task": "tasks",
        "modality": "modalities",
        "brain_region": "brain_regions",
        "behavioral_event": "behaviors",
        "data_standard": "data_standards",
        "analysis_affordance": "analysis_affordances",
    }

    for slot_name, field_name in field_mapping.items():
        slots[slot_name] = _normalize_set(dataset.get(field_name, []))

    # Derived slots
    species_set = slots.get("species", set())
    slots["taxon_group"] = {
        TAXON_GROUPS.get(s, "unknown") for s in species_set
        if TAXON_GROUPS.get(s)
    }

    modality_set = slots.get("modality", set())
    slots["modality_family"] = {
        MODALITY_FAMILIES.get(m, "unknown") for m in modality_set
        if MODALITY_FAMILIES.get(m)
    }

    # Readiness score (convert to categorical)
    # Only set if explicitly provided or if dataset has meaningful content
    has_content = any(
        slots.get(k) for k in ("species", "task", "modality", "brain_region")
    )
    readiness = dataset.get("analysis_readiness_score")
    if readiness is not None and isinstance(readiness, (int, float)):
        if readiness >= 80:
            slots["readiness"] = {"high"}
        elif readiness >= 50:
            slots["readiness"] = {"medium"}
        else:
            slots["readiness"] = {"low"}
    elif has_content:
        # Only default to "unknown" if dataset has other content
        slots["readiness"] = {"unknown"}
    # Empty datasets get no readiness slot

    return slots


def _compute_slot_match(
    slot_name: str,
    values_i: set[str],
    values_j: set[str],
    use_homology: bool = False,
) -> SlotMatch:
    """Compute match score between two slot values."""
    if not values_i or not values_j:
        return SlotMatch(
            slot_name=slot_name,
            values_i=values_i,
            values_j=values_j,
            overlap=set(),
            match_score=0.0,
            match_type="none",
        )

    # Direct overlap
    overlap = values_i & values_j

    if overlap:
        # Jaccard-like score with overlap bonus
        union_size = len(values_i | values_j)
        score = len(overlap) / union_size if union_size > 0 else 0.0
        return SlotMatch(
            slot_name=slot_name,
            values_i=values_i,
            values_j=values_j,
            overlap=overlap,
            match_score=score,
            match_type="exact" if overlap == values_i == values_j else "partial",
        )

    # Check for homology/hierarchy if enabled
    if use_homology and slot_name == "species":
        for v_i in values_i:
            homologs = SPECIES_HOMOLOGS.get(v_i, set())
            if homologs & values_j:
                return SlotMatch(
                    slot_name=slot_name,
                    values_i=values_i,
                    values_j=values_j,
                    overlap=homologs & values_j,
                    match_score=0.5,  # Partial credit for homology
                    match_type="hierarchical",
                )

    return SlotMatch(
        slot_name=slot_name,
        values_i=values_i,
        values_j=values_j,
        overlap=set(),
        match_score=0.0,
        match_type="none",
    )


def _detect_complementary(
    slots_i: dict[str, set[str]],
    slots_j: dict[str, set[str]],
) -> tuple[bool, str]:
    """Detect if datasets are complementary (same task, different modality, etc.)."""
    task_match = _compute_slot_match("task", slots_i.get("task", set()), slots_j.get("task", set()))
    modality_match = _compute_slot_match("modality", slots_i.get("modality", set()), slots_j.get("modality", set()))
    region_match = _compute_slot_match("brain_region", slots_i.get("brain_region", set()), slots_j.get("brain_region", set()))
    species_match = _compute_slot_match("species", slots_i.get("species", set()), slots_j.get("species", set()))

    # Same task, different modality
    if task_match.match_score > 0.5 and modality_match.match_score < 0.3:
        return True, "Same task with different recording modalities (cross-modal)"

    # Same region, different species
    if region_match.match_score > 0.5 and species_match.match_score < 0.3:
        return True, "Same brain region in different species (comparative)"

    # Same modality family, different species
    family_match = _compute_slot_match(
        "modality_family",
        slots_i.get("modality_family", set()),
        slots_j.get("modality_family", set()),
    )
    if family_match.match_score > 0.5 and species_match.match_score < 0.3:
        return True, "Same modality family across species (translational)"

    return False, ""


def _detect_translational(
    slots_i: dict[str, set[str]],
    slots_j: dict[str, set[str]],
) -> tuple[bool, str]:
    """Detect if datasets have translational relationship."""
    species_i = slots_i.get("species", set())
    species_j = slots_j.get("species", set())

    # Check species homology
    for s_i in species_i:
        homologs = SPECIES_HOMOLOGS.get(s_i, set())
        if homologs & species_j:
            # Check if tasks or regions align
            task_match = _compute_slot_match("task", slots_i.get("task", set()), slots_j.get("task", set()))
            region_match = _compute_slot_match("brain_region", slots_i.get("brain_region", set()), slots_j.get("brain_region", set()))

            if task_match.match_score > 0.3 or region_match.match_score > 0.3:
                return True, f"Cross-species comparison ({s_i} <-> {homologs & species_j})"

    return False, ""


def _detect_contrastive(
    slots_i: dict[str, set[str]],
    slots_j: dict[str, set[str]],
) -> tuple[bool, str]:
    """Detect if datasets form a controlled contrast."""
    # Count matching vs differing dimensions
    core_slots = ["species", "task", "modality", "brain_region"]
    matches = 0
    differences = 0
    diff_slot = ""

    for slot in core_slots:
        match = _compute_slot_match(slot, slots_i.get(slot, set()), slots_j.get(slot, set()))
        if match.match_score > 0.5:
            matches += 1
        elif slots_i.get(slot) and slots_j.get(slot):
            differences += 1
            diff_slot = slot

    # Contrastive: mostly same except one controlled difference
    if matches >= 3 and differences == 1:
        return True, f"Controlled contrast varying {diff_slot}"

    return False, ""


class CompatibilityScorer:
    """Score compatibility between datasets for scientific context pairing.

    Example:
        >>> scorer = CompatibilityScorer()
        >>> result = scorer.score_compatibility(dataset_i, dataset_j)
        >>> print(result.compatibility_type, result.overall_score)
    """

    def __init__(self, config: CompatibilityConfig | None = None):
        self.config = config or CompatibilityConfig()

    def score_compatibility(
        self,
        dataset_i: Mapping[str, Any],
        dataset_j: Mapping[str, Any],
    ) -> CompatibilityScore:
        """Compute detailed compatibility score between two datasets.

        Args:
            dataset_i: First dataset record.
            dataset_j: Second dataset record.

        Returns:
            CompatibilityScore with detailed breakdown.
        """
        # Get dataset IDs
        id_i = str(dataset_i.get("dataset_id", dataset_i.get("id", dataset_i.get("source_id", "unknown_i"))))
        id_j = str(dataset_j.get("dataset_id", dataset_j.get("id", dataset_j.get("source_id", "unknown_j"))))

        # Extract context slots
        slots_i = _get_dataset_slots(dataset_i)
        slots_j = _get_dataset_slots(dataset_j)

        # Compute slot matches
        slot_matches: dict[str, SlotMatch] = {}
        for slot_name in set(slots_i.keys()) | set(slots_j.keys()):
            use_homology = slot_name == "species"
            slot_matches[slot_name] = _compute_slot_match(
                slot_name,
                slots_i.get(slot_name, set()),
                slots_j.get(slot_name, set()),
                use_homology=use_homology,
            )

        # Determine compatibility type and compute score
        is_complementary, comp_reason = _detect_complementary(slots_i, slots_j)
        is_translational, trans_reason = _detect_translational(slots_i, slots_j)
        is_contrastive, contrast_reason = _detect_contrastive(slots_i, slots_j)

        # Compute scores for each type
        scores_by_type: dict[CompatibilityType, tuple[float, dict[str, float]]] = {}

        for compat_type, weights in [
            (CompatibilityType.EQUIVALENT, self.config.equivalent_weights),
            (CompatibilityType.COMPLEMENTARY, self.config.complementary_weights),
            (CompatibilityType.TRANSLATIONAL, self.config.translational_weights),
            (CompatibilityType.ANALYSIS_COMPATIBLE, self.config.analysis_compatible_weights),
        ]:
            score, contributions = self._compute_weighted_score(slot_matches, weights)
            scores_by_type[compat_type] = (score, contributions)

        # Determine best compatibility type
        best_type = CompatibilityType.EQUIVALENT
        explanation_parts: list[str] = []

        # Start with equivalent score as baseline (always compute)
        equiv_score, equiv_contrib = scores_by_type[CompatibilityType.EQUIVALENT]
        best_score = equiv_score
        best_contributions = equiv_contrib

        # Check each type against thresholds
        if equiv_score >= self.config.equivalent_threshold:
            explanation_parts.append("High overlap across experimental dimensions")

        comp_score, comp_contrib = scores_by_type[CompatibilityType.COMPLEMENTARY]
        if is_complementary and comp_score >= self.config.complementary_threshold:
            if comp_score > best_score or best_type == CompatibilityType.EQUIVALENT:
                best_type = CompatibilityType.COMPLEMENTARY
                best_score = comp_score
                best_contributions = comp_contrib
                explanation_parts = [comp_reason]

        trans_score, trans_contrib = scores_by_type[CompatibilityType.TRANSLATIONAL]
        if is_translational and trans_score >= self.config.translational_threshold:
            if trans_score > best_score:
                best_type = CompatibilityType.TRANSLATIONAL
                best_score = trans_score
                best_contributions = trans_contrib
                explanation_parts = [trans_reason]

        anal_score, anal_contrib = scores_by_type[CompatibilityType.ANALYSIS_COMPATIBLE]
        if anal_score >= self.config.analysis_compatible_threshold:
            if best_score < self.config.equivalent_threshold:
                # Use analysis compatible if not strongly equivalent
                best_type = CompatibilityType.ANALYSIS_COMPATIBLE
                best_score = anal_score
                best_contributions = anal_contrib
                explanation_parts.append("Shared analysis affordances")

        if is_contrastive:
            best_type = CompatibilityType.CONTRASTIVE
            explanation_parts.append(contrast_reason)

        # Build supporting evidence
        evidence: list[str] = []
        for slot_name, match in slot_matches.items():
            if match.match_score > 0.5:
                evidence.append(f"{slot_name}: {', '.join(match.overlap)}")

        # Detect contradictions (only for unexpected differences)
        # Don't penalize expected differences based on compatibility type
        contradictions: list[str] = []
        expected_differences: set[str] = set()

        if best_type == CompatibilityType.COMPLEMENTARY or is_complementary:
            # Cross-modal comparisons expect modality differences
            expected_differences.add("modality")
        if best_type == CompatibilityType.TRANSLATIONAL or is_translational:
            # Translational comparisons expect species differences
            expected_differences.add("species")
        if best_type == CompatibilityType.CONTRASTIVE or is_contrastive:
            # Contrastive comparisons expect one controlled difference
            expected_differences.update(["species", "modality", "task", "brain_region"])

        for slot_name, match in slot_matches.items():
            if match.values_i and match.values_j and match.match_score == 0:
                if slot_name in ("species", "modality", "task"):
                    # Only count as contradiction if not an expected difference
                    if slot_name not in expected_differences:
                        contradictions.append(
                            f"No {slot_name} overlap: {match.values_i} vs {match.values_j}"
                        )

        # Apply contradiction penalty (reduced for partial contradictions)
        if contradictions:
            penalty = self.config.contradiction_penalty * len(contradictions) * 0.5
            best_score = max(0.0, best_score - penalty)

        return CompatibilityScore(
            dataset_i_id=id_i,
            dataset_j_id=id_j,
            compatibility_type=best_type,
            overall_score=round(best_score, 4),
            slot_matches=slot_matches,
            slot_contributions=best_contributions,
            explanation=" | ".join(explanation_parts) if explanation_parts else "Limited compatibility",
            supporting_evidence=evidence,
            contradictions=contradictions,
        )

    def _compute_weighted_score(
        self,
        slot_matches: dict[str, SlotMatch],
        weights: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Compute weighted compatibility score from slot matches."""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0, {}

        score = 0.0
        contributions: dict[str, float] = {}

        for slot_name, weight in weights.items():
            if slot_name in slot_matches:
                match = slot_matches[slot_name]
                contribution = (weight / total_weight) * match.match_score
                score += contribution
                contributions[slot_name] = round(contribution, 4)

        return round(score, 4), contributions

    def find_compatible_datasets(
        self,
        query_dataset: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        compatibility_types: list[CompatibilityType] | None = None,
        min_score: float = 0.3,
        top_k: int = 10,
    ) -> CompatibilityResult:
        """Find compatible datasets from a candidate pool.

        Args:
            query_dataset: The dataset to find compatibles for.
            candidates: Pool of candidate datasets.
            compatibility_types: Filter to specific types (None = all).
            min_score: Minimum compatibility score.
            top_k: Maximum results per type.

        Returns:
            CompatibilityResult with categorized matches.
        """
        query_id = str(query_dataset.get("dataset_id", query_dataset.get("id", "query")))

        all_scores: list[CompatibilityScore] = []

        for candidate in candidates:
            candidate_id = str(candidate.get("dataset_id", candidate.get("id", "")))
            if candidate_id == query_id:
                continue  # Skip self-comparison

            score = self.score_compatibility(query_dataset, candidate)

            if score.overall_score >= min_score:
                if compatibility_types is None or score.compatibility_type in compatibility_types:
                    all_scores.append(score)

        # Sort by score
        all_scores.sort(key=lambda s: -s.overall_score)

        # Find best of each type
        best_by_type: dict[CompatibilityType, CompatibilityScore | None] = dict.fromkeys(CompatibilityType)

        for score in all_scores:
            if best_by_type[score.compatibility_type] is None:
                best_by_type[score.compatibility_type] = score

        return CompatibilityResult(
            query_dataset_id=query_id,
            candidates=all_scores[:top_k],
            best_equivalent=best_by_type[CompatibilityType.EQUIVALENT],
            best_complementary=best_by_type[CompatibilityType.COMPLEMENTARY],
            best_translational=best_by_type[CompatibilityType.TRANSLATIONAL],
            best_analysis_compatible=best_by_type[CompatibilityType.ANALYSIS_COMPATIBLE],
        )


def compute_compatibility(
    dataset_i: Mapping[str, Any],
    dataset_j: Mapping[str, Any],
    config: CompatibilityConfig | None = None,
) -> CompatibilityScore:
    """Convenience function to compute compatibility between two datasets.

    Args:
        dataset_i: First dataset.
        dataset_j: Second dataset.
        config: Optional custom configuration.

    Returns:
        CompatibilityScore with detailed analysis.
    """
    scorer = CompatibilityScorer(config)
    return scorer.score_compatibility(dataset_i, dataset_j)


def explain_compatibility(score: CompatibilityScore) -> dict[str, Any]:
    """Generate detailed explanation of compatibility score.

    Args:
        score: Compatibility score to explain.

    Returns:
        Dictionary with human-readable explanation.
    """
    slot_details: list[dict[str, Any]] = []

    for slot_name, match in score.slot_matches.items():
        if match.values_i or match.values_j:
            slot_details.append({
                "slot": slot_name,
                "dataset_i_values": sorted(match.values_i),
                "dataset_j_values": sorted(match.values_j),
                "overlap": sorted(match.overlap),
                "match_score": round(match.match_score, 3),
                "match_type": match.match_type,
            })

    return {
        "datasets": [score.dataset_i_id, score.dataset_j_id],
        "compatibility_type": score.compatibility_type.value,
        "overall_score": score.overall_score,
        "explanation": score.explanation,
        "slot_details": sorted(slot_details, key=lambda x: -x["match_score"]),
        "slot_contributions": score.slot_contributions,
        "supporting_evidence": score.supporting_evidence,
        "contradictions": score.contradictions,
    }
