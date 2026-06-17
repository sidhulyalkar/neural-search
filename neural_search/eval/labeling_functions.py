"""13 deterministic labeling functions for weak-supervision qrels generation.

Each LF receives a PairEvidence and returns an LFVote.
LFs abstain (abstain=True) when they have no signal for the pair.
Hard-negative LF dominates: label=0, confidence=0.95 when a known
failure mode is matched.
"""
from __future__ import annotations

from neural_search.eval.evidence import LFVote, PairEvidence

_OPEN_LICENSES = {
    "cc0", "cc-by", "cc by", "cc-by-4.0", "cc by 4.0",
    "pddl", "odc-by", "cc-by-sa", "open data commons",
    "mit", "apache", "bsd", "public domain",
}

_RESTRICTIVE_TERMS = {
    "all rights reserved", "proprietary", "not for redistribution",
    "restricted", "not open",
}

_STOP_WORDS = {
    "with", "that", "from", "this", "when", "have", "been",
    "will", "their", "they", "some", "into", "than", "then",
    "data", "dataset",
}

_PIPELINE_STANDARDS = {"NWB", "nwb", "BIDS", "bids", "NeuroData Without Borders"}

_META_ANALYSIS_MODALITIES = {"fmri", "eeg", "meg"}


def _candidate_text(pair: PairEvidence) -> str:
    d = pair.dataset
    parts = [
        d.title or "",
        d.description or "",
        " ".join(d.species),
        " ".join(d.modalities),
        " ".join(d.tasks),
        " ".join(d.regions),
    ]
    return " ".join(parts).lower()


def _hn_terms(hn_phrase: str) -> list[str]:
    normalized = hn_phrase.lower().replace("-", " ")
    return [
        t for t in normalized.split()
        if len(t) > 3 and t not in _STOP_WORDS
    ]


def lf_hard_negative(pair: PairEvidence) -> LFVote:
    """Vote label=0 at high confidence when candidate matches a known failure mode."""
    if not pair.query.hard_negatives:
        return LFVote("lf_hard_negative", 0, 0.0, "No hard negatives defined.", abstain=True)

    candidate = _candidate_text(pair)
    for hn in pair.query.hard_negatives:
        terms = _hn_terms(hn)
        if not terms:
            continue
        matches = sum(1 for t in terms if t in candidate)
        if matches / len(terms) >= 0.5:
            return LFVote(
                "lf_hard_negative", 0, 0.95,
                f"Matches hard negative: '{hn}' ({matches}/{len(terms)} terms)"
            )
    return LFVote("lf_hard_negative", 0, 0.0, "No hard negative matched.", abstain=True)


def lf_required_modality(pair: PairEvidence) -> LFVote:
    required = set(pair.query.required_modalities)
    if not required:
        return LFVote("lf_required_modality", 0, 0.0, "No required modalities.", abstain=True)

    dataset_mods = set(pair.dataset.modalities)
    matched = required & dataset_mods
    n_req, n_matched = len(required), len(matched)

    if n_matched == n_req:
        return LFVote("lf_required_modality", 3, 0.90, f"All required modalities present: {matched}")
    if n_matched > 0:
        return LFVote("lf_required_modality", 2, 0.70,
                      f"Partial modality match {n_matched}/{n_req}: {matched}")
    return LFVote("lf_required_modality", 0, 0.85,
                  f"Required modalities {required} absent from dataset {dataset_mods}")


def lf_partial_modality(pair: PairEvidence) -> LFVote:
    preferred = set(pair.query.preferred_modalities)
    if not preferred:
        return LFVote("lf_partial_modality", 0, 0.0, "No preferred modalities.", abstain=True)

    dataset_mods = set(pair.dataset.modalities)
    matched = preferred & dataset_mods
    if matched:
        return LFVote("lf_partial_modality", 1, 0.55, f"Preferred modalities present: {matched}")
    return LFVote("lf_partial_modality", 0, 0.40, "No preferred modalities matched.", abstain=True)


def lf_species_constraint(pair: PairEvidence) -> LFVote:
    required = set(pair.query.required_species)
    if not required:
        return LFVote("lf_species_constraint", 0, 0.0, "No species constraint.", abstain=True)

    dataset_sp = set(pair.dataset.species)
    if required & dataset_sp:
        return LFVote("lf_species_constraint", 3, 0.90, f"Species match: {required & dataset_sp}")
    return LFVote("lf_species_constraint", 0, 0.85,
                  f"Species mismatch: required {required}, found {dataset_sp}")


def lf_task_constraint(pair: PairEvidence) -> LFVote:
    constraints = set(pair.query.task_constraints)
    if not constraints:
        return LFVote("lf_task_constraint", 0, 0.0, "No task constraints.", abstain=True)

    dataset_tasks = set(pair.dataset.tasks)
    matched = constraints & dataset_tasks
    if matched:
        return LFVote("lf_task_constraint", 3, 0.80, f"Task match: {matched}")
    return LFVote("lf_task_constraint", 1, 0.55, "No task constraint matched.")


def lf_region_constraint(pair: PairEvidence) -> LFVote:
    required = set(pair.query.brain_regions)
    if not required:
        return LFVote("lf_region_constraint", 0, 0.0, "No region constraints.", abstain=True)

    dataset_regions = set(pair.dataset.regions)
    if required & dataset_regions:
        return LFVote("lf_region_constraint", 3, 0.80,
                      f"Region match: {required & dataset_regions}")
    return LFVote("lf_region_constraint", 1, 0.55,
                  f"Region mismatch: required {required}, found {dataset_regions}")


def lf_data_level_required(pair: PairEvidence) -> LFVote:
    required = set(pair.query.data_level_requirements)
    if not required:
        return LFVote("lf_data_level_required", 0, 0.0, "No data level requirement.", abstain=True)

    dataset_levels = set(pair.dataset.data_levels)
    if required & dataset_levels:
        return LFVote("lf_data_level_required", 3, 0.80, f"Data level match: {required & dataset_levels}")
    return LFVote("lf_data_level_required", 0, 0.75,
                  f"Required data levels {required} not present; found {dataset_levels}")


def lf_raw_data_available(pair: PairEvidence) -> LFVote:
    if pair.dataset.raw_data_available:
        return LFVote("lf_raw_data_available", 3, 0.70, "Raw data available.")
    return LFVote("lf_raw_data_available", 2, 0.55, "Raw data not available — processed only.")


def lf_license_reusable(pair: PairEvidence) -> LFVote:
    lic = pair.dataset.license
    if not lic:
        return LFVote("lf_license_reusable", 0, 0.0, "License unknown.", abstain=True)

    lic_lower = lic.lower()
    if any(term in lic_lower for term in _OPEN_LICENSES):
        return LFVote("lf_license_reusable", 3, 0.85, f"Open license: {lic}")
    if any(term in lic_lower for term in _RESTRICTIVE_TERMS):
        return LFVote("lf_license_reusable", 0, 0.80, f"Restrictive license: {lic}")
    return LFVote("lf_license_reusable", 1, 0.50, f"Unknown license reusability: {lic}")


def lf_metadata_completeness(pair: PairEvidence) -> LFVote:
    score = pair.dataset.metadata_completeness
    if score >= 0.8:
        return LFVote("lf_metadata_completeness", 3, 0.70, f"High metadata completeness: {score:.2f}")
    if score >= 0.5:
        return LFVote("lf_metadata_completeness", 2, 0.55, f"Moderate completeness: {score:.2f}")
    if score >= 0.3:
        return LFVote("lf_metadata_completeness", 1, 0.50, f"Low completeness: {score:.2f}")
    return LFVote("lf_metadata_completeness", 0, 0.40, f"Very low completeness: {score:.2f}", abstain=True)


def lf_analysis_affordance(pair: PairEvidence) -> LFVote:
    affordances = set(str(a).lower() for a in pair.query.analysis_affordances)
    if not affordances:
        return LFVote("lf_analysis_affordance", 0, 0.0, "No affordances specified.", abstain=True)

    candidate = _candidate_text(pair)
    matched = [a for a in affordances if a in candidate]
    if matched:
        return LFVote("lf_analysis_affordance", 2, 0.60, f"Affordance signals present: {matched}")
    return LFVote("lf_analysis_affordance", 1, 0.40, "No affordance signals detected.", abstain=True)


def lf_pipeline_reuse(pair: PairEvidence) -> LFVote:
    """For PIPELINE_REUSE intent: reward standardized formats (NWB/BIDS)."""
    if pair.query.intent != "PIPELINE_REUSE":
        return LFVote("lf_pipeline_reuse", 0, 0.0, "Not a pipeline-reuse query.", abstain=True)

    standards = set(pair.dataset.data_standards)
    if standards & _PIPELINE_STANDARDS:
        return LFVote("lf_pipeline_reuse", 3, 0.80,
                      f"Standardized format present: {standards & _PIPELINE_STANDARDS}")
    return LFVote("lf_pipeline_reuse", 1, 0.55, "No standardized data format detected.")


def lf_meta_analysis_depth(pair: PairEvidence) -> LFVote:
    """For META_ANALYSIS intent: reward behavioral metadata + large-n + NWB."""
    if pair.query.intent != "META_ANALYSIS":
        return LFVote("lf_meta_analysis_depth", 0, 0.0, "Not a meta-analysis query.", abstain=True)

    score = 0
    rationale: list[str] = []
    if pair.dataset.has_behavior:
        score += 1
        rationale.append("has_behavior")
    if pair.dataset.has_trials:
        score += 1
        rationale.append("has_trials")
    if set(pair.dataset.data_standards) & _PIPELINE_STANDARDS:
        score += 1
        rationale.append("standardized_format")
    if pair.dataset.modalities and pair.dataset.modalities[0] in _META_ANALYSIS_MODALITIES:
        score += 1
        rationale.append("imaging_modality")

    label = min(score, 3)
    conf = 0.50 + 0.10 * score
    return LFVote("lf_meta_analysis_depth", label, min(conf, 0.85),
                  f"Meta-analysis signals: {', '.join(rationale) or 'none'}")


_ALL_LFS = [
    lf_hard_negative,
    lf_required_modality,
    lf_partial_modality,
    lf_species_constraint,
    lf_task_constraint,
    lf_region_constraint,
    lf_data_level_required,
    lf_raw_data_available,
    lf_license_reusable,
    lf_metadata_completeness,
    lf_analysis_affordance,
    lf_pipeline_reuse,
    lf_meta_analysis_depth,
]


def run_all_lfs(pair: PairEvidence) -> list[LFVote]:
    """Run all 13 labeling functions and return their votes."""
    return [lf(pair) for lf in _ALL_LFS]
