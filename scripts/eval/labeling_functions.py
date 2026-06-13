"""Rule-based labeling functions for silver qrels generation.

Each labeling function takes a benchmark query and a corpus dataset record
and returns a LabelingFunctionVote (vote 0–3, or None to abstain).

Design principles:
- Hard-negative violations push relevance to 0 and dominate other signals.
- Missing metadata lowers confidence and raises needs_human_review, but does
  NOT automatically mean irrelevant.
- Exact metadata matches do not automatically produce relevance=3; scientific
  goal and analysis affordance must also align.
- Each function is deterministic given the same inputs.
"""

from __future__ import annotations

import re
from typing import Any

from scripts.eval.benchmark_schema import BenchmarkQueryV1
from scripts.eval.silver_qrels_schema import LabelingFunctionVote

# ---------------------------------------------------------------------------
# Synonym maps for normalised matching
# ---------------------------------------------------------------------------

_SPECIES_SYNONYMS: dict[str, frozenset[str]] = {
    "human": frozenset(
        {"human", "homo sapiens", "h. sapiens", "homo_sapiens", "person", "people", "participant"}
    ),
    "mouse": frozenset(
        {"mouse", "mice", "mus musculus", "m. musculus", "mus_musculus", "murine"}
    ),
    "rat": frozenset(
        {"rat", "rattus norvegicus", "r. norvegicus", "rattus", "rodent"}
    ),
    "monkey": frozenset(
        {
            "monkey", "macaque", "macaca mulatta", "macaca fascicularis",
            "non-human primate", "non human primate", "nhp", "primate",
            "marmoset", "rhesus",
        }
    ),
    "zebrafish": frozenset({"zebrafish", "danio rerio", "zebrafish larvae"}),
    "drosophila": frozenset({"drosophila", "drosophila melanogaster", "fly", "flies"}),
    "ferret": frozenset({"ferret", "mustela putorius furo"}),
}

_MODALITY_SYNONYMS: dict[str, frozenset[str]] = {
    "fmri": frozenset(
        {
            "fmri", "functional mri", "bold", "bold fmri",
            "functional magnetic resonance imaging", "functional_mri",
        }
    ),
    "mri": frozenset(
        {"mri", "magnetic resonance imaging", "structural mri", "t1w", "t2w", "anatomical"}
    ),
    "eeg": frozenset({"eeg", "electroencephalography", "electroencephalogram"}),
    "ecog": frozenset({"ecog", "electrocorticography"}),
    "ephys": frozenset(
        {
            "ephys", "electrophysiology", "extracellular electrophysiology",
            "extracellular_ephys", "neuropixels", "single unit", "single-unit",
            "multi-unit", "mua", "sua", "spike", "spikes", "tetrode", "silicon probe",
        }
    ),
    "calcium_imaging": frozenset(
        {
            "calcium imaging", "calcium_imaging", "two-photon", "2-photon",
            "two photon", "2p imaging", "gcamp", "fluorescence imaging",
            "two_photon", "2p",
        }
    ),
    "fiber_photometry": frozenset({"fiber photometry", "fiberphot", "photometry", "fibre photometry"}),
    "behavior": frozenset({"behavior", "behaviour", "behavioral", "behavioural"}),
}

_BRAIN_REGION_SYNONYMS: dict[str, frozenset[str]] = {
    "hippocampus": frozenset({"hippocampus", "hippocampal", "hpc", "ca1", "ca3", "dentate gyrus"}),
    "prefrontal_cortex": frozenset(
        {"prefrontal cortex", "pfc", "dlpfc", "vlpfc", "orbitofrontal", "ofc", "medial pfc"}
    ),
    "visual_cortex": frozenset(
        {
            "visual cortex", "v1", "v2", "v4", "it", "inferotemporal", "mt", "v5",
            "primary visual cortex", "secondary visual cortex",
        }
    ),
    "motor_cortex": frozenset(
        {"motor cortex", "m1", "primary motor cortex", "premotor cortex", "sma"}
    ),
    "striatum": frozenset(
        {"striatum", "striatal", "caudate", "putamen", "nucleus accumbens", "basal ganglia"}
    ),
    "amygdala": frozenset({"amygdala", "amygdaloid"}),
    "cerebellum": frozenset({"cerebellum", "cerebellar"}),
    "thalamus": frozenset({"thalamus", "thalamic"}),
    "lip": frozenset({"lip", "lateral intraparietal", "intraparietal sulcus", "ips"}),
}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _norm(s: str) -> str:
    """Lowercase, collapse punctuation/underscores to spaces, strip."""
    return re.sub(r"[_\-]", " ", s.lower()).strip()


def _contains_any(text: str, terms: frozenset[str]) -> bool:
    """Return True if the normalised text contains any term as a substring."""
    nt = _norm(text)
    return any(t in nt for t in terms)


def _resolve_synonyms(raw_list: list[str], syn_map: dict[str, frozenset[str]]) -> set[str]:
    """Map a list of raw labels to canonical synonym-key names."""
    resolved: set[str] = set()
    for raw in raw_list:
        nr = _norm(raw)
        for key, synonyms in syn_map.items():
            if nr in synonyms or any(s in nr for s in synonyms):
                resolved.add(key)
                break
        else:
            resolved.add(nr)  # keep unrecognised labels normalised
    return resolved


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union)


def _list_field(record: dict[str, Any], *keys: str) -> list[str]:
    """Return the first non-empty list field found under any of the given keys."""
    for key in keys:
        val = record.get(key)
        if isinstance(val, list) and val:
            return [str(v) for v in val]
    return []


# ---------------------------------------------------------------------------
# Labeling functions
# ---------------------------------------------------------------------------


def lf_species_match(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote +1 when query expected species appear in the dataset."""
    expected = query.expected_species
    if not expected:
        return LabelingFunctionVote(source="rules", vote=None, rationale="no expected species in query")

    dataset_species = _list_field(record, "species")
    if not dataset_species:
        return LabelingFunctionVote(
            source="rules",
            vote=None,
            confidence=0.3,
            rationale="species field absent from record",
        )

    eq = _resolve_synonyms(expected, _SPECIES_SYNONYMS)
    dq = _resolve_synonyms(dataset_species, _SPECIES_SYNONYMS)
    overlap = eq & dq

    if overlap:
        return LabelingFunctionVote(
            source="rules",
            vote=2,
            confidence=0.75,
            rationale=f"species match: {sorted(overlap)}",
            evidence=[f"species matched: {s}" for s in sorted(overlap)],
        )
    return LabelingFunctionVote(
        source="rules",
        vote=None,
        confidence=0.6,
        rationale="expected species not found; deferring to other labelers",
    )


def lf_species_mismatch(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote 0 when query explicitly requires a species that is absent AND a
    different (incompatible) species is present."""
    expected = query.expected_species
    if not expected:
        return LabelingFunctionVote(source="rules", vote=None, rationale="no expected species in query")

    dataset_species = _list_field(record, "species")
    if not dataset_species:
        return LabelingFunctionVote(
            source="rules", vote=None, confidence=0.4,
            rationale="species field absent; cannot confirm mismatch",
        )

    eq = _resolve_synonyms(expected, _SPECIES_SYNONYMS)
    dq = _resolve_synonyms(dataset_species, _SPECIES_SYNONYMS)

    if eq & dq:
        return LabelingFunctionVote(source="rules", vote=None, rationale="species overlap found; not a mismatch")

    if dq:
        return LabelingFunctionVote(
            source="rules",
            vote=0,
            confidence=0.80,
            rationale=f"species mismatch: query wants {sorted(eq)}, dataset has {sorted(dq)}",
            evidence=[f"species mismatch: expected {sorted(eq)}, got {sorted(dq)}"],
        )
    return LabelingFunctionVote(source="rules", vote=None, rationale="dataset species unknown")


def lf_modality_match(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote +1 when expected modalities overlap with dataset modalities."""
    expected = query.expected_modalities
    if not expected:
        return LabelingFunctionVote(source="rules", vote=None, rationale="no expected modalities in query")

    dataset_mods = _list_field(record, "modalities")
    if not dataset_mods:
        return LabelingFunctionVote(
            source="rules", vote=None, confidence=0.3,
            rationale="modalities field absent from record",
        )

    eq = _resolve_synonyms(expected, _MODALITY_SYNONYMS)
    dq = _resolve_synonyms(dataset_mods, _MODALITY_SYNONYMS)
    overlap = eq & dq

    if overlap:
        return LabelingFunctionVote(
            source="rules",
            vote=2,
            confidence=0.80,
            rationale=f"modality match: {sorted(overlap)}",
            evidence=[f"modality matched: {m}" for m in sorted(overlap)],
        )
    return LabelingFunctionVote(
        source="rules", vote=None, confidence=0.55,
        rationale="expected modalities not found in dataset",
    )


def lf_modality_mismatch(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote 0 when query requires a modality that is clearly absent and an
    incompatible modality is present."""
    expected = query.expected_modalities
    if not expected:
        return LabelingFunctionVote(source="rules", vote=None, rationale="no expected modalities in query")

    dataset_mods = _list_field(record, "modalities")
    if not dataset_mods:
        return LabelingFunctionVote(
            source="rules", vote=None, confidence=0.4,
            rationale="modalities absent; cannot confirm mismatch",
        )

    eq = _resolve_synonyms(expected, _MODALITY_SYNONYMS)
    dq = _resolve_synonyms(dataset_mods, _MODALITY_SYNONYMS)

    if eq & dq:
        return LabelingFunctionVote(source="rules", vote=None, rationale="modality overlap found")

    # Only vote 0 if the dataset has concrete modalities (not empty)
    if dq:
        return LabelingFunctionVote(
            source="rules",
            vote=0,
            confidence=0.75,
            rationale=f"modality mismatch: query wants {sorted(eq)}, dataset has {sorted(dq)}",
            evidence=[f"modality mismatch: expected {sorted(eq)}, got {sorted(dq)}"],
        )
    return LabelingFunctionVote(source="rules", vote=None, rationale="dataset modality unknown")


def lf_task_match(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote +1 when expected tasks overlap with dataset tasks."""
    expected = query.expected_tasks
    if not expected:
        return LabelingFunctionVote(source="rules", vote=None, rationale="no expected tasks in query")

    dataset_tasks = _list_field(record, "tasks", "behaviors")
    if not dataset_tasks:
        return LabelingFunctionVote(
            source="rules", vote=None, confidence=0.3,
            rationale="tasks field absent from record",
        )

    eq = {_norm(t) for t in expected}
    dq = {_norm(t) for t in dataset_tasks}

    # Substring matching for partial overlaps
    matches = [qt for qt in eq for dt in dq if qt in dt or dt in qt]
    if matches:
        return LabelingFunctionVote(
            source="rules",
            vote=2,
            confidence=0.70,
            rationale=f"task match: {matches}",
            evidence=[f"task matched: {m}" for m in matches],
        )
    return LabelingFunctionVote(
        source="rules", vote=None, confidence=0.50,
        rationale="expected tasks not found in dataset",
    )


def lf_task_partial_match(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote 1 when the dataset description mentions task-adjacent terms but
    the tasks field is absent or sparse."""
    expected = query.expected_tasks
    if not expected:
        return LabelingFunctionVote(source="rules", vote=None, rationale="no expected tasks in query")

    description = str(record.get("description", "") or "")
    title = str(record.get("title", "") or "")
    text = _norm(description + " " + title)

    hits = [t for t in expected if _norm(t) in text]
    if hits:
        return LabelingFunctionVote(
            source="rules",
            vote=1,
            confidence=0.50,
            rationale=f"task terms in description (partial): {hits}",
            evidence=[f"task mentioned in text: {h}" for h in hits],
        )
    return LabelingFunctionVote(source="rules", vote=None, rationale="no task terms in text")


def lf_brain_region_match(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote +1 when expected brain regions overlap with dataset brain_regions."""
    expected = query.expected_brain_regions
    if not expected:
        return LabelingFunctionVote(source="rules", vote=None, rationale="no expected brain regions in query")

    dataset_regions = _list_field(record, "brain_regions")
    if not dataset_regions:
        return LabelingFunctionVote(
            source="rules", vote=None, confidence=0.35,
            rationale="brain_regions field absent from record",
        )

    eq = _resolve_synonyms(expected, _BRAIN_REGION_SYNONYMS)
    dq = _resolve_synonyms(dataset_regions, _BRAIN_REGION_SYNONYMS)
    overlap = eq & dq

    if overlap:
        return LabelingFunctionVote(
            source="rules",
            vote=2,
            confidence=0.75,
            rationale=f"brain region match: {sorted(overlap)}",
            evidence=[f"brain_region matched: {r}" for r in sorted(overlap)],
        )
    return LabelingFunctionVote(
        source="rules", vote=None, confidence=0.50,
        rationale="expected brain regions not found",
    )


def lf_brain_region_mismatch(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote 0 when required brain regions are absent and incompatible regions present."""
    expected = query.expected_brain_regions
    if not expected:
        return LabelingFunctionVote(source="rules", vote=None, rationale="no expected brain regions in query")

    dataset_regions = _list_field(record, "brain_regions")
    if not dataset_regions:
        return LabelingFunctionVote(source="rules", vote=None, rationale="brain_regions absent")

    eq = _resolve_synonyms(expected, _BRAIN_REGION_SYNONYMS)
    dq = _resolve_synonyms(dataset_regions, _BRAIN_REGION_SYNONYMS)

    if eq & dq:
        return LabelingFunctionVote(source="rules", vote=None, rationale="brain region overlap found")

    if dq:
        return LabelingFunctionVote(
            source="rules",
            vote=0,
            confidence=0.65,
            rationale=f"brain region mismatch: want {sorted(eq)}, got {sorted(dq)}",
            evidence=[f"brain_region mismatch: expected {sorted(eq)}, got {sorted(dq)}"],
        )
    return LabelingFunctionVote(source="rules", vote=None, rationale="dataset brain regions unknown")


def lf_hard_negative_violation(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote 0 when the dataset matches a known hard-negative pattern.

    Hard-negative violations STRONGLY override positive signals.
    """
    hard_negatives = query.hard_negatives
    if not hard_negatives:
        return LabelingFunctionVote(source="rules", vote=None, rationale="no hard negatives defined")

    title = _norm(str(record.get("title", "") or ""))
    description = _norm(str(record.get("description", "") or ""))
    modalities = " ".join(_norm(m) for m in _list_field(record, "modalities"))
    species = " ".join(_norm(s) for s in _list_field(record, "species"))
    text_blob = f"{title} {description} {modalities} {species}"

    violations = []
    for hn in hard_negatives:
        hn_norm = _norm(hn)
        # Check key phrases from the hard-negative description
        words = [w for w in hn_norm.split() if len(w) > 3]
        if sum(1 for w in words if w in text_blob) >= min(2, len(words)):
            violations.append(hn)

    if violations:
        return LabelingFunctionVote(
            source="rules",
            vote=0,
            confidence=0.85,
            rationale=f"hard-negative pattern matched: {violations}",
            evidence=[f"hard_negative_violation: {v}" for v in violations],
        )
    return LabelingFunctionVote(source="rules", vote=None, rationale="no hard-negative patterns matched")


def lf_missing_critical_metadata(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Abstain with lowered confidence when must_have fields are absent.

    Does NOT vote 0 — missing metadata means we don't know, not that it's
    irrelevant. It raises needs_human_review via the builder.
    """
    must_have = query.must_have
    if not must_have:
        return LabelingFunctionVote(source="rules", vote=None, rationale="no must_have constraints")

    missing = []
    for req in must_have:
        req_norm = _norm(req).split(":")[0]  # e.g. "modality:fMRI" → "modality"
        val = record.get(req_norm) or record.get(req_norm + "s")
        if not val:
            # Also check description for a mention
            desc = _norm(str(record.get("description", "") or ""))
            if req_norm not in desc:
                missing.append(req)

    if missing:
        return LabelingFunctionVote(
            source="rules",
            vote=None,
            confidence=0.35,
            rationale=f"required fields absent: {missing}",
            evidence=[f"missing_required_field: {m}" for m in missing],
        )
    return LabelingFunctionVote(source="rules", vote=None, rationale="all required fields present")


def lf_license_access(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote 1 (not 0) when license is missing or restrictive — lowers
    reusability but doesn't disqualify scientific relevance."""
    license_val = str(record.get("license", "") or "").lower()

    if not license_val or license_val in ("unknown", "none", ""):
        return LabelingFunctionVote(
            source="rules",
            vote=None,
            confidence=0.40,
            rationale="license field absent or unknown",
            evidence=["missing_metadata: license"],
        )

    open_licenses = {"cc by", "cc-by", "cc0", "public domain", "mit", "apache", "bsd", "pddl"}
    if any(ol in license_val for ol in open_licenses):
        return LabelingFunctionVote(
            source="rules",
            vote=None,
            confidence=0.60,
            rationale=f"open license detected: {license_val}",
            evidence=[f"license: {license_val}"],
        )

    # Restrictive license — note it but don't disqualify
    return LabelingFunctionVote(
        source="rules",
        vote=None,
        confidence=0.50,
        rationale=f"potentially restrictive license: {license_val}",
        evidence=[f"license: {license_val}"],
    )


def lf_data_standard_match(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote +1 when must_have data standard (NWB, BIDS) is present."""
    must_have_text = " ".join(_norm(m) for m in query.must_have)
    standards_needed: list[str] = []

    for std in ("nwb", "bids", "neurodata without borders", "brain imaging data structure"):
        if std in must_have_text:
            standards_needed.append(std)

    if not standards_needed:
        return LabelingFunctionVote(source="rules", vote=None, rationale="no data standard required")

    dataset_standards = _list_field(record, "data_standards")
    dataset_std_text = " ".join(_norm(s) for s in dataset_standards)

    matches = [s for s in standards_needed if s in dataset_std_text]
    if matches:
        return LabelingFunctionVote(
            source="rules",
            vote=2,
            confidence=0.70,
            rationale=f"data standard match: {matches}",
            evidence=[f"data_standard matched: {m}" for m in matches],
        )

    if dataset_standards:
        return LabelingFunctionVote(
            source="rules",
            vote=None,
            confidence=0.40,
            rationale=f"required standard {standards_needed} not in dataset standards {dataset_standards}",
        )
    return LabelingFunctionVote(
        source="rules",
        vote=None,
        confidence=0.35,
        rationale="data_standards field absent; cannot check",
    )


def lf_exact_dataset_match(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> LabelingFunctionVote:
    """Vote 3 when the query text references a specific known dataset ID or
    well-known name that appears to match this record."""
    query_text = _norm(query.query_text)
    source_id = _norm(str(record.get("source_id", "") or ""))
    title = _norm(str(record.get("title", "") or ""))

    if not source_id and not title:
        return LabelingFunctionVote(source="rules", vote=None, rationale="record has no id or title")

    # Check if query text mentions the source ID or a substantial fragment of title
    if source_id and len(source_id) > 3 and source_id in query_text:
        return LabelingFunctionVote(
            source="rules",
            vote=3,
            confidence=0.90,
            rationale=f"exact source_id match: {source_id}",
            evidence=[f"exact_id_match: {source_id}"],
        )

    title_words = [w for w in title.split() if len(w) > 4]
    if len(title_words) >= 3:
        hits = sum(1 for w in title_words if w in query_text)
        if hits >= max(3, len(title_words) // 2):
            return LabelingFunctionVote(
                source="rules",
                vote=2,
                confidence=0.65,
                rationale=f"strong title overlap with query ({hits}/{len(title_words)} words)",
                evidence=[f"title_word_overlap: {hits}/{len(title_words)}"],
            )

    return LabelingFunctionVote(source="rules", vote=None, rationale="no exact dataset match detected")


# ---------------------------------------------------------------------------
# Registry and batch application
# ---------------------------------------------------------------------------

#: All deterministic rule-based labeling functions in application order.
ALL_LABELING_FUNCTIONS = [
    lf_hard_negative_violation,   # must run first — overrides everything
    lf_species_mismatch,
    lf_modality_mismatch,
    lf_brain_region_mismatch,
    lf_species_match,
    lf_modality_match,
    lf_task_match,
    lf_task_partial_match,
    lf_brain_region_match,
    lf_data_standard_match,
    lf_exact_dataset_match,
    lf_missing_critical_metadata,
    lf_license_access,
]


def apply_all_labeling_functions(
    query: BenchmarkQueryV1, record: dict[str, Any]
) -> list[LabelingFunctionVote]:
    """Apply all labeling functions and return their votes."""
    return [lf(query, record) for lf in ALL_LABELING_FUNCTIONS]
