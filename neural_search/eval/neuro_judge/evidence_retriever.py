"""Build EvidencePackets from corpus records, concept memory, and affordance probes.

Usage::

    packet = build_evidence_packet(query, candidate_record, concept_result=cr)
"""

from __future__ import annotations

import re
from typing import Any

from neural_search.eval.neuro_judge.evidence_packet import (
    AffordanceMatch,
    EvidencePacket,
    LinkedPaper,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DATA_STANDARD_SIGNALS = frozenset({"nwb", "bids", "alf", "neurodata", "h5", "hdf5", "zarr"})
_RAW_DATA_KEYWORDS = frozenset(
    {"raw", "ap_band", "raw_ephys", "raw_voltage", "broadband", "continuous", "lfp_raw"}
)
_RAW_DATA_NEGATION_PHRASES = (
    "no raw",
    "without raw",
    "raw data not",
    "raw ap-band data not",
    "raw ap band data not",
    "not explicitly confirmed",
    "not confirmed",
    "not available",
)
_PROCESSED_DATA_KEYWORDS = frozenset(
    {
        "processed",
        "spike_sorted",
        "spike_times",
        "units",
        "trials",
        "kilosort",
        "preprocessed",
        "aligned",
    }
)

_NEGATIVE_PATTERN_SPLIT = re.compile(r"[,;]|\bor\b", re.IGNORECASE)
_WORD_PATTERN = re.compile(r"[a-z0-9]+")


def _normalise_text(text: str) -> str:
    """Return lowercase alphanumeric tokens joined by spaces for phrase matching."""
    return " ".join(_WORD_PATTERN.findall(text.lower()))


def _hard_negative_pattern_matches(pattern: str, record_text: str) -> bool:
    """Conservative hard-negative match.

    Hard negatives in benchmark queries often read like short explanations
    ("fear conditioning instead of spatial navigation"), not literal zero-label
    triggers. Match only contiguous normalized phrases or comma/semicolon
    alternatives, avoiding broad single-token matches such as "raw" or "no".
    """
    normalized_record = _normalise_text(record_text)
    if not normalized_record:
        return False

    alternatives: list[str] = []
    for part in _NEGATIVE_PATTERN_SPLIT.split(pattern):
        normalized_part = _normalise_text(part)
        if normalized_part:
            alternatives.append(normalized_part)
    for alt in alternatives:
        if len(alt.split()) < 2:
            continue
        if alt in normalized_record:
            return True
    return False


def _extract_file_format_evidence(record: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    for field in ("data_standards", "file_types", "formats"):
        values = record.get(field) or []
        if field == "data_standards":
            values = values or meta.get("record_data_standards") or []
        for val in values:
            s = str(val).lower()
            for kw in _DATA_STANDARD_SIGNALS:
                if kw in s:
                    signals.append(f"{field}:{val}")
    desc = str(record.get("description") or "").lower()
    for kw in _DATA_STANDARD_SIGNALS:
        if kw in desc:
            signals.append(f"description_mentions:{kw}")
    return list(dict.fromkeys(signals))  # deduplicate, preserve order


def _infer_raw_data(record: dict[str, Any]) -> bool | None:
    desc = str(record.get("description") or "").lower()
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    standards = record.get("data_standards") or meta.get("record_data_standards") or []
    file_evidence = " ".join(str(v) for v in standards)
    combined = desc + " " + file_evidence.lower()
    if any(phrase in combined for phrase in _RAW_DATA_NEGATION_PHRASES):
        if any(phrase in combined for phrase in ("no raw", "without raw", "not available")):
            return False
        return None
    has_raw = any(kw in combined for kw in _RAW_DATA_KEYWORDS)
    has_processed = any(kw in combined for kw in _PROCESSED_DATA_KEYWORDS)
    if has_raw:
        return True
    if has_processed and not has_raw:
        return None  # processed only, raw unknown
    return None


def _infer_processed_data(record: dict[str, Any]) -> bool | None:
    desc = str(record.get("description") or "").lower()
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    standards = " ".join(str(v).lower() for v in (record.get("data_standards") or meta.get("record_data_standards") or []))
    has_processed = any(kw in f"{desc} {standards}" for kw in _PROCESSED_DATA_KEYWORDS)
    return True if has_processed else None


def _build_linked_papers(record: dict[str, Any]) -> list[LinkedPaper]:
    papers: list[LinkedPaper] = []
    # Try common corpus field names for linked paper metadata
    for field_name in ("linked_paper", "paper", "publication", "associated_paper"):
        raw = record.get(field_name)
        if isinstance(raw, dict):
            papers.append(
                LinkedPaper(
                    title=str(raw.get("title") or ""),
                    abstract=str(raw.get("abstract") or ""),
                    doi=str(raw.get("doi") or ""),
                    year=raw.get("year"),
                )
            )
        elif isinstance(raw, str) and raw.strip():
            papers.append(LinkedPaper(title=raw))
    for meta_key in ("metadata",):
        meta = record.get(meta_key)
        if not isinstance(meta, dict):
            continue
        for sub in ("linked_paper", "paper_title", "paper_abstract"):
            val = meta.get(sub)
            if isinstance(val, dict):
                papers.append(
                    LinkedPaper(
                        title=str(val.get("title") or ""),
                        abstract=str(val.get("abstract") or ""),
                    )
                )
            elif isinstance(val, str) and val.strip():
                papers.append(LinkedPaper(title=val))
    return papers


def _build_affordance_matches(
    query: Any,
    record: dict[str, Any],
    concept_result: dict[str, Any] | None,
) -> list[AffordanceMatch]:
    expected: list[str] = list(getattr(query, "expected_analysis_affordances", None) or [])
    if not expected:
        # Fall back to must_have as proxy for required affordances
        expected = list(getattr(query, "must_have", None) or [])

    record_desc = str(record.get("description") or record.get("dataset_description") or "").lower()
    record_affordances = [str(a).lower() for a in (record.get("affordances") or [])]

    concept_missing: list[str] = []
    if concept_result:
        concept_missing = [str(m) for m in (concept_result.get("missing_evidence") or [])]

    matches: list[AffordanceMatch] = []
    for affordance in expected:
        aff_lower = affordance.lower()
        direct_match = any(aff_lower in ra for ra in record_affordances)
        desc_mention = aff_lower in record_desc
        matched = direct_match or desc_mention
        missing_reqs = [m for m in concept_missing if aff_lower in m.lower()]
        confidence = 0.7 if direct_match else (0.4 if desc_mention else 0.0)
        matches.append(
            AffordanceMatch(
                affordance=affordance,
                matched=matched,
                confidence=confidence,
                missing_requirements=missing_reqs,
                rationale=(
                    "direct affordance field match"
                    if direct_match
                    else ("description mention" if desc_mention else "no evidence found")
                ),
            )
        )
    return matches


def _build_known_failure_warnings(
    query: Any,
    record: dict[str, Any],
) -> list[str]:
    warnings: list[str] = []
    hard_negatives: list[str] = list(getattr(query, "hard_negatives", None) or [])
    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    modalities = record.get("modalities") or record.get("dataset_modalities") or meta.get("record_modalities") or []
    species = record.get("species") or record.get("dataset_species") or meta.get("record_species") or []
    record_text = (
        str(record.get("description") or record.get("dataset_description") or "")
        + " "
        + str(record.get("title") or record.get("dataset_title") or "")
        + " "
        + " ".join(str(m) for m in modalities)
        + " "
        + " ".join(str(s) for s in species)
    ).lower()
    for hn in hard_negatives:
        if _hard_negative_pattern_matches(hn, record_text):
            warnings.append(f"possible_hard_negative: {hn}")
    return warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_evidence_packet(
    query: Any,
    record: dict[str, Any],
    concept_result: dict[str, Any] | None = None,
) -> EvidencePacket:
    """Assemble an EvidencePacket from a corpus record and optional concept signals.

    Args:
        query: BenchmarkQueryV1 (or any object with the expected attributes).
        record: Corpus record dict (from pooled candidates JSONL).
        concept_result: Optional dict from ConceptRerankedResult / ConceptExplanation.
    """
    # Concept memory signals
    concept_summary = ""
    matched_concept_names: list[str] = []
    concept_missing: list[str] = []
    concept_hn_conflicts: list[str] = []
    if concept_result:
        concept_summary = str(concept_result.get("explanation_summary") or "")
        for mc in concept_result.get("matched_concepts") or []:
            if isinstance(mc, dict):
                matched_concept_names.append(str(mc.get("canonical_name") or mc.get("name") or ""))
        concept_missing = [str(m) for m in (concept_result.get("missing_evidence") or [])]
        concept_hn_conflicts = [str(c) for c in (concept_result.get("hard_negative_conflicts") or [])]

    meta = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}

    return EvidencePacket(
        # query
        query_id=str(getattr(query, "query_id", "") or record.get("query_id", "")),
        query_text=str(getattr(query, "query_text", "") or ""),
        query_intent=str(getattr(query, "intent", "") or getattr(query, "query_intent", "") or ""),
        hard_negatives=list(getattr(query, "hard_negatives", None) or []),
        expected_species=list(getattr(query, "expected_species", None) or []),
        expected_modalities=list(getattr(query, "expected_modalities", None) or []),
        expected_brain_regions=list(getattr(query, "expected_brain_regions", None) or []),
        expected_tasks=list(getattr(query, "expected_tasks", None) or []),
        expected_analysis_affordances=list(
            getattr(query, "expected_analysis_affordances", None) or []
        ),
        # dataset — accept both pooled-candidate keys (dataset_*) and flat corpus keys
        dataset_id=str(record.get("dataset_id") or record.get("source_id") or ""),
        title=str(
            record.get("title") or record.get("dataset_title") or ""
        ),
        source_archive=str(
            record.get("source") or record.get("dataset_source") or ""
        ),
        source_url=str(
            record.get("source_url")
            or record.get("dataset_source_url")
            or record.get("url")
            or ""
        ),
        description=str(
            record.get("description") or record.get("dataset_description") or ""
        ),
        dataset_modalities=[
            str(m)
            for m in (
                record.get("modalities")
                or record.get("dataset_modalities")
                or meta.get("record_modalities")
                or []
            )
        ],
        dataset_species=[
            str(s)
            for s in (
                record.get("species")
                or record.get("dataset_species")
                or meta.get("record_species")
                or []
            )
        ],
        dataset_brain_regions=[
            str(r)
            for r in (
                record.get("brain_regions")
                or record.get("dataset_brain_regions")
                or meta.get("record_brain_regions")
                or []
            )
        ],
        dataset_tasks=[
            str(t)
            for t in (
                record.get("tasks")
                or record.get("dataset_tasks")
                or meta.get("record_tasks")
                or []
            )
        ],
        data_standards=[
            str(s)
            for s in (
                record.get("data_standards")
                or record.get("dataset_data_standards")
                or meta.get("record_data_standards")
                or []
            )
        ],
        license=str(record.get("license") or ""),
        # derived evidence
        linked_papers=_build_linked_papers(record),
        affordance_matches=_build_affordance_matches(query, record, concept_result),
        concept_explanation_summary=concept_summary,
        matched_concept_names=[n for n in matched_concept_names if n],
        concept_missing_evidence=concept_missing,
        concept_hard_negative_conflicts=concept_hn_conflicts,
        # raw/processed signals
        has_raw_data=_infer_raw_data(record),
        has_processed_data=_infer_processed_data(record),
        file_format_evidence=_extract_file_format_evidence(record),
        # warnings
        known_failure_warnings=_build_known_failure_warnings(query, record),
    )
