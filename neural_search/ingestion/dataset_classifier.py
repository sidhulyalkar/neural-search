"""Dataset inclusion classifier — Tier 2 ingestion gate.

Tier 2 sources (OSF, figshare, zenodo) contain papers, slides, code, and
non-neuroscience data. Every record must pass all four checks before ingestion.

Records that fail are written to data/corpus/rejected/tier2_rejected.jsonl
with the failure reason.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_NEURO_SIGNALS = [
    "neuron", "neural", "brain", "cortex", "hippocampus", "cerebellum", "striatum",
    "fmri", "eeg", "ecog", "meg", "electrophysiology", "ephys", "spike",
    "calcium imaging", "neuropixels", "fiber photometry", "patch clamp",
    "mouse", "rat", "macaque", "human subject", "participant", "recording",
    "modality", "electrode", "stimulus", "behavior", "trial", "task",
    "nwb", "bids", "dandiset", "openneuro",
]

_EXCLUSION_SIGNALS = [
    "review article", "meta-analysis paper", "python script", "jupyter notebook",
    "analysis code", "source code", "preprint text", "supplementary material",
    "presentation slides", "poster presentation",
]

_NON_DATASET_TYPES = {"publication", "preprint", "software", "code", "presentation", "other"}

_OPEN_LICENSES = {
    "cc-by", "cc-0", "cc0", "cc by", "cc-by-4.0", "cc-by-sa", "pddl",
    "odc-by", "apache", "mit", "gpl", "bsd", "public domain",
}


@dataclass
class ClassificationResult:
    accepted: bool
    failure_reason: str = ""
    signals_found: list[str] = field(default_factory=list)


def _has_raw_or_processed_data(rec: dict) -> bool:
    rtype = str(rec.get("resource_type") or rec.get("type") or "").lower()
    if rtype in _NON_DATASET_TYPES:
        return False
    text = f"{rec.get('title', '')} {rec.get('description', '')}".lower()
    for sig in _EXCLUSION_SIGNALS:
        if sig in text:
            return False
    return True


def _has_species_or_modality_signal(rec: dict) -> tuple[bool, list[str]]:
    text = (
        f"{rec.get('title', '')} {rec.get('description', '')} "
        f"{' '.join(rec.get('keywords', []))}"
    ).lower()
    found = [s for s in _NEURO_SIGNALS if s in text]
    return len(found) > 0, found


def _has_reuse_license(rec: dict) -> bool:
    license_raw = str(rec.get("license") or rec.get("license_name") or "").lower()
    if not license_raw:
        return False
    return any(lic in license_raw for lic in _OPEN_LICENSES)


def _has_doi_or_accession(rec: dict) -> bool:
    doi = rec.get("doi") or rec.get("DOI") or ""
    accession = rec.get("accession") or rec.get("identifier") or rec.get("id") or ""
    return bool(doi) or bool(str(accession).strip())


def is_valid_dataset(record: dict[str, Any]) -> ClassificationResult:
    """Check all four gates. Returns ClassificationResult with accepted=True/False."""
    if not _has_raw_or_processed_data(record):
        return ClassificationResult(
            accepted=False,
            failure_reason="not_raw_or_processed_data: resource_type or content suggests non-dataset",
        )

    has_signal, found_signals = _has_species_or_modality_signal(record)
    if not has_signal:
        return ClassificationResult(
            accepted=False,
            failure_reason="no_species_or_modality_signal: no neuroscience keywords found",
        )

    if not _has_reuse_license(record):
        return ClassificationResult(
            accepted=False,
            failure_reason=f"no_reuse_license: license '{record.get('license', 'none')}' not recognized as open",
        )

    if not _has_doi_or_accession(record):
        return ClassificationResult(
            accepted=False,
            failure_reason="no_persistent_identifier: no DOI or accession number",
        )

    return ClassificationResult(accepted=True, signals_found=found_signals)
