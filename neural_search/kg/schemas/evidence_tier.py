"""Evidence tier taxonomy for reanalysis suggestions.

Formalizes the six-tier framework documented in the whitepaper
(``docs/whitepaper/neural_search_whitepaper.tex``, \\S evidence-tiers) as an
actual graph-edge property rather than prose-only documentation. Ordered
weakest to strongest:

1. ``heuristic_candidate`` -- metadata/profile match only
   (``dataset_old_dataset_new_method_candidate``).
2. ``evidence_backed_bridge`` -- a similar dataset had the method applied in
   a linked paper (``dataset_reanalysis_bridge_dataset``).
3. ``source_declared`` -- the dataset's own metadata or linked paper
   explicitly declares the required fields.
4. ``file_validated`` -- direct NWB/BIDS/raw-file inspection confirms the
   requirements (``neural_search.affordances.validators.nwb_validator``,
   ``neural_search.file_inspection``).
5. ``human_validated`` -- an expert reviewed and accepted the suggestion.
6. ``computed`` -- the reanalysis was actually run and passed QC.

Only ``file_validated``, ``human_validated``, and ``computed`` should read as
trustworthy to a researcher deciding whether to act on a suggestion; the
first two are surfaced but explicitly not validated.
"""

from __future__ import annotations

from enum import Enum


class EvidenceTier(str, Enum):
    HEURISTIC_CANDIDATE = "heuristic_candidate"
    EVIDENCE_BACKED_BRIDGE = "evidence_backed_bridge"
    SOURCE_DECLARED = "source_declared"
    FILE_VALIDATED = "file_validated"
    HUMAN_VALIDATED = "human_validated"
    COMPUTED = "computed"


# Ordered weakest -> strongest, for comparisons/upgrades.
EVIDENCE_TIER_ORDER: tuple[EvidenceTier, ...] = (
    EvidenceTier.HEURISTIC_CANDIDATE,
    EvidenceTier.EVIDENCE_BACKED_BRIDGE,
    EvidenceTier.SOURCE_DECLARED,
    EvidenceTier.FILE_VALIDATED,
    EvidenceTier.HUMAN_VALIDATED,
    EvidenceTier.COMPUTED,
)

TRUSTWORTHY_TIERS: frozenset[EvidenceTier] = frozenset(
    {EvidenceTier.FILE_VALIDATED, EvidenceTier.HUMAN_VALIDATED, EvidenceTier.COMPUTED}
)


def tier_rank(tier: EvidenceTier | str) -> int:
    """Index into EVIDENCE_TIER_ORDER; higher is stronger evidence."""

    tier = EvidenceTier(tier)
    return EVIDENCE_TIER_ORDER.index(tier)


def is_trustworthy(tier: EvidenceTier | str) -> bool:
    return EvidenceTier(tier) in TRUSTWORTHY_TIERS


def upgrade_tier(current: EvidenceTier | str, candidate: EvidenceTier | str) -> EvidenceTier:
    """Return the stronger of two tiers — never silently downgrade evidence."""

    current = EvidenceTier(current)
    candidate = EvidenceTier(candidate)
    return candidate if tier_rank(candidate) > tier_rank(current) else current
