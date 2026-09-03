"""Paper-software exposure and open-data reanalysis joins.

Exposure is deliberately represented separately from demonstrated scientific impact:
a paper can have used an affected code path without its conclusions being changed.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from neural_search.software.schema import AuditFinding


class PaperSoftwareUsage(BaseModel):
    usage_id: str
    paper_id: str
    package_id: str
    release_id: str | None = None
    component_ids: list[str] = Field(default_factory=list)
    command: str | None = None
    options: list[str] = Field(default_factory=list)
    evidence_text: str | None = None
    evidence_source: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class PaperDatasetLink(BaseModel):
    paper_id: str
    dataset_id: str
    access: str = "unknown"
    raw_data_available: bool = False
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ReanalysisCandidate(BaseModel):
    candidate_id: str
    finding_id: str
    paper_id: str
    dataset_id: str
    exposure_confidence: float = Field(ge=0.0, le=1.0)
    dataset_link_confidence: float = Field(ge=0.0, le=1.0)
    raw_data_available: bool
    rationale: list[str] = Field(default_factory=list)

    @property
    def confidence(self) -> float:
        return round(self.exposure_confidence * self.dataset_link_confidence, 6)


def find_reanalysis_candidates(
    finding: AuditFinding,
    usages: list[PaperSoftwareUsage],
    paper_dataset_links: list[PaperDatasetLink],
    *,
    require_raw_data: bool = True,
) -> list[ReanalysisCandidate]:
    """Join an audit finding to exposed papers and their reusable datasets.

    The join requires either an explicit component match or an affected release match;
    generic package use is not enough to declare exposure.
    """

    affected_components = {finding.component_id}
    affected_releases = set(finding.affected_release_ids)
    exposed: dict[str, float] = {}
    for usage in usages:
        component_match = bool(affected_components.intersection(usage.component_ids))
        release_match = bool(usage.release_id and usage.release_id in affected_releases)
        if component_match or release_match:
            exposed[usage.paper_id] = max(exposed.get(usage.paper_id, 0.0), usage.confidence)

    candidates: list[ReanalysisCandidate] = []
    seen: set[tuple[str, str]] = set()
    for link in paper_dataset_links:
        if link.paper_id not in exposed:
            continue
        if require_raw_data and not link.raw_data_available:
            continue
        key = (link.paper_id, link.dataset_id)
        if key in seen:
            continue
        seen.add(key)
        rationale = ["paper is mapped to the affected component or release"]
        if link.raw_data_available:
            rationale.append("raw data are reported available for controlled rerun")
        candidates.append(
            ReanalysisCandidate(
                candidate_id=f"reanalysis:{finding.finding_id}:{link.paper_id}:{link.dataset_id}",
                finding_id=finding.finding_id,
                paper_id=link.paper_id,
                dataset_id=link.dataset_id,
                exposure_confidence=exposed[link.paper_id],
                dataset_link_confidence=link.confidence,
                raw_data_available=link.raw_data_available,
                rationale=rationale,
            )
        )
    return sorted(candidates, key=lambda candidate: (-candidate.confidence, candidate.candidate_id))
