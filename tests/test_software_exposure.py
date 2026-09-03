"""Tests for conservative software-exposure to open-data reanalysis joins."""

from neural_search.software.exposure import (
    PaperDatasetLink,
    PaperSoftwareUsage,
    find_reanalysis_candidates,
)
from neural_search.software.schema import AuditFinding, AuditState


def test_reanalysis_candidates_require_component_or_release_exposure() -> None:
    finding = AuditFinding(
        finding_id="f1",
        hypothesis_id="h1",
        component_id="kilosort:cluster",
        summary="verified discrepancy",
        state=AuditState.NUMERICALLY_VERIFIED,
        verification_ids=["v1"],
        affected_release_ids=["kilosort:4.1.5"],
    )
    usages = [
        PaperSoftwareUsage(
            usage_id="u1",
            paper_id="p1",
            package_id="kilosort",
            component_ids=["kilosort:cluster"],
            confidence=0.9,
        ),
        PaperSoftwareUsage(
            usage_id="u2",
            paper_id="p2",
            package_id="kilosort",
            component_ids=[],
            confidence=0.9,
        ),
        PaperSoftwareUsage(
            usage_id="u3",
            paper_id="p3",
            package_id="kilosort",
            release_id="kilosort:4.1.5",
            confidence=0.8,
        ),
    ]
    links = [
        PaperDatasetLink(paper_id="p1", dataset_id="d1", raw_data_available=True, confidence=0.95),
        PaperDatasetLink(paper_id="p2", dataset_id="d2", raw_data_available=True, confidence=0.95),
        PaperDatasetLink(paper_id="p3", dataset_id="d3", raw_data_available=True, confidence=0.7),
    ]

    candidates = find_reanalysis_candidates(finding, usages, links)

    assert [candidate.dataset_id for candidate in candidates] == ["d1", "d3"]
    assert candidates[0].confidence == 0.855


def test_reanalysis_candidates_can_require_raw_data() -> None:
    finding = AuditFinding(
        finding_id="f1",
        hypothesis_id="h1",
        component_id="suite2p:phase",
        summary="verified discrepancy",
        state=AuditState.NUMERICALLY_VERIFIED,
        verification_ids=["v1"],
    )
    usages = [
        PaperSoftwareUsage(
            usage_id="u1",
            paper_id="p1",
            package_id="suite2p",
            component_ids=["suite2p:phase"],
        )
    ]
    links = [PaperDatasetLink(paper_id="p1", dataset_id="d1", raw_data_available=False)]

    assert find_reanalysis_candidates(finding, usages, links) == []
    assert len(find_reanalysis_candidates(finding, usages, links, require_raw_data=False)) == 1
