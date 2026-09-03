"""Historical benchmark contracts for scientific software auditing.

Benchmark cases should freeze a repository immediately before a known fix so model-assisted
analysis can be evaluated without leaking the fix itself into the input snapshot.
"""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field


class HistoricalAuditCase(BaseModel):
    case_id: str
    repository_url: str
    pre_fix_revision: str
    fix_revision: str
    affected_component_ids: list[str]
    expected_finding_summary: str
    verification_oracle: str | None = None
    scientific_domain: str
    metadata: dict[str, object] = Field(default_factory=dict)


class AuditBenchmarkPrediction(BaseModel):
    case_id: str
    ranked_component_ids: list[str]
    finding_detected: bool
    reproducer_succeeded: bool = False
    patch_correct: bool | None = None
    regression_test_adequate: bool | None = None
    duplicate_detected: bool | None = None
    submitted: bool = False
    maintainer_accepted: bool | None = None


class AuditBenchmarkMetrics(BaseModel):
    cases: int
    component_recall_at_1: float
    component_recall_at_5: float
    finding_recall: float
    reproducer_success_rate: float
    patch_correctness_rate: float | None
    regression_test_adequacy_rate: float | None
    duplicate_detection_rate: float | None
    maintainer_acceptance_rate: float | None


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def compute_audit_benchmark_metrics(
    cases: list[HistoricalAuditCase],
    predictions: list[AuditBenchmarkPrediction],
) -> AuditBenchmarkMetrics:
    """Compute evidence-oriented metrics over frozen historical audit cases."""

    by_case = {prediction.case_id: prediction for prediction in predictions}
    missing = sorted({case.case_id for case in cases} - set(by_case))
    if missing:
        raise ValueError(f"missing benchmark predictions for: {', '.join(missing)}")
    if not cases:
        raise ValueError("audit benchmark requires at least one case")

    recall1: list[bool] = []
    recall5: list[bool] = []
    finding: list[bool] = []
    reproducer: list[bool] = []
    patch: list[bool] = []
    tests: list[bool] = []
    duplicates: list[bool] = []
    acceptance: list[bool] = []

    for case in cases:
        prediction = by_case[case.case_id]
        gold = set(case.affected_component_ids)
        recall1.append(bool(gold.intersection(prediction.ranked_component_ids[:1])))
        recall5.append(bool(gold.intersection(prediction.ranked_component_ids[:5])))
        finding.append(prediction.finding_detected)
        reproducer.append(prediction.reproducer_succeeded)
        if prediction.patch_correct is not None:
            patch.append(prediction.patch_correct)
        if prediction.regression_test_adequate is not None:
            tests.append(prediction.regression_test_adequate)
        if prediction.duplicate_detected is not None:
            duplicates.append(prediction.duplicate_detected)
        if prediction.submitted and prediction.maintainer_accepted is not None:
            acceptance.append(prediction.maintainer_accepted)

    return AuditBenchmarkMetrics(
        cases=len(cases),
        component_recall_at_1=float(_rate(recall1) or 0.0),
        component_recall_at_5=float(_rate(recall5) or 0.0),
        finding_recall=float(_rate(finding) or 0.0),
        reproducer_success_rate=float(_rate(reproducer) or 0.0),
        patch_correctness_rate=_rate(patch),
        regression_test_adequacy_rate=_rate(tests),
        duplicate_detection_rate=_rate(duplicates),
        maintainer_acceptance_rate=_rate(acceptance),
    )


def group_cases_by_domain(cases: list[HistoricalAuditCase]) -> dict[str, list[str]]:
    """Return case IDs grouped by scientific domain for stratified reporting."""

    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for case in cases:
        grouped[case.scientific_domain].append(case.case_id)
    return {domain: sorted(case_ids) for domain, case_ids in sorted(grouped.items())}
