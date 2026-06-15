"""Calibration of neuro_judge labels against human gold labels.

calibrate(judge_records, human_records) → CalibrationReport

Metrics:
  - exact_agreement      (fraction identical)
  - agreement_within_1   (fraction |diff| <= 1)
  - quadratic_weighted_kappa
  - confusion_matrix      (4×4 list-of-lists)
  - false_high_examples   (judge > human)
  - false_low_examples    (judge < human)
  - by_query_intent       (breakdown per intent)
  - by_modality           (breakdown per modality)
  - by_failure_mode       (breakdown per failure_mode)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from neural_search.eval.neuro_judge.evidence_packet import NEURO_JUDGE_WATERMARK

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _qwk(labels_true: list[int], labels_pred: list[int], num_classes: int = 4) -> float:
    """Quadratic weighted kappa for ordinal labels in [0, num_classes)."""
    n = len(labels_true)
    if n == 0:
        return 0.0

    # Confusion matrix
    conf: list[list[int]] = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(labels_true, labels_pred, strict=False):
        conf[t][p] += 1

    # Weight matrix (quadratic)
    weight: list[list[float]] = [
        [(i - j) ** 2 / (num_classes - 1) ** 2 for j in range(num_classes)]
        for i in range(num_classes)
    ]

    # Expected matrix from marginals
    hist_true = [sum(conf[i]) for i in range(num_classes)]
    hist_pred = [sum(conf[i][j] for i in range(num_classes)) for j in range(num_classes)]
    expected: list[list[float]] = [
        [hist_true[i] * hist_pred[j] / n for j in range(num_classes)]
        for i in range(num_classes)
    ]

    numerator = sum(
        weight[i][j] * conf[i][j]
        for i in range(num_classes)
        for j in range(num_classes)
    )
    denominator = sum(
        weight[i][j] * expected[i][j]
        for i in range(num_classes)
        for j in range(num_classes)
    )
    if denominator == 0.0:
        return 1.0
    return 1.0 - numerator / denominator


def _confusion_matrix(
    labels_true: list[int],
    labels_pred: list[int],
    num_classes: int = 4,
) -> list[list[int]]:
    conf: list[list[int]] = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(labels_true, labels_pred, strict=False):
        conf[t][p] += 1
    return conf


# ---------------------------------------------------------------------------
# Example containers
# ---------------------------------------------------------------------------


@dataclass
class LabelMismatch:
    query_id: str
    dataset_id: str
    judge_label: int
    human_label: int
    judge_rationale: str = ""


# ---------------------------------------------------------------------------
# CalibrationReport
# ---------------------------------------------------------------------------


@dataclass
class CalibrationReport:
    n_pairs: int = 0
    exact_agreement: float = 0.0
    agreement_within_1: float = 0.0
    quadratic_weighted_kappa: float = 0.0
    confusion_matrix: list[list[int]] = field(default_factory=list)
    false_high_examples: list[LabelMismatch] = field(default_factory=list)
    false_low_examples: list[LabelMismatch] = field(default_factory=list)
    by_query_intent: dict[str, dict[str, float]] = field(default_factory=dict)
    by_modality: dict[str, dict[str, float]] = field(default_factory=dict)
    by_failure_mode: dict[str, dict[str, float]] = field(default_factory=dict)
    watermark: str = NEURO_JUDGE_WATERMARK

    def summary(self) -> str:
        return (
            f"n={self.n_pairs} | exact={self.exact_agreement:.3f} "
            f"| within1={self.agreement_within_1:.3f} "
            f"| QWK={self.quadratic_weighted_kappa:.3f}"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calibrate(
    judge_records: list[dict[str, Any]],
    human_records: list[dict[str, Any]],
) -> CalibrationReport:
    """Compare neuro_judge labels to human gold labels.

    Both lists should contain dicts with at least:
      ``query_id``, ``dataset_id``, ``label`` (int 0–3).

    Judge records may also have ``rationale_short``, ``failure_modes``,
    ``matched_dimensions``.
    Human records may also have ``intent``, ``modalities``.
    """
    # Index human labels by (query_id, dataset_id)
    human_index: dict[tuple[str, str], dict[str, Any]] = {}
    for rec in human_records:
        key = (str(rec.get("query_id", "")), str(rec.get("dataset_id", "")))
        human_index[key] = rec

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for j_rec in judge_records:
        key = (str(j_rec.get("query_id", "")), str(j_rec.get("dataset_id", "")))
        h_rec = human_index.get(key)
        if h_rec is not None:
            pairs.append((j_rec, h_rec))

    if not pairs:
        return CalibrationReport()

    _str_label_map = {
        "not_relevant": 0, "weakly_relevant": 1,
        "partially_relevant": 2, "highly_relevant": 3,
    }

    def _to_int(v: Any) -> int:
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v in _str_label_map:
            return _str_label_map[v]
        return int(v)

    labels_judge = [_to_int(j.get("label", 0)) for j, _ in pairs]
    # Human records: prefer numeric 'relevance' over potentially-string 'label'
    labels_human = [
        _to_int(h.get("relevance") if h.get("relevance") is not None else h.get("label", 0))
        for _, h in pairs
    ]

    n = len(pairs)
    exact = sum(lj == lh for lj, lh in zip(labels_judge, labels_human, strict=False)) / n
    within1 = sum(abs(lj - lh) <= 1 for lj, lh in zip(labels_judge, labels_human, strict=False)) / n
    kappa = _qwk(labels_human, labels_judge)
    conf_matrix = _confusion_matrix(labels_human, labels_judge)

    false_high = [
        LabelMismatch(
            query_id=j.get("query_id", ""),
            dataset_id=j.get("dataset_id", ""),
            judge_label=lj,
            human_label=lh,
            judge_rationale=str(j.get("rationale_short", "")),
        )
        for (j, h), lj, lh in zip(pairs, labels_judge, labels_human, strict=False)
        if lj > lh
    ]
    false_low = [
        LabelMismatch(
            query_id=j.get("query_id", ""),
            dataset_id=j.get("dataset_id", ""),
            judge_label=lj,
            human_label=lh,
            judge_rationale=str(j.get("rationale_short", "")),
        )
        for (j, h), lj, lh in zip(pairs, labels_judge, labels_human, strict=False)
        if lj < lh
    ]

    # Breakdowns
    def _breakdown(key_fn: Any) -> dict[str, dict[str, float]]:
        buckets: dict[str, list[tuple[int, int]]] = {}
        for (j, h), lj, lh in zip(pairs, labels_judge, labels_human, strict=False):
            k = key_fn(j, h)
            if k:
                buckets.setdefault(k, []).append((lj, lh))
        result: dict[str, dict[str, float]] = {}
        for k, items in buckets.items():
            ni = len(items)
            result[k] = {
                "n": float(ni),
                "exact": sum(a == b for a, b in items) / ni,
                "within1": sum(abs(a - b) <= 1 for a, b in items) / ni,
            }
        return result

    by_intent = _breakdown(
        lambda j, h: str(h.get("intent") or j.get("query_intent") or "unknown")
    )
    by_modality = _breakdown(
        lambda j, h: (str((h.get("modalities") or ["unknown"])[0]) if h.get("modalities") else "unknown")
    )
    by_failure_mode = _breakdown(
        lambda j, h: (str((j.get("failure_modes") or ["none"])[0]) if j.get("failure_modes") else "none")
    )

    return CalibrationReport(
        n_pairs=n,
        exact_agreement=exact,
        agreement_within_1=within1,
        quadratic_weighted_kappa=kappa,
        confusion_matrix=conf_matrix,
        false_high_examples=false_high,
        false_low_examples=false_low,
        by_query_intent=by_intent,
        by_modality=by_modality,
        by_failure_mode=by_failure_mode,
    )
