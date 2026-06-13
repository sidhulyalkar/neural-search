"""Affordance Validation v2: precision/recall vs ground-truth labels.

Improvements over v1:
- Structured ground-truth label ingestion
- Per-affordance precision/recall/F1
- Confusion table (TP, FP, FN)
- Machine-readable JSON output
- Human-readable Markdown report
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SyntheticCard:
    """Lightweight dataset card for validation (no heavy schema dependency)."""
    dataset_id: str
    predicted_affordances: list[str] = field(default_factory=list)
    modalities: list[str] = field(default_factory=list)
    has_trials: bool = False
    has_timestamps: bool = False


@dataclass
class GroundTruthLabel:
    """Ground-truth label for one affordance on one dataset."""
    dataset_id: str
    affordance: str
    supported: bool


@dataclass
class ValidationReport:
    n_datasets: int
    n_labeled: int
    n_unlabeled: int
    per_affordance_precision: dict[str, float]
    per_affordance_recall: dict[str, float]
    per_affordance_f1: dict[str, float]
    confusion_table: dict[str, dict[str, int]]
    coverage_by_affordance: dict[str, int]
    notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "# Affordance Validation v2 Report\n",
            f"- Datasets evaluated: {self.n_datasets}",
            f"- Datasets with ground-truth labels: {self.n_labeled}",
            f"- Datasets without labels (unknown): {self.n_unlabeled}",
            "",
            "## Per-Affordance Precision/Recall/F1\n",
            "| Affordance | Precision | Recall | F1 | TP | FP | FN |",
            "|------------|-----------|--------|----|----|----|-----|",
        ]
        for aff in sorted(self.per_affordance_precision):
            p = self.per_affordance_precision[aff]
            r = self.per_affordance_recall.get(aff, 0.0)
            f = self.per_affordance_f1.get(aff, 0.0)
            ct = self.confusion_table.get(aff, {})
            lines.append(
                f"| {aff} | {p:.3f} | {r:.3f} | {f:.3f} | "
                f"{ct.get('tp', 0)} | {ct.get('fp', 0)} | {ct.get('fn', 0)} |"
            )
        if self.notes:
            lines += ["", "## Notes"] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_datasets": self.n_datasets,
            "n_labeled": self.n_labeled,
            "n_unlabeled": self.n_unlabeled,
            "per_affordance_precision": self.per_affordance_precision,
            "per_affordance_recall": self.per_affordance_recall,
            "per_affordance_f1": self.per_affordance_f1,
            "confusion_table": self.confusion_table,
            "coverage_by_affordance": self.coverage_by_affordance,
            "notes": self.notes,
        }


class AffordanceValidationV2:
    """Validates predicted affordances against ground-truth labels."""

    def __init__(self, cards: list[SyntheticCard], labels: list[GroundTruthLabel]):
        self.cards = cards
        self._labels: dict[tuple[str, str], bool] = {
            (label.dataset_id, label.affordance): label.supported for label in labels
        }
        self._labeled_datasets = {label.dataset_id for label in labels}
        self._card_map = {c.dataset_id: c for c in cards}

    def run(self) -> ValidationReport:
        counts: dict[str, dict[str, int]] = {}

        # Count TP and FP from predictions
        for card in self.cards:
            for aff in card.predicted_affordances:
                key = (card.dataset_id, aff)
                if key not in self._labels:
                    continue
                truth = self._labels[key]
                bucket = counts.setdefault(aff, {"tp": 0, "fp": 0, "fn": 0})
                if truth:
                    bucket["tp"] += 1
                else:
                    bucket["fp"] += 1

        # Count FN: labeled supported but not predicted
        for (ds_id, aff), supported in self._labels.items():
            if not supported:
                continue
            card = self._card_map.get(ds_id)
            if card is None:
                continue
            if aff not in card.predicted_affordances:
                counts.setdefault(aff, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1

        precision: dict[str, float] = {}
        recall: dict[str, float] = {}
        f1: dict[str, float] = {}

        for aff, ct in counts.items():
            tp, fp, fn = ct["tp"], ct["fp"], ct["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precision[aff] = p
            recall[aff] = r
            f1[aff] = f

        coverage: dict[str, int] = {}
        for card in self.cards:
            for aff in card.predicted_affordances:
                coverage[aff] = coverage.get(aff, 0) + 1

        labeled_ds = {c.dataset_id for c in self.cards if c.dataset_id in self._labeled_datasets}

        return ValidationReport(
            n_datasets=len(self.cards),
            n_labeled=len(labeled_ds),
            n_unlabeled=len(self.cards) - len(labeled_ds),
            per_affordance_precision=precision,
            per_affordance_recall=recall,
            per_affordance_f1=f1,
            confusion_table=counts,
            coverage_by_affordance=coverage,
        )


def run_validation(
    cards: list[SyntheticCard],
    labels: list[GroundTruthLabel],
    out_path: Path | None = None,
    json_out_path: Path | None = None,
) -> ValidationReport:
    """Run affordance validation and optionally write reports."""
    validator = AffordanceValidationV2(cards, labels)
    report = validator.run()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.to_markdown(), encoding="utf-8")
    if json_out_path:
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
    return report
