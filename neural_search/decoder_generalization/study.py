"""Conservative, provenance-aware evaluation of neural decoder transfer.

This is a transparent log-variance + regularized LDA baseline. Diagnostics
identify associations, not causal explanations of neural mechanisms. No
preprocessing statistics are learned from a held-out partition.
"""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

PROTOCOLS = ("within_session", "cross_session", "cross_subject")


@dataclass(frozen=True)
class TrialDataset:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    source: str

    def __post_init__(self) -> None:
        x = np.asarray(self.X)
        n = len(x)
        if x.ndim != 3 or n < 4 or min(x.shape[1:]) < 2:
            raise ValueError("X must contain trials x channels x timepoints")
        if not np.issubdtype(x.dtype, np.number) or not np.isfinite(x).all():
            raise ValueError("X must be finite numeric epochs")
        for name in ("y", "subject", "session"):
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or len(value) != n:
                raise ValueError(f"{name} must contain one value per trial")
            if any(str(item).strip() == "" for item in value):
                raise ValueError(f"{name} must not contain empty identifiers")
        if len(set(map(str, self.y))) < 2:
            raise ValueError("At least two class labels are required")
        if not self.source.strip():
            raise ValueError("Source provenance is required")


@dataclass(frozen=True)
class Fold:
    protocol: str
    key: str
    train: np.ndarray
    test: np.ndarray


def build_folds(
    dataset: TrialDataset,
    protocols: tuple[str, ...] = PROTOCOLS,
    *,
    seed: int = 17,
) -> list[Fold]:
    """Generate nested-group-safe, deterministic, subject-aware partitions."""
    if any(p not in PROTOCOLS for p in protocols):
        raise ValueError(f"Unknown protocol; choose from {PROTOCOLS}")
    y = np.asarray(dataset.y).astype(str)
    subject = np.asarray(dataset.subject).astype(str)
    session = np.asarray(dataset.session).astype(str)
    folds: list[Fold] = []

    if "within_session" in protocols:
        for s in sorted(set(subject)):
            for t in sorted(set(session[subject == s])):
                indices = np.flatnonzero((subject == s) & (session == t))
                labels = sorted(set(y[indices]))
                if len(labels) < 2 or any(np.sum(y[indices] == c) < 3 for c in labels):
                    continue
                random = np.random.default_rng(seed)
                splits = [list() for _ in range(3)]
                for label in labels:
                    class_indices = random.permutation(indices[y[indices] == label])
                    for k, portion in enumerate(np.array_split(class_indices, 3)):
                        splits[k].extend(portion.tolist())
                for k, test_indices in enumerate(splits):
                    test = np.asarray(sorted(test_indices), dtype=int)
                    train = np.setdiff1d(indices, test)
                    folds.append(Fold("within_session", f"{s}/{t}/fold{k}", train, test))

    if "cross_session" in protocols:
        for s in sorted(set(subject)):
            subject_indices = np.flatnonzero(subject == s)
            sessions = sorted(set(session[subject_indices]))
            if len(sessions) < 2:
                continue
            for t in sessions:
                test = subject_indices[session[subject_indices] == t]
                train = subject_indices[session[subject_indices] != t]
                folds.append(Fold("cross_session", f"{s}/{t}", train, test))

    if "cross_subject" in protocols:
        subjects = sorted(set(subject))
        if len(subjects) >= 3:
            for s in subjects:
                test = np.flatnonzero(subject == s)
                train = np.flatnonzero(subject != s)
                folds.append(Fold("cross_subject", s, train, test))
    return folds


def _fingerprints(x: np.ndarray) -> list[str]:
    return [
        hashlib.sha256(np.ascontiguousarray(trial).tobytes()).hexdigest()
        for trial in x
    ]


def _features(x: np.ndarray) -> np.ndarray:
    """Channel-wise log-variance, a simple motor-imagery baseline feature."""
    return np.log(np.maximum(np.var(x, axis=-1), 1e-12))


def _fit_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a regularized LDA using TRAIN-ONLY feature scaling and covariance."""
    a = _features(train_x)
    b = _features(test_x)
    mean = np.mean(a, axis=0)
    scale = np.std(a, axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    a = (a - mean) / scale
    b = (b - mean) / scale
    labels = np.unique(train_y)
    pooled = np.zeros((a.shape[1], a.shape[1]))
    class_means = []
    for label in labels:
        class_features = a[train_y == label]
        center = np.mean(class_features, axis=0)
        class_means.append(center)
        centered = class_features - center
        pooled += centered.T @ centered
    pooled /= max(len(a) - len(labels), 1)
    diag = np.diag(np.diag(pooled))
    shrinkage = 0.25 * diag + 0.75 * pooled
    ridge = 1e-3 * np.eye(a.shape[1])
    inverse = np.linalg.pinv(shrinkage + ridge)
    centers = np.stack(class_means)
    scores = b @ inverse @ centers.T
    scores -= 0.5 * np.einsum("ij,jk,ik->i", centers, inverse, centers)
    prediction = labels[np.argmax(scores, axis=1)]
    standardized_shift = float(np.mean(np.abs(np.mean(b, axis=0))))
    return prediction, labels, standardized_shift


def _score(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    confusion = [
        [int(np.sum((y_true == a) & (y_pred == b))) for b in labels]
        for a in labels
    ]
    recalls = [
        confusion[k][k] / sum(row)
        for k, row in enumerate(confusion)
        if (total := sum(row)) > 0
    ]
    return {
        "balanced_accuracy": round(float(np.mean(recalls)), 6),
        "accuracy": round(float(np.mean(y_true == y_pred)), 6),
        "classes": labels.tolist(),
        "confusion_matrix": confusion,
        "n_test": int(len(y_true)),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for protocol in PROTOCOLS:
        values = [
            r["balanced_accuracy"]
            for r in rows
            if r["protocol"] == protocol and r["status"] == "evaluated"
        ]
        if values:
            out[protocol] = {
                "folds": len(values),
                "mean_balanced_accuracy": round(float(np.mean(values)), 6),
                "std_fold_balanced_accuracy": round(float(np.std(values)), 6),
            }
        else:
            out[protocol] = {"folds": 0, "status": "insufficient_evaluable_groups"}
    return out


def run_study(
    dataset: TrialDataset,
    protocols: tuple[str, ...] = PROTOCOLS,
    *,
    seed: int = 17,
) -> dict[str, Any]:
    """Measure transfer gaps; never infer a biological cause from drift alone."""
    started = time.perf_counter()
    x = np.asarray(dataset.X)
    y = np.asarray(dataset.y).astype(str)
    subject = np.asarray(dataset.subject).astype(str)
    session = np.asarray(dataset.session).astype(str)
    trial_hashes = _fingerprints(x)
    rows: list[dict[str, Any]] = []
    for fold in build_folds(dataset, protocols, seed=seed):
        if np.intersect1d(fold.train, fold.test).size:
            raise ValueError("Split leakage: overlapping trial indices")
        if set(trial_hashes[k] for k in fold.train) & set(
            trial_hashes[k] for k in fold.test
        ):
            raise ValueError(
                f"Split leakage: identical trial content across train/test in {fold.key}"
            )
        train_y = y[fold.train]
        test_y = y[fold.test]
        record: dict[str, Any] = {
            "protocol": fold.protocol,
            "key": fold.key,
            "n_train": len(fold.train),
            "n_test": len(fold.test),
            "train_subjects": sorted(set(subject[fold.train])),
            "test_subjects": sorted(set(subject[fold.test])),
            "train_sessions": sorted(set(session[fold.train])),
            "test_sessions": sorted(set(session[fold.test])),
        }
        if (
            len(set(train_y)) < 2
            or len(set(test_y)) < 2
            or not set(test_y).issubset(set(train_y))
        ):
            record.update(
                status="skipped",
                reason="Inadequate train/test class coverage for a comparable score",
            )
            rows.append(record)
            continue
        predicted, labels, shift = _fit_predict(x[fold.train], train_y, x[fold.test])
        record.update(
            status="evaluated",
            feature_shift_z=round(shift, 6),
            **_score(test_y, predicted, labels),
        )
        rows.append(record)

    baseline_by_subject: dict[str, list[float]] = {}
    for row in rows:
        if row["status"] == "evaluated" and row["protocol"] == "within_session":
            key = row["key"].split("/")[0]
            baseline_by_subject.setdefault(key, []).append(row["balanced_accuracy"])
    paired_gaps = []
    for row in rows:
        if row["status"] == "evaluated" and row["protocol"] == "cross_session":
            key = row["key"].split("/")[0]
            if key in baseline_by_subject:
                paired_gaps.append({
                    "held_out": row["key"],
                    "within_session_reference": round(
                        float(np.mean(baseline_by_subject[key])), 6
                    ),
                    "cross_session_score": row["balanced_accuracy"],
                    "performance_drop": round(
                        float(np.mean(baseline_by_subject[key]))
                        - row["balanced_accuracy"], 6
                    ),
                })
    digest = hashlib.sha256()
    for item in trial_hashes:
        digest.update(item.encode("ascii"))
    for values in (y, subject, session):
        digest.update("\x00".join(values).encode("utf-8"))
    return {
        "schema_version": "1.0",
        "dataset": {
            "source": dataset.source,
            "n_trials": len(x),
            "n_channels": x.shape[1],
            "n_timepoints": x.shape[2],
            "subjects": len(set(subject)),
            "fingerprint_sha256": digest.hexdigest(),
        },
        "model": {
            "name": "logvar_shrinkage_lda",
            "shrinkage": 0.25,
            "features": "per-channel log variance",
            "scaling": "fit on training partition only",
        },
        "seed": seed,
        "protocols": list(protocols),
        "results": rows,
        "summary": _aggregate(rows),
        "paired_within_to_cross_session_gaps": paired_gaps,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "evidence_status": "diagnostic_association_only",
        "cautions": [
            "Synthetic results do not establish performance on real neural data.",
            "Within-session folds, cross-session folds, and cross-subject folds differ in their training samples; gaps are descriptive, not causal.",
            "Feature shift is not a validated neural mechanism or a distribution-shift significance test.",
            "Inspect subject-level variation and label balance before interpreting aggregate results.",
        ],
    }
