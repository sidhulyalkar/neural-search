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

PROTOCOLS = ("within_session", "within_session_runout", "cross_session", "cross_subject")


@dataclass(frozen=True)
class TrialDataset:
    X: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    source: str
    run: np.ndarray | None = None

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
        if self.run is not None:
            runs = np.asarray(self.run)
            if runs.ndim != 1 or len(runs) != n or any(not str(v).strip() for v in runs):
                raise ValueError("run must contain one nonempty ID per trial")
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
                splits = [[] for _ in range(3)]
                for label in labels:
                    class_indices = random.permutation(indices[y[indices] == label])
                    for k, portion in enumerate(np.array_split(class_indices, 3)):
                        splits[k].extend(portion.tolist())
                for k, test_indices in enumerate(splits):
                    test = np.asarray(sorted(test_indices), dtype=int)
                    train = np.setdiff1d(indices, test)
                    folds.append(Fold("within_session", f"{s}/{t}/fold{k}", train, test))

    if "within_session_runout" in protocols and dataset.run is not None:
        run_ids = np.asarray(dataset.run).astype(str)
        for s in sorted(set(subject)):
            for t in sorted(set(session[subject == s])):
                group = np.flatnonzero((subject == s) & (session == t))
                runs = sorted(set(run_ids[group]))
                if len(runs) < 2:
                    continue
                for r in runs:
                    test = group[run_ids[group] == r]
                    train = group[run_ids[group] != r]
                    folds.append(Fold(
                        "within_session_runout", f"{s}/{t}/{r}", train, test
                    ))

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


def _conditional_shift(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> float:
    """Label-conditioned log-variance shift for POST-HOC diagnosis only.

    Test labels are used exclusively in this diagnostic, NEVER by the
    classifier or in any train-fitted preprocessing statistic.
    """
    train_features = _features(train_x)
    test_features = _features(test_x)
    scale = np.std(train_features, axis=0)
    scale = np.where(scale < 1e-9, 1.0, scale)
    comparisons = []
    for label in sorted(set(train_y) & set(test_y)):
        a = np.mean(train_features[train_y == label], axis=0)
        b = np.mean(test_features[test_y == label], axis=0)
        comparisons.append(float(np.mean(np.abs((b - a) / scale))))
    return float(np.mean(comparisons)) if comparisons else 0.0


def _score(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    confusion = [
        [int(np.sum((y_true == a) & (y_pred == b))) for b in labels]
        for a in labels
    ]
    recalls = [
        confusion[k][k] / sum(row)
        for k, row in enumerate(confusion)
        if sum(row) > 0
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


def _subject_cluster_bootstrap(
    paired_gaps: list[dict[str, Any]], *, seed: int, n_boot: int = 2000
) -> dict[str, Any]:
    """Bootstrap independent subjects, NOT correlated sessions or CV folds."""
    grouped: dict[str, list[float]] = {}
    for row in paired_gaps:
        grouped.setdefault(row["subject"], []).append(row["performance_drop"])
    per_subject = {
        subject: {
            "n_sessions": len(gaps),
            "mean_gap": round(float(np.mean(gaps)), 6),
        }
        for subject, gaps in sorted(grouped.items())
    }
    means = np.asarray(
        [float(np.mean(gaps)) for _, gaps in sorted(grouped.items())],
        dtype=float,
    )
    if len(means) < 3:
        return {
            "status": "insufficient_independent_subjects",
            "n_subjects": len(means),
            "per_subject": per_subject,
            "mean_gap": round(float(np.mean(means)), 6) if len(means) else None,
            "ci95": None,
            "note": "No interval with fewer than three independent subjects.",
        }
    rng = np.random.default_rng(seed + 4711)
    draws = rng.choice(means, size=(n_boot, len(means)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {
        "status": "descriptive_cluster_bootstrap",
        "n_subjects": len(means),
        "n_boot": n_boot,
        "per_subject": per_subject,
        "mean_gap": round(float(np.mean(means)), 6),
        "ci95": [round(float(lo), 6), round(float(hi), 6)],
        "note": (
            "Percentile bootstrap across subjects; not a causal effect, "
            "and too few subjects can give misleadingly narrow intervals."
        ),
    }


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
    run = None if dataset.run is None else np.asarray(dataset.run).astype(str)
    trial_hashes = _fingerprints(x)
    rows: list[dict[str, Any]] = []
    for fold in build_folds(dataset, protocols, seed=seed):
        if np.intersect1d(fold.train, fold.test).size:
            raise ValueError("Split leakage: overlapping trial indices")
        if {trial_hashes[k] for k in fold.train} & {
            trial_hashes[k] for k in fold.test
        }:
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
            "train_runs": [] if run is None else sorted(set(run[fold.train])),
            "test_runs": [] if run is None else sorted(set(run[fold.test])),
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
            class_conditional_shift_z=round(
                _conditional_shift(x[fold.train], train_y, x[fold.test], test_y),
                6,
            ),
            **_score(test_y, predicted, labels),
        )
        rows.append(record)

    # Only compare a cross-session result with a within-session baseline
    # evaluated on the SAME target session, not a different session. Prefer
    # run-blocked baselines where recording-run IDs are available.
    baseline_by_session: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if (
            row["status"] == "evaluated"
            and row["protocol"] in {"within_session", "within_session_runout"}
        ):
            key = "/".join(row["key"].split("/")[:2])
            baseline_by_session.setdefault(key, {}).setdefault(
                row["protocol"], []
            ).append(row["balanced_accuracy"])
    paired_gaps: list[dict[str, Any]] = []
    for row in rows:
        if row["status"] == "evaluated" and row["protocol"] == "cross_session":
            refs = baseline_by_session.get(row["key"], {})
            reference_protocol = (
                "within_session_runout"
                if refs.get("within_session_runout")
                else "within_session"
            )
            values = refs.get(reference_protocol, [])
            if values:
                reference_score = float(np.mean(values))
                paired_gaps.append({
                    "held_out": row["key"],
                    "subject": row["key"].split("/")[0],
                    "reference_protocol": reference_protocol,
                    "within_session_reference": round(reference_score, 6),
                    "cross_session_score": row["balanced_accuracy"],
                    "performance_drop": round(
                        reference_score - row["balanced_accuracy"], 6
                    ),
                })
    digest = hashlib.sha256()
    for item in trial_hashes:
        digest.update(item.encode("ascii"))
    for values in (y, subject, session, *([run] if run is not None else [])):
        digest.update("\x00".join(values).encode("utf-8"))
    return {
        "schema_version": "1.0",
        "dataset": {
            "source": dataset.source,
            "n_trials": len(x),
            "n_channels": x.shape[1],
            "n_timepoints": x.shape[2],
            "subjects": len(set(subject)),
            "has_run_metadata": run is not None,
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
        "subject_clustered_uncertainty": _subject_cluster_bootstrap(
            paired_gaps, seed=seed
        ),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "evidence_status": "diagnostic_association_only",
        "cautions": [
            "Synthetic results do not establish performance on real neural data.",
            "Within-session, run-heldout, cross-session and cross-subject training sizes differ; gaps are descriptive, not causal.",
            "Naive within-session trial CV may overestimate transfer due to temporal correlation; prefer run-heldout where available.",
            "Feature shifts are descriptive, not validated neural mechanisms or significance tests.",
            "Class-conditional shift uses test labels AFTER prediction solely for retrospective diagnosis.",
            "Inspect subject-level variation and label balance before interpreting aggregate results.",
        ],
    }
