"""Portable synthetic / NPZ sources plus an opt-in, public MOABB adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neural_search.decoder_generalization.study import TrialDataset


def synthetic_dataset(*, seed: int = 17) -> TrialDataset:
    """Deterministic toy epochs with subject and session shifts.

    Never treat this fixture as evidence for any empirical neuroscience claim.
    """
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[str] = []
    subjects: list[str] = []
    sessions: list[str] = []
    t = np.linspace(0, 1, 128, endpoint=False)
    carrier = np.sin(2 * np.pi * 11 * t)
    for subject in range(4):
        for session in range(2):
            for label in ("left", "right"):
                for _ in range(36):
                    trial = rng.normal(0, 0.75, (6, 128))
                    channel = 0 if label == "left" else 1
                    trial[channel] += (1.5 + 0.15 * subject) * carrier
                    if session:
                        trial = trial * np.array([0.75, 1.3, 1.0, 0.9, 1.1, 1.0])[:, None]
                    xs.append(trial)
                    ys.append(label)
                    subjects.append(f"s{subject + 1}")
                    sessions.append(f"session{session + 1}")
    return TrialDataset(
        X=np.stack(xs),
        y=np.asarray(ys),
        subject=np.asarray(subjects),
        session=np.asarray(sessions),
        source=f"synthetic_shift_v1_seed_{seed}",
    )


def load_npz(path: str | Path) -> TrialDataset:
    """Load an explicit, non-pickled epoch bundle; never trust NPZ as gold evidence."""
    file = Path(path)
    with np.load(file, allow_pickle=False) as data:
        required = {"X", "y", "subject", "session"}
        if missing := required.difference(data.files):
            raise ValueError(f"Missing NPZ fields: {sorted(missing)}")
        x = np.asarray(data["X"])
        y = np.asarray(data["y"])
        subject = np.asarray(data["subject"])
        session = np.asarray(data["session"])
    return TrialDataset(x, y, subject, session, source=f"npz:{file.name}")


def load_bnci2014_001(subjects: list[int]) -> TrialDataset:
    """Load public BCI Competition IV 2a via MOABB (network access may be needed).

    Uses the two-class left/right motor imagery paradigm to match all labels
    across subjects and sessions. MOABB handles epoch extraction; downstream
    model scaling is strictly fit on each training partition.
    """
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("Provide a nonempty list of unique subject IDs")
    if any(n < 1 or n > 9 for n in subjects):
        raise ValueError("BNCI2014_001 subject IDs must be between 1 and 9")
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import LeftRightImagery
    except ImportError as exc:
        raise RuntimeError(
            "MOABB is optional. Install with: pip install -e '.[decoder]'"
        ) from exc
    paradigm = LeftRightImagery()
    x, labels, metadata = paradigm.get_data(
        dataset=BNCI2014_001(), subjects=subjects
    )
    for key in ("subject", "session"):
        if key not in metadata.columns:
            raise ValueError(f"MOABB metadata missing required group: {key}")
    return TrialDataset(
        X=np.asarray(x, dtype=float),
        y=np.asarray(labels).astype(str),
        subject=metadata["subject"].astype(str).to_numpy(),
        session=metadata["session"].astype(str).to_numpy(),
        source="moabb:BNCI2014_001:LeftRightImagery",
    )
