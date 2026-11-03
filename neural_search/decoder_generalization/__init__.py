"""Leakage-aware, reproducible neural decoder generalization studies."""

from neural_search.decoder_generalization.study import (
    TrialDataset,
    build_folds,
    run_study,
)

__all__ = ["TrialDataset", "build_folds", "run_study"]
