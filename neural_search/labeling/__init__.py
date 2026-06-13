"""Human Relevance Labeling Workflow.

This module provides infrastructure for collecting and managing human relevance
labels for search evaluation.

Components:
- LabelingSession: Manages a labeling session with query-result pairs
- LabelingUI: Command-line interface for labeling
- LabelStorage: Persistent storage for labels
- ActiveLearner: Sample selection for efficient labeling
- Agreement: Multi-annotator agreement metrics (Cohen's kappa, Fleiss' kappa)
"""

from neural_search.labeling.agreement import (
    AnnotatorStats,
    MultiAnnotatorReport,
    PairwiseAgreement,
    compute_agreement_report,
    compute_cohens_kappa,
    compute_fleiss_kappa,
    compute_krippendorff_alpha,
)
from neural_search.labeling.cli import (
    run_labeling_cli,
)
from neural_search.labeling.session import (
    LabelingSession,
    RelevanceGrade,
    RelevanceLabel,
)
from neural_search.labeling.storage import (
    LabelStorage,
    load_labels,
    save_labels,
)

__all__ = [
    "LabelingSession",
    "RelevanceLabel",
    "RelevanceGrade",
    "LabelStorage",
    "load_labels",
    "save_labels",
    "run_labeling_cli",
    # Multi-annotator agreement
    "compute_agreement_report",
    "compute_cohens_kappa",
    "compute_fleiss_kappa",
    "compute_krippendorff_alpha",
    "MultiAnnotatorReport",
    "AnnotatorStats",
    "PairwiseAgreement",
]
