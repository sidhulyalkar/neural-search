"""Experimental latent neural/behavioral state search module.

This module provides a scaffold for future neural-state search capabilities.
The goal is to complement ontology/provenance search with latent representations
derived from neural activity patterns, behavioral summaries, and task states.

Current status: Scaffold only - no trained models included.

Future capabilities:
- Trial-aligned event histograms
- Neural summary statistics (firing rates, synchrony, etc.)
- Behavior event transition summaries
- Task-state labels
- Session-level QC vectors
"""

from neural_search.latent.embedding_schema import (
    FeatureSummary,
    LatentIndex,
    LatentSearchResult,
    SessionFeatures,
)
from neural_search.latent.summary_features import extract_session_features

__all__ = [
    "FeatureSummary",
    "LatentIndex",
    "LatentSearchResult",
    "SessionFeatures",
    "extract_session_features",
]
