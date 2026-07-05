"""Aperiodic spectral phenotype / spectral parameterization reanalysis layer.

This package determines whether normalized dataset records can support
aperiodic ("1/f") spectral parameterization analyses, optionally computes
standardized spectral features (periodic + aperiodic components) from
power spectra, attaches QC, and exports the results into the knowledge
graph and search affordance pipeline.

Scientific caution: an aperiodic exponent / spectral slope is a descriptive
spectral statistic. It is NOT, by itself, a validated direct measurement of
excitation/inhibition (E/I) balance or any single cellular or circuit
mechanism. See docs/aperiodic_reanalysis.md for caveats that must accompany
any interpretation of these features.
"""

from __future__ import annotations

from neural_search.spectral.eligibility import detect_aperiodic_eligibility
from neural_search.spectral.schemas import (
    AperiodicEligibility,
    PeriodicPeak,
    SpectralEstimate,
    SpectralFeatureBundle,
    SpectralRunConfig,
)

__all__ = [
    "AperiodicEligibility",
    "PeriodicPeak",
    "SpectralEstimate",
    "SpectralFeatureBundle",
    "SpectralRunConfig",
    "detect_aperiodic_eligibility",
]
