"""Affordance validators for NWB and BIDS data formats.

These validators inspect actual file contents to verify that datasets
support the analysis affordances predicted from metadata.
"""

from neural_search.affordances.validators.bids_validator import (
    BIDSValidationResult,
    BIDSValidator,
    validate_bids_affordances,
)
from neural_search.affordances.validators.nwb_validator import (
    NWBValidationResult,
    NWBValidator,
    validate_nwb_affordances,
)

__all__ = [
    "NWBValidator",
    "NWBValidationResult",
    "validate_nwb_affordances",
    "BIDSValidator",
    "BIDSValidationResult",
    "validate_bids_affordances",
]
