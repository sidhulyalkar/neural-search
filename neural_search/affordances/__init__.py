"""Analysis Affordance Validation.

This module provides the affordance registry and validation framework
for determining whether datasets can support specific analyses.

Usage:
    from neural_search.affordances import (
        get_affordance,
        list_affordances,
        validate_affordance,
        detect_features_from_metadata,
    )

    # Get affordance requirements
    req = get_affordance("choice_decoding")

    # Detect features from a dataset
    features = detect_features_from_metadata(dataset_dict)

    # Validate affordance support
    result = validate_affordance("choice_decoding", features)
    print(f"Supported: {result.supported}, Level: {result.support_level}")
"""

from neural_search.affordances.registry import (
    AFFORDANCE_REGISTRY,
    DataFormat,
    DatasetFeatures,
    SupportLevel,
    detect_features_from_metadata,
    get_affordance,
    get_all_affordances,
    list_affordances,
    validate_affordance,
    validate_all_affordances,
)

__all__ = [
    "AFFORDANCE_REGISTRY",
    "DataFormat",
    "DatasetFeatures",
    "SupportLevel",
    "detect_features_from_metadata",
    "get_affordance",
    "get_all_affordances",
    "list_affordances",
    "validate_affordance",
    "validate_all_affordances",
]
