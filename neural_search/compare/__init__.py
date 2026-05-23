"""Dataset comparison module for side-by-side analysis."""

from neural_search.compare.comparison import (
    ComparisonResult,
    DatasetComparisonItem,
    compare_datasets,
    generate_comparison_markdown,
)

__all__ = [
    "compare_datasets",
    "generate_comparison_markdown",
    "ComparisonResult",
    "DatasetComparisonItem",
]
