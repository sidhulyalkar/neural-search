"""Dataset comparison module for side-by-side analysis."""

from neural_search.compare.comparison import (
    compare_datasets,
    generate_comparison_markdown,
    ComparisonResult,
    DatasetComparisonItem,
)

__all__ = [
    "compare_datasets",
    "generate_comparison_markdown",
    "ComparisonResult",
    "DatasetComparisonItem",
]