"""File inspection tools for evidence-backed corpus claims."""

from neural_search.file_inspection.claims import FileInspectionClaim
from neural_search.file_inspection.inspect import (
    inspect_dataset_files,
    write_claims_jsonl,
)

__all__ = ["FileInspectionClaim", "inspect_dataset_files", "write_claims_jsonl"]
