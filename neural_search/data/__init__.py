"""Data access and streaming module.

This module provides efficient access to neuroscience data archives:
- DANDI streaming via remfile
- Metadata extraction without full downloads
- Neural signature extraction from remote files
"""

from neural_search.data.dandi_streaming import (
    DANDIAsset,
    extract_nwb_metadata_streaming,
    extract_signature_streaming,
    get_dandiset_metadata,
    list_dandiset_assets,
    scan_dandiset_for_affordances,
    scan_multiple_dandisets,
    stream_nwb_file,
)

__all__ = [
    "DANDIAsset",
    "extract_nwb_metadata_streaming",
    "extract_signature_streaming",
    "get_dandiset_metadata",
    "list_dandiset_assets",
    "scan_dandiset_for_affordances",
    "scan_multiple_dandisets",
    "stream_nwb_file",
]
