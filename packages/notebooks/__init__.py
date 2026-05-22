"""
Neural Search Notebooks Package

Generate starter Jupyter notebooks for NWB and BIDS datasets.
"""

from .generator import NotebookGenerator
from .nwb_template import NWBNotebookTemplate
from .bids_template import BIDSNotebookTemplate

__all__ = ["NotebookGenerator", "NWBNotebookTemplate", "BIDSNotebookTemplate"]
