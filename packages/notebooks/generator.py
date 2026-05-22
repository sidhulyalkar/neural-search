"""Generate starter Jupyter notebooks for datasets."""

from typing import Optional

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import DatasetRecord, DatasetCard, DataStandard

from .nwb_template import NWBNotebookTemplate
from .bids_template import BIDSNotebookTemplate


class NotebookGenerator:
    """
    Generate starter Jupyter notebooks for datasets.

    Creates notebooks with:
    - Installation and imports
    - Data loading
    - Metadata inspection
    - Basic visualizations
    - TODO cells for analysis
    """

    def __init__(self):
        self.nwb_template = NWBNotebookTemplate()
        self.bids_template = BIDSNotebookTemplate()

    def generate(
        self,
        dataset: DatasetRecord,
        card: Optional[DatasetCard] = None,
        nwb_path: Optional[str] = None,
    ) -> nbformat.NotebookNode:
        """
        Generate a starter notebook for a dataset.

        Args:
            dataset: The dataset record.
            card: Optional dataset card for additional context.
            nwb_path: Optional specific NWB file path to load.

        Returns:
            A Jupyter notebook as NotebookNode.
        """
        # Determine data standard
        data_standard = dataset.data_standard
        if card and card.data_standard:
            data_standard = card.data_standard

        # Generate appropriate notebook
        if data_standard == DataStandard.NWB:
            return self.nwb_template.generate(dataset, card, nwb_path)
        elif data_standard == DataStandard.BIDS:
            return self.bids_template.generate(dataset, card)
        else:
            # Default to NWB if from DANDI
            if dataset.source.value == "dandi":
                return self.nwb_template.generate(dataset, card, nwb_path)
            else:
                return self.bids_template.generate(dataset, card)

    def save(
        self,
        notebook: nbformat.NotebookNode,
        path: str,
    ) -> None:
        """Save notebook to file."""
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

    def to_string(self, notebook: nbformat.NotebookNode) -> str:
        """Convert notebook to JSON string."""
        return nbformat.writes(notebook)
