"""BIDS notebook template."""

from typing import Optional

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import DatasetRecord, DatasetCard


class BIDSNotebookTemplate:
    """Generate BIDS starter notebooks."""

    def generate(
        self,
        dataset: DatasetRecord,
        card: Optional[DatasetCard] = None,
    ) -> nbformat.NotebookNode:
        """Generate a BIDS exploration notebook."""
        cells = []

        # Title cell
        cells.append(self._title_cell(dataset, card))

        # Installation cell
        cells.append(self._install_cell())

        # Imports cell
        cells.append(self._imports_cell())

        # Load BIDS cell
        cells.append(self._load_bids_cell(dataset))

        # Dataset description cell
        cells.append(self._description_cell())

        # Participants cell
        cells.append(self._participants_cell())

        # Sessions cell
        cells.append(self._sessions_cell())

        # Data files cell
        cells.append(self._data_files_cell())

        # Events cell
        cells.append(self._events_cell())

        # Basic visualization
        cells.append(self._plot_cell())

        # TODO cells
        cells.extend(self._todo_cells(card))

        # Create notebook
        nb = new_notebook(cells=cells)
        nb.metadata["kernelspec"] = {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }

        return nb

    def _title_cell(
        self, dataset: DatasetRecord, card: Optional[DatasetCard]
    ) -> nbformat.NotebookNode:
        """Create title markdown cell."""
        lines = [
            f"# {dataset.title}",
            "",
            f"**Source:** {dataset.source.value.upper()}",
            f"**Dataset ID:** {dataset.source_id}",
            "**Format:** BIDS",
        ]
        if dataset.doi:
            lines.append(f"**DOI:** {dataset.doi}")
        if dataset.url:
            lines.append(f"**URL:** [{dataset.url}]({dataset.url})")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(
            "This notebook provides a starting point for exploring this BIDS dataset."
        )

        return new_markdown_cell("\n".join(lines))

    def _install_cell(self) -> nbformat.NotebookNode:
        """Create installation cell."""
        code = '''# Install required packages (run once)
# !pip install pybids mne nibabel pandas matplotlib numpy'''
        return new_code_cell(code)

    def _imports_cell(self) -> nbformat.NotebookNode:
        """Create imports cell."""
        code = '''import warnings
warnings.filterwarnings('ignore')

from bids import BIDSLayout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# For neuroimaging data (optional)
# import nibabel as nib
# import mne'''
        return new_code_cell(code)

    def _load_bids_cell(self, dataset: DatasetRecord) -> nbformat.NotebookNode:
        """Create BIDS loading cell."""
        code = f'''# Load BIDS dataset
bids_root = "path/to/bids/dataset"  # TODO: Update with actual path

# Initialize BIDS layout
layout = BIDSLayout(bids_root, validate=False)

print(f"BIDS Root: {{layout.root}}")
print(f"Dataset: {dataset.title}")'''
        return new_code_cell(code)

    def _description_cell(self) -> nbformat.NotebookNode:
        """Create dataset description cell."""
        code = '''# Dataset description
print("=" * 60)
print("DATASET DESCRIPTION")
print("=" * 60)

desc_file = Path(layout.root) / "dataset_description.json"
if desc_file.exists():
    with open(desc_file) as f:
        desc = json.load(f)
    for key, value in desc.items():
        print(f"{key}: {value}")
else:
    print("No dataset_description.json found")'''
        return new_code_cell(code)

    def _participants_cell(self) -> nbformat.NotebookNode:
        """Create participants inspection cell."""
        code = '''# Participants
print("=" * 60)
print("PARTICIPANTS")
print("=" * 60)

subjects = layout.get_subjects()
print(f"Number of subjects: {len(subjects)}")
print(f"Subject IDs: {subjects[:10]}{'...' if len(subjects) > 10 else ''}")

# Load participants.tsv if available
participants_file = Path(layout.root) / "participants.tsv"
if participants_file.exists():
    participants_df = pd.read_csv(participants_file, sep='\\t')
    print(f"\\nParticipants table columns: {list(participants_df.columns)}")
    display(participants_df.head())'''
        return new_code_cell(code)

    def _sessions_cell(self) -> nbformat.NotebookNode:
        """Create sessions inspection cell."""
        code = '''# Sessions
print("=" * 60)
print("SESSIONS")
print("=" * 60)

sessions = layout.get_sessions()
if sessions:
    print(f"Number of sessions: {len(sessions)}")
    print(f"Session IDs: {sessions}")
else:
    print("No session structure (single session per subject)")'''
        return new_code_cell(code)

    def _data_files_cell(self) -> nbformat.NotebookNode:
        """Create data files inspection cell."""
        code = '''# Data files by type
print("=" * 60)
print("DATA FILES")
print("=" * 60)

# Get data types
datatypes = layout.get_datatypes()
print(f"Data types: {datatypes}")

# Count files by suffix
all_files = layout.get()
suffix_counts = {}
for f in all_files:
    suffix = f.entities.get('suffix', 'unknown')
    suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1

print(f"\\nFiles by suffix:")
for suffix, count in sorted(suffix_counts.items(), key=lambda x: -x[1]):
    print(f"  {suffix}: {count}")

# Example: get specific file type
# bold_files = layout.get(suffix='bold', extension='.nii.gz')
# print(f"\\nBOLD files: {len(bold_files)}")'''
        return new_code_cell(code)

    def _events_cell(self) -> nbformat.NotebookNode:
        """Create events inspection cell."""
        code = '''# Events files
print("=" * 60)
print("EVENTS")
print("=" * 60)

events_files = layout.get(suffix='events', extension='.tsv')
print(f"Number of events files: {len(events_files)}")

if events_files:
    # Load first events file as example
    example_events = pd.read_csv(events_files[0].path, sep='\\t')
    print(f"\\nExample events file: {events_files[0].filename}")
    print(f"Columns: {list(example_events.columns)}")

    if 'trial_type' in example_events.columns:
        print(f"\\nTrial types: {example_events['trial_type'].unique()}")

    print(f"\\nFirst few events:")
    display(example_events.head(10))'''
        return new_code_cell(code)

    def _plot_cell(self) -> nbformat.NotebookNode:
        """Create basic plotting cell."""
        code = '''# Basic visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Files per subject
ax1 = axes[0]
subjects = layout.get_subjects()
files_per_subject = [len(layout.get(subject=s)) for s in subjects[:20]]
ax1.bar(range(len(files_per_subject)), files_per_subject)
ax1.set_xlabel('Subject Index')
ax1.set_ylabel('Number of Files')
ax1.set_title('Files per Subject (first 20)')

# Plot 2: Event timeline (if events exist)
ax2 = axes[1]
events_files = layout.get(suffix='events', extension='.tsv')
if events_files:
    events_df = pd.read_csv(events_files[0].path, sep='\\t')
    if 'onset' in events_df.columns:
        ax2.eventplot([events_df['onset'].values], lineoffsets=0, linelengths=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_title(f'Event Onsets: {events_files[0].filename}')
    else:
        ax2.text(0.5, 0.5, 'No onset column', ha='center', va='center')
else:
    ax2.text(0.5, 0.5, 'No events files', ha='center', va='center')
    ax2.set_title('Events')

plt.tight_layout()
plt.show()'''
        return new_code_cell(code)

    def _todo_cells(
        self, card: Optional[DatasetCard]
    ) -> list[nbformat.NotebookNode]:
        """Create TODO cells for advanced analysis."""
        cells = []

        cells.append(
            new_markdown_cell(
                "---\n\n## TODO: Advanced Analysis\n\n"
                "The cells below provide starting points for common analyses."
            )
        )

        # Load neuroimaging data
        cells.append(
            new_code_cell(
                '''# TODO: Load and inspect neuroimaging data

# For MRI/fMRI data:
# import nibabel as nib
# bold_files = layout.get(suffix='bold', extension='.nii.gz')
# if bold_files:
#     img = nib.load(bold_files[0].path)
#     print(f"Shape: {img.shape}")
#     print(f"Affine:\\n{img.affine}")
#     data = img.get_fdata()

# For EEG/MEG data:
# import mne
# eeg_files = layout.get(suffix='eeg', extension='.vhdr')  # or .edf, .set
# if eeg_files:
#     raw = mne.io.read_raw_brainvision(eeg_files[0].path)
#     print(raw.info)'''
            )
        )

        # Preprocessing pipeline
        cells.append(
            new_code_cell(
                '''# TODO: Set up preprocessing pipeline

# For fMRI:
# from nilearn import image, plotting
# from nilearn.glm.first_level import FirstLevelModel

# For EEG/MEG:
# raw.filter(l_freq=0.1, h_freq=40)
# raw.set_eeg_reference('average')
# events = mne.find_events(raw)
# epochs = mne.Epochs(raw, events)'''
            )
        )

        return cells
