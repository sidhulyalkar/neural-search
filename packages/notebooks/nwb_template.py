"""NWB notebook template."""

from typing import Optional

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

import sys
sys.path.insert(0, str(__file__).rsplit("/packages", 1)[0] + "/packages")

from shared.schemas import DatasetRecord, DatasetCard


class NWBNotebookTemplate:
    """Generate NWB starter notebooks."""

    def generate(
        self,
        dataset: DatasetRecord,
        card: Optional[DatasetCard] = None,
        nwb_path: Optional[str] = None,
    ) -> nbformat.NotebookNode:
        """Generate an NWB exploration notebook."""
        cells = []

        # Title cell
        cells.append(self._title_cell(dataset, card))

        # Installation cell
        cells.append(self._install_cell())

        # Imports cell
        cells.append(self._imports_cell())

        # Load NWB cell
        cells.append(self._load_nwb_cell(dataset, nwb_path))

        # Session metadata cell
        cells.append(self._session_metadata_cell())

        # Acquisition cell
        cells.append(self._acquisition_cell())

        # Processing modules cell
        cells.append(self._processing_cell())

        # Units cell (if electrophysiology)
        if self._has_ephys(dataset, card):
            cells.append(self._units_cell())

        # Trials cell
        cells.append(self._trials_cell())

        # Events cell
        cells.append(self._events_cell())

        # Basic plot cell
        cells.append(self._plot_cell())

        # TODO cells for advanced analysis
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
        ]
        if dataset.doi:
            lines.append(f"**DOI:** {dataset.doi}")
        if dataset.url:
            lines.append(f"**URL:** [{dataset.url}]({dataset.url})")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(
            "This notebook provides a starting point for exploring this NWB dataset."
        )

        return new_markdown_cell("\n".join(lines))

    def _install_cell(self) -> nbformat.NotebookNode:
        """Create installation cell."""
        code = '''# Install required packages (run once)
# !pip install pynwb h5py pandas matplotlib numpy dandi'''
        return new_code_cell(code)

    def _imports_cell(self) -> nbformat.NotebookNode:
        """Create imports cell."""
        code = '''import warnings
warnings.filterwarnings('ignore')

from pynwb import NWBHDF5IO
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# For remote streaming (optional)
# import remfile
# import fsspec'''
        return new_code_cell(code)

    def _load_nwb_cell(
        self, dataset: DatasetRecord, nwb_path: Optional[str]
    ) -> nbformat.NotebookNode:
        """Create NWB loading cell."""
        if nwb_path:
            path_str = f'"{nwb_path}"'
        else:
            path_str = '"path/to/your/file.nwb"  # TODO: Update with actual path'

        code = f'''# Load NWB file
nwb_path = {path_str}

# For local files:
io = NWBHDF5IO(nwb_path, mode='r')
nwb = io.read()

# For remote streaming from DANDI (uncomment if needed):
# from dandi.dandiapi import DandiAPIClient
# client = DandiAPIClient()
# dandiset = client.get_dandiset("{dataset.source_id}")
# assets = list(dandiset.get_assets())
# # Get first NWB file
# asset = next(a for a in assets if a.path.endswith('.nwb'))
# url = asset.get_content_url(follow_redirects=1, strip_query=True)
# # Stream with fsspec
# fs = fsspec.filesystem("http")
# f = fs.open(url, "rb")
# file = h5py.File(f)
# io = NWBHDF5IO(file=file, load_namespaces=True)
# nwb = io.read()

print(f"Loaded: {{nwb.identifier}}")'''
        return new_code_cell(code)

    def _session_metadata_cell(self) -> nbformat.NotebookNode:
        """Create session metadata inspection cell."""
        code = '''# Session metadata
print("=" * 60)
print("SESSION METADATA")
print("=" * 60)
print(f"Session ID: {nwb.identifier}")
print(f"Session Description: {nwb.session_description}")
print(f"Session Start: {nwb.session_start_time}")
print(f"Experimenter: {nwb.experimenter}")
print(f"Lab: {nwb.lab}")
print(f"Institution: {nwb.institution}")

# Subject info
if nwb.subject:
    print("\\n" + "-" * 40)
    print("SUBJECT")
    print("-" * 40)
    print(f"Subject ID: {nwb.subject.subject_id}")
    print(f"Species: {nwb.subject.species}")
    print(f"Age: {nwb.subject.age}")
    print(f"Sex: {nwb.subject.sex}")
    print(f"Description: {nwb.subject.description}")'''
        return new_code_cell(code)

    def _acquisition_cell(self) -> nbformat.NotebookNode:
        """Create acquisition inspection cell."""
        code = '''# Inspect acquisition data
print("=" * 60)
print("ACQUISITION")
print("=" * 60)

for name, data in nwb.acquisition.items():
    print(f"\\n{name}:")
    print(f"  Type: {type(data).__name__}")
    if hasattr(data, 'data'):
        if hasattr(data.data, 'shape'):
            print(f"  Shape: {data.data.shape}")
    if hasattr(data, 'description'):
        print(f"  Description: {data.description[:100]}...")
    if hasattr(data, 'rate'):
        print(f"  Sampling Rate: {data.rate} Hz")'''
        return new_code_cell(code)

    def _processing_cell(self) -> nbformat.NotebookNode:
        """Create processing modules inspection cell."""
        code = '''# Inspect processing modules
print("=" * 60)
print("PROCESSING MODULES")
print("=" * 60)

for module_name, module in nwb.processing.items():
    print(f"\\n{module_name}:")
    print(f"  Description: {module.description}")
    print("  Data interfaces:")
    for interface_name, interface in module.data_interfaces.items():
        print(f"    - {interface_name}: {type(interface).__name__}")'''
        return new_code_cell(code)

    def _units_cell(self) -> nbformat.NotebookNode:
        """Create units inspection cell."""
        code = '''# Inspect units (spike sorted data)
print("=" * 60)
print("UNITS (Spike Sorted Data)")
print("=" * 60)

if nwb.units is not None:
    units_df = nwb.units.to_dataframe()
    print(f"Number of units: {len(units_df)}")
    print(f"\\nColumns: {list(units_df.columns)}")
    print(f"\\nFirst few units:")
    display(units_df.head())
else:
    print("No units table found in this NWB file.")'''
        return new_code_cell(code)

    def _trials_cell(self) -> nbformat.NotebookNode:
        """Create trials inspection cell."""
        code = '''# Inspect trials table
print("=" * 60)
print("TRIALS")
print("=" * 60)

if nwb.trials is not None:
    trials_df = nwb.trials.to_dataframe()
    print(f"Number of trials: {len(trials_df)}")
    print(f"\\nColumns: {list(trials_df.columns)}")
    print(f"\\nTrial duration stats:")
    if 'start_time' in trials_df.columns and 'stop_time' in trials_df.columns:
        durations = trials_df['stop_time'] - trials_df['start_time']
        print(f"  Mean: {durations.mean():.3f} s")
        print(f"  Std: {durations.std():.3f} s")
        print(f"  Min: {durations.min():.3f} s")
        print(f"  Max: {durations.max():.3f} s")
    print(f"\\nFirst few trials:")
    display(trials_df.head())
else:
    print("No trials table found in this NWB file.")'''
        return new_code_cell(code)

    def _events_cell(self) -> nbformat.NotebookNode:
        """Create events summary cell."""
        code = '''# Summarize event columns in trials
print("=" * 60)
print("EVENT COLUMNS")
print("=" * 60)

if nwb.trials is not None:
    trials_df = nwb.trials.to_dataframe()

    # Find time-related columns (potential events)
    time_cols = [c for c in trials_df.columns if 'time' in c.lower()]
    event_cols = [c for c in trials_df.columns if any(
        x in c.lower() for x in ['onset', 'offset', 'start', 'stop', 'cue', 'stim']
    )]

    print("Time-related columns:")
    for col in time_cols:
        print(f"  - {col}")

    print("\\nPotential event columns:")
    for col in set(event_cols) - set(time_cols):
        print(f"  - {col}")
        if trials_df[col].dtype in ['object', 'category']:
            print(f"    Values: {trials_df[col].unique()[:5]}")
else:
    print("No trials table available.")'''
        return new_code_cell(code)

    def _plot_cell(self) -> nbformat.NotebookNode:
        """Create basic plotting cell."""
        code = '''# Basic trial/event visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

if nwb.trials is not None:
    trials_df = nwb.trials.to_dataframe()

    # Plot 1: Trial durations
    ax1 = axes[0]
    if 'start_time' in trials_df.columns and 'stop_time' in trials_df.columns:
        durations = trials_df['stop_time'] - trials_df['start_time']
        ax1.hist(durations, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Trial Duration (s)')
        ax1.set_ylabel('Count')
        ax1.set_title('Distribution of Trial Durations')
    else:
        ax1.text(0.5, 0.5, 'No duration data', ha='center', va='center')
        ax1.set_title('Trial Durations')

    # Plot 2: Trial timeline
    ax2 = axes[1]
    if 'start_time' in trials_df.columns:
        ax2.eventplot([trials_df['start_time'].values], lineoffsets=0, linelengths=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Trials')
        ax2.set_title('Trial Start Times')
        ax2.set_xlim(0, trials_df['start_time'].max())
    else:
        ax2.text(0.5, 0.5, 'No timing data', ha='center', va='center')
        ax2.set_title('Trial Timeline')

else:
    axes[0].text(0.5, 0.5, 'No trials data', ha='center', va='center')
    axes[1].text(0.5, 0.5, 'No trials data', ha='center', va='center')

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

        # Event-aligned activity
        cells.append(
            new_code_cell(
                '''# TODO: Event-aligned activity analysis
# This requires spike times and event times

def align_spikes_to_events(spike_times, event_times, window=(-0.5, 1.0)):
    """Align spike times to event times within a window."""
    aligned = []
    for event_time in event_times:
        mask = (spike_times >= event_time + window[0]) & \\
               (spike_times <= event_time + window[1])
        aligned.append(spike_times[mask] - event_time)
    return aligned

# Example usage:
# if nwb.units is not None and nwb.trials is not None:
#     spike_times = nwb.units['spike_times'][0]
#     event_times = nwb.trials['start_time'][:]
#     aligned = align_spikes_to_events(spike_times, event_times)'''
            )
        )

        # Decoding placeholder
        cells.append(
            new_code_cell(
                '''# TODO: Basic decoding analysis
# Decode trial type or choice from neural activity

# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LogisticRegression

# Example structure:
# X = neural_features  # (n_trials, n_features)
# y = trial_labels     # (n_trials,)
# clf = LogisticRegression()
# scores = cross_val_score(clf, X, y, cv=5)
# print(f"Decoding accuracy: {scores.mean():.2f} +/- {scores.std():.2f}")'''
            )
        )

        # Cleanup
        cells.append(
            new_code_cell(
                '''# Clean up
io.close()
print("NWB file closed.")'''
            )
        )

        return cells

    def _has_ephys(
        self, dataset: DatasetRecord, card: Optional[DatasetCard]
    ) -> bool:
        """Check if dataset likely has electrophysiology data."""
        ephys_terms = [
            "ephys",
            "electrophysiology",
            "neuropixels",
            "tetrode",
            "spike",
            "unit",
        ]
        modalities = dataset.modalities or []
        if card and card.modalities:
            modalities = card.modalities

        for mod in modalities:
            if any(term in mod.lower() for term in ephys_terms):
                return True

        desc = (dataset.description or "").lower()
        return any(term in desc for term in ephys_terms)
