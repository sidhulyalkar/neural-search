"""NWB starter notebook generation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from neural_search.schemas import NotebookGenerationResponse


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def generate_nwb_starter_notebook(
    dataset: Any, asset: Any, output_path: str | Path
) -> NotebookGenerationResponse:
    """Generate a valid .ipynb starter notebook for an NWB asset."""

    dataset_id = _get_value(dataset, "id", _get_value(dataset, "source_id", "DEMO"))
    asset_id = _get_value(asset, "id", _get_value(asset, "path", "DEMO"))
    title = _get_value(dataset, "title", "Neural Search Dataset")
    asset_path = _get_value(asset, "path", "path/to/file.nwb")
    warnings: list[str] = []
    if not str(asset_path).endswith(".nwb"):
        warnings.append("Asset path does not end in .nwb; notebook assumes NWB-compatible input.")

    notebook = new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "neural_search": {"dataset_id": str(dataset_id), "asset_id": str(asset_id)},
        },
        cells=[
            new_markdown_cell(f"# NWB Starter Notebook\n\nDataset: {title}\n\nAsset: `{asset_path}`"),
            new_code_cell(
                "from pathlib import Path\n"
                "import matplotlib.pyplot as plt\n"
                "import pandas as pd\n"
                "from pynwb import NWBHDF5IO\n"
            ),
            new_code_cell(f"nwb_path = Path({str(asset_path)!r})\nprint(nwb_path)"),
            new_markdown_cell("## Load NWB"),
            new_code_cell(
                "io = NWBHDF5IO(str(nwb_path), mode='r')\n"
                "nwbfile = io.read()\n"
                "nwbfile\n"
            ),
            new_markdown_cell("## Session Metadata"),
            new_code_cell(
                "session_metadata = {\n"
                "    'session_description': nwbfile.session_description,\n"
                "    'identifier': nwbfile.identifier,\n"
                "    'session_start_time': nwbfile.session_start_time,\n"
                "    'experimenter': nwbfile.experimenter,\n"
                "    'lab': nwbfile.lab,\n"
                "    'institution': nwbfile.institution,\n"
                "}\n"
                "session_metadata\n"
            ),
            new_markdown_cell("## Acquisition Objects"),
            new_code_cell(
                "acquisition = list(nwbfile.acquisition.keys())\n"
                "pd.DataFrame({'acquisition_object': acquisition})\n"
            ),
            new_markdown_cell("## Processing Modules"),
            new_code_cell(
                "processing_modules = list(nwbfile.processing.keys())\n"
                "pd.DataFrame({'processing_module': processing_modules})\n"
            ),
            new_markdown_cell("## Units Table"),
            new_code_cell(
                "if nwbfile.units is not None:\n"
                "    units_df = nwbfile.units.to_dataframe()\n"
                "    display(units_df.head())\n"
                "else:\n"
                "    print('No units table found.')\n"
            ),
            new_markdown_cell("## Trials Table"),
            new_code_cell(
                "if nwbfile.trials is not None:\n"
                "    trials_df = nwbfile.trials.to_dataframe()\n"
                "    display(trials_df.head())\n"
                "else:\n"
                "    print('No trials table found.')\n"
            ),
            new_markdown_cell("## Event Column Summary"),
            new_code_cell(
                "if nwbfile.trials is not None:\n"
                "    trials_df = nwbfile.trials.to_dataframe()\n"
                "    summary = pd.DataFrame({\n"
                "        'column': trials_df.columns,\n"
                "        'dtype': [str(dtype) for dtype in trials_df.dtypes],\n"
                "        'missing': [int(trials_df[col].isna().sum()) for col in trials_df.columns],\n"
                "    })\n"
                "    display(summary)\n"
                "else:\n"
                "    print('No trial events available to summarize.')\n"
            ),
            new_markdown_cell("## Placeholder Plots"),
            new_code_cell(
                "if nwbfile.trials is not None:\n"
                "    trials_df = nwbfile.trials.to_dataframe()\n"
                "    if {'start_time', 'stop_time'}.issubset(trials_df.columns):\n"
                "        duration = trials_df['stop_time'] - trials_df['start_time']\n"
                "        duration.plot(kind='hist', title='Trial duration')\n"
                "        plt.xlabel('seconds')\n"
                "        plt.show()\n"
                "else:\n"
                "    print('Add dataset-specific plots after inspection.')\n"
            ),
            new_code_cell("io.close()"),
        ],
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.validate(notebook)
    nbformat.write(notebook, output)
    return NotebookGenerationResponse(
        dataset_id=dataset_id,
        asset_id=asset_id,
        output_path=str(output),
        valid=True,
        warnings=warnings,
    )

