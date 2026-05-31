"""NWB starter notebook generation with comprehensive data inspection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from neural_search.schemas import DatasetCardRead, NotebookGenerationResponse


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _format_list(items: list[str], max_items: int = 5) -> str:
    """Format a list for display in markdown."""
    if not items:
        return "None"
    if len(items) <= max_items:
        return ", ".join(items)
    return ", ".join(items[:max_items]) + f" (+{len(items) - max_items} more)"


def _build_dataset_card_section(card: DatasetCardRead | Mapping[str, Any] | None) -> list[str]:
    """Build markdown lines for dataset card metadata."""
    if card is None:
        return ["No dataset card available."]

    if isinstance(card, Mapping):
        summary = card.get("summary", "No summary")
        readiness = card.get("analysis_readiness", {})
        score = readiness.get("score", "N/A") if isinstance(readiness, dict) else getattr(readiness, "score", "N/A")
        strengths = readiness.get("strengths", []) if isinstance(readiness, dict) else getattr(readiness, "strengths", [])
        limitations = readiness.get("limitations", []) if isinstance(readiness, dict) else getattr(readiness, "limitations", [])
        suggested = card.get("suggested_analyses", [])
    else:
        summary = card.summary
        score = card.analysis_readiness.score
        strengths = card.analysis_readiness.strengths
        limitations = card.analysis_readiness.limitations
        suggested = card.suggested_analyses

    lines = [
        f"**Summary:** {summary}",
        "",
        f"**Analysis Readiness Score:** {score}/100",
        "",
    ]

    if strengths:
        lines.append("**Strengths:**")
        for s in strengths[:3]:
            lines.append(f"- {s}")
        lines.append("")

    if limitations:
        lines.append("**Limitations:**")
        for lim in limitations[:3]:
            lines.append(f"- {lim}")
        lines.append("")

    if suggested:
        lines.append("**Suggested Analyses:**")
        for sug in suggested[:5]:
            lines.append(f"- {sug}")

    return lines


def _build_recipe_cells(recipes: Sequence[Mapping[str, Any]] | None) -> list[Any]:
    """Build notebook cells for selected analysis recipes."""

    cells: list[Any] = []
    for recipe in recipes or []:
        recipe_id = str(recipe.get("id", "recipe"))
        title = str(recipe.get("title", recipe_id))
        summary = str(recipe.get("summary", ""))
        analyses = recipe.get("analyses", [])
        required_fields = recipe.get("required_fields", [])
        cells.append(
            new_markdown_cell(
                "## Analysis Recipe: "
                + title
                + "\n\n"
                + summary
                + "\n\n"
                + "**Recipe ID:** `"
                + recipe_id
                + "`\n\n"
                + "**Analyses:** "
                + _format_list([str(item) for item in analyses])
                + "\n\n"
                + "**Expected fields:** "
                + _format_list([str(item) for item in required_fields])
            )
        )
        for cell in recipe.get("cells", []):
            if not isinstance(cell, Mapping):
                continue
            body = str(cell.get("body", ""))
            if not body:
                continue
            cell_type = str(cell.get("type", "markdown"))
            cell_title = str(cell.get("title", "Recipe step"))
            if cell_type == "code":
                cells.append(new_markdown_cell(f"### {cell_title}"))
                cells.append(new_code_cell(body.rstrip() + "\n"))
            else:
                cells.append(new_markdown_cell(body.rstrip()))
    return cells


def _build_template_cells(template: Mapping[str, Any] | None) -> list[Any]:
    """Build notebook cells supplied by a selected template."""

    if not template:
        return []
    cells = [
        new_markdown_cell(
            "## Notebook Template\n\n"
            f"**Template ID:** `{template.get('id', 'unknown')}`\n\n"
            f"**Template:** {template.get('title', template.get('id', 'unknown'))}\n\n"
            f"{template.get('description', '')}"
        )
    ]
    for cell in template.get("cells", []) or []:
        if not isinstance(cell, Mapping):
            continue
        body = str(cell.get("body", "")).rstrip()
        if not body:
            continue
        title = str(cell.get("title", "Template step"))
        if cell.get("type") == "code":
            cells.append(new_markdown_cell(f"### {title}"))
            cells.append(new_code_cell(body + "\n"))
        else:
            cells.append(new_markdown_cell(body))
    return cells


def _build_template_warning_cells(warnings: Sequence[str]) -> list[Any]:
    if not warnings:
        return []
    return [
        new_markdown_cell(
            "## Template Availability Warnings\n\n"
            + "\n".join(f"- {warning}" for warning in warnings)
        )
    ]


def _inspection_summary_cell(dataset_id: Any, template_id: str | None) -> Any:
    return new_code_cell(
        "# Save an inspection summary JSON for this notebook run\n"
        "import json\n"
        "from pathlib import Path\n"
        "\n"
        "inspection_summary = {\n"
        f"    'dataset_id': {str(dataset_id)!r},\n"
        f"    'template_id': {str(template_id or 'generic_nwb_inspection')!r},\n"
        "    'n_acquisition_objects': len(nwbfile.acquisition) if 'nwbfile' in globals() else None,\n"
        "    'n_processing_modules': len(nwbfile.processing) if 'nwbfile' in globals() else None,\n"
        "    'n_trials': len(nwbfile.trials.to_dataframe()) if 'nwbfile' in globals() and nwbfile.trials is not None else None,\n"
        "    'n_units': len(nwbfile.units.to_dataframe()) if 'nwbfile' in globals() and nwbfile.units is not None else None,\n"
        "    'template_warnings': notebook_template_warnings if 'notebook_template_warnings' in globals() else [],\n"
        "}\n"
        "summary_path = Path(f'{inspection_summary[\"dataset_id\"]}_inspection_summary.json')\n"
        "summary_path.write_text(json.dumps(inspection_summary, indent=2, default=str))\n"
        "print(f'Inspection summary written to {summary_path}')\n"
        "inspection_summary\n"
    )


def generate_nwb_starter_notebook(
    dataset: Any,
    asset: Any,
    output_path: str | Path,
    card: DatasetCardRead | Mapping[str, Any] | None = None,
    recipes: Sequence[Mapping[str, Any]] | None = None,
    notebook_template: Mapping[str, Any] | None = None,
    template_warnings: Sequence[str] | None = None,
) -> NotebookGenerationResponse:
    """Generate a comprehensive NWB starter notebook for an asset.

    Args:
        dataset: Dataset record with metadata.
        asset: Asset record with path and type info.
        output_path: Where to write the .ipynb file.
        card: Optional dataset card for metadata display.

    Returns:
        NotebookGenerationResponse with validation status and warnings.
    """
    dataset_id = _get_value(dataset, "id", _get_value(dataset, "source_id", "UNKNOWN"))
    asset_id = _get_value(asset, "id", _get_value(asset, "path", "UNKNOWN"))
    title = _get_value(dataset, "title", "Neural Search Dataset")
    description = _get_value(dataset, "description", "")
    source_url = _get_value(dataset, "url", "")
    asset_path = _get_value(asset, "path", "path/to/file.nwb")
    source = _get_value(dataset, "source", "unknown")

    # Extract metadata lists
    species = _get_value(dataset, "species", [])
    modalities = _get_value(dataset, "modalities", [])
    tasks = _get_value(dataset, "tasks", [])
    behaviors = _get_value(dataset, "behaviors", [])
    brain_regions = _get_value(dataset, "brain_regions", [])

    warnings: list[str] = list(template_warnings or [])
    if not str(asset_path).endswith(".nwb"):
        warnings.append("Asset path does not end in .nwb; notebook assumes NWB-compatible input.")

    # Build header markdown
    header_lines = [
        f"# NWB Starter Notebook: {title}",
        "",
        f"**Dataset ID:** `{dataset_id}`  ",
        f"**Source:** {source}  ",
    ]
    if source_url:
        header_lines.append(f"**URL:** [{source_url}]({source_url})  ")
    header_lines.extend([
        f"**Asset:** `{asset_path}`  ",
        f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ])

    # Dataset metadata section
    if species or modalities or tasks:
        header_lines.append("## Dataset Metadata")
        header_lines.append("")
        if species:
            header_lines.append(f"- **Species:** {_format_list(species)}")
        if modalities:
            header_lines.append(f"- **Modalities:** {_format_list(modalities)}")
        if tasks:
            header_lines.append(f"- **Tasks:** {_format_list(tasks)}")
        if behaviors:
            header_lines.append(f"- **Behaviors:** {_format_list(behaviors)}")
        if brain_regions:
            header_lines.append(f"- **Brain Regions:** {_format_list(brain_regions)}")
        header_lines.append("")

    if description:
        header_lines.extend([
            "## Description",
            "",
            description[:500] + ("..." if len(description) > 500 else ""),
            "",
        ])

    # Dataset card section
    header_lines.extend([
        "## Dataset Card",
        "",
    ])
    header_lines.extend(_build_dataset_card_section(card))

    cells = [
        # Header
        new_markdown_cell("\n".join(header_lines)),

        # Setup
        new_markdown_cell("---\n## Setup"),
        new_code_cell(
            "from pathlib import Path\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            f"notebook_template_id = {str(_get_value(notebook_template or {}, 'id', 'generic_nwb_inspection'))!r}\n"
            f"notebook_template_warnings = {list(template_warnings or [])!r}\n"
            "\n"
            "import matplotlib.pyplot as plt\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "from pynwb import NWBHDF5IO\n"
            "\n"
            "pd.set_option('display.max_columns', 50)\n"
            "pd.set_option('display.max_rows', 20)\n"
            "%matplotlib inline\n"
        ),

        # NWB path
        new_markdown_cell("## NWB File Path"),
        new_code_cell(
            f"# TODO: Update this path to point to your local NWB file\n"
            f"nwb_path = Path({str(asset_path)!r})\n"
            f"\n"
            f"# Verify file exists\n"
            f"if not nwb_path.exists():\n"
            f"    print(f'WARNING: File not found: {{nwb_path}}')\n"
            f"    print('Please update nwb_path to point to your local copy.')\n"
            f"else:\n"
            f"    print(f'NWB file: {{nwb_path}}')\n"
            f"    print(f'Size: {{nwb_path.stat().st_size / 1e6:.1f}} MB')\n"
        ),

        # Load NWB
        new_markdown_cell("## Load NWB File"),
        new_code_cell(
            "io = NWBHDF5IO(str(nwb_path), mode='r')\n"
            "nwbfile = io.read()\n"
            "print(f'Loaded: {nwbfile.identifier}')\n"
            "nwbfile\n"
        ),

        # Session metadata
        new_markdown_cell("## Session Metadata"),
        new_code_cell(
            "session_metadata = {\n"
            "    'identifier': nwbfile.identifier,\n"
            "    'session_description': nwbfile.session_description,\n"
            "    'session_start_time': str(nwbfile.session_start_time),\n"
            "    'timestamps_reference_time': str(nwbfile.timestamps_reference_time) if nwbfile.timestamps_reference_time else None,\n"
            "    'experimenter': nwbfile.experimenter,\n"
            "    'lab': nwbfile.lab,\n"
            "    'institution': nwbfile.institution,\n"
            "    'experiment_description': nwbfile.experiment_description,\n"
            "    'keywords': list(nwbfile.keywords) if nwbfile.keywords else None,\n"
            "}\n"
            "\n"
            "for key, value in session_metadata.items():\n"
            "    if value:\n"
            "        print(f'{key}: {value}')\n"
        ),

        # Subject metadata
        new_markdown_cell("## Subject Metadata"),
        new_code_cell(
            "if nwbfile.subject is not None:\n"
            "    subject = nwbfile.subject\n"
            "    subject_metadata = {\n"
            "        'subject_id': subject.subject_id,\n"
            "        'species': subject.species,\n"
            "        'sex': subject.sex,\n"
            "        'age': subject.age,\n"
            "        'date_of_birth': str(subject.date_of_birth) if subject.date_of_birth else None,\n"
            "        'strain': subject.strain,\n"
            "        'genotype': subject.genotype,\n"
            "        'description': subject.description,\n"
            "    }\n"
            "    for key, value in subject_metadata.items():\n"
            "        if value:\n"
            "            print(f'{key}: {value}')\n"
            "else:\n"
            "    print('No subject metadata found.')\n"
        ),

        # Acquisition objects
        new_markdown_cell("## Acquisition Objects"),
        new_code_cell(
            "acquisition_keys = list(nwbfile.acquisition.keys())\n"
            "print(f'Found {len(acquisition_keys)} acquisition objects:')\n"
            "\n"
            "if acquisition_keys:\n"
            "    acq_info = []\n"
            "    for key in acquisition_keys:\n"
            "        obj = nwbfile.acquisition[key]\n"
            "        acq_info.append({\n"
            "            'name': key,\n"
            "            'type': type(obj).__name__,\n"
            "            'description': getattr(obj, 'description', '')[:50] if hasattr(obj, 'description') else '',\n"
            "        })\n"
            "    display(pd.DataFrame(acq_info))\n"
            "else:\n"
            "    print('No acquisition objects found.')\n"
        ),

        # Processing modules
        new_markdown_cell("## Processing Modules"),
        new_code_cell(
            "processing_keys = list(nwbfile.processing.keys())\n"
            "print(f'Found {len(processing_keys)} processing modules:')\n"
            "\n"
            "if processing_keys:\n"
            "    proc_info = []\n"
            "    for key in processing_keys:\n"
            "        module = nwbfile.processing[key]\n"
            "        data_interfaces = list(module.data_interfaces.keys())\n"
            "        proc_info.append({\n"
            "            'module': key,\n"
            "            'description': module.description[:50] if module.description else '',\n"
            "            'data_interfaces': ', '.join(data_interfaces[:3]) + ('...' if len(data_interfaces) > 3 else ''),\n"
            "            'n_interfaces': len(data_interfaces),\n"
            "        })\n"
            "    display(pd.DataFrame(proc_info))\n"
            "else:\n"
            "    print('No processing modules found.')\n"
        ),

        # Units table
        new_markdown_cell("## Units Table (Spike Data)"),
        new_code_cell(
            "if nwbfile.units is not None:\n"
            "    units_df = nwbfile.units.to_dataframe()\n"
            "    print(f'Found {len(units_df)} units with columns: {list(units_df.columns)}')\n"
            "    display(units_df.head(10))\n"
            "else:\n"
            "    print('No units table found.')\n"
        ),

        # Trials table
        new_markdown_cell("## Trials Table"),
        new_code_cell(
            "if nwbfile.trials is not None:\n"
            "    trials_df = nwbfile.trials.to_dataframe()\n"
            "    print(f'Found {len(trials_df)} trials with columns: {list(trials_df.columns)}')\n"
            "    display(trials_df.head(10))\n"
            "else:\n"
            "    print('No trials table found.')\n"
        ),

        # Intervals
        new_markdown_cell("## Intervals (Other Time Tables)"),
        new_code_cell(
            "interval_keys = list(nwbfile.intervals.keys()) if nwbfile.intervals else []\n"
            "# Also check for trials in intervals if not in main trials\n"
            "if 'trials' in interval_keys:\n"
            "    interval_keys.remove('trials')  # Already shown above\n"
            "\n"
            "print(f'Found {len(interval_keys)} interval tables:')\n"
            "\n"
            "if interval_keys:\n"
            "    for key in interval_keys:\n"
            "        interval_table = nwbfile.intervals[key]\n"
            "        interval_df = interval_table.to_dataframe()\n"
            "        print(f'\\n--- {key} ({len(interval_df)} rows) ---')\n"
            "        print(f'Columns: {list(interval_df.columns)}')\n"
            "        display(interval_df.head(5))\n"
            "else:\n"
            "    print('No additional interval tables found.')\n"
        ),

        # TimeSeries summary helper
        new_markdown_cell("## TimeSeries Summary Helper"),
        new_code_cell(
            "def summarize_timeseries(ts):\n"
            "    \"\"\"Summarize a TimeSeries object.\"\"\"\n"
            "    info = {\n"
            "        'name': ts.name,\n"
            "        'type': type(ts).__name__,\n"
            "        'description': ts.description[:60] if ts.description else '',\n"
            "        'unit': ts.unit if hasattr(ts, 'unit') else None,\n"
            "        'rate': ts.rate if hasattr(ts, 'rate') and ts.rate else None,\n"
            "    }\n"
            "    if hasattr(ts, 'data') and ts.data is not None:\n"
            "        data = ts.data\n"
            "        info['shape'] = data.shape if hasattr(data, 'shape') else len(data)\n"
            "        info['dtype'] = str(data.dtype) if hasattr(data, 'dtype') else type(data).__name__\n"
            "    return info\n"
            "\n"
            "\n"
            "def list_all_timeseries(nwbfile):\n"
            "    \"\"\"List all TimeSeries in the file.\"\"\"\n"
            "    all_ts = []\n"
            "    \n"
            "    # Acquisition\n"
            "    for name, obj in nwbfile.acquisition.items():\n"
            "        info = summarize_timeseries(obj)\n"
            "        info['location'] = 'acquisition'\n"
            "        all_ts.append(info)\n"
            "    \n"
            "    # Processing modules\n"
            "    for mod_name, module in nwbfile.processing.items():\n"
            "        for di_name, di in module.data_interfaces.items():\n"
            "            if hasattr(di, 'data'):\n"
            "                info = summarize_timeseries(di)\n"
            "                info['location'] = f'processing/{mod_name}'\n"
            "                all_ts.append(info)\n"
            "    \n"
            "    return pd.DataFrame(all_ts)\n"
            "\n"
            "\n"
            "ts_summary = list_all_timeseries(nwbfile)\n"
            "print(f'Found {len(ts_summary)} TimeSeries objects:')\n"
            "display(ts_summary)\n"
        ),

        # Trials column summary
        new_markdown_cell("## Trial Column Analysis"),
        new_code_cell(
            "if nwbfile.trials is not None:\n"
            "    trials_df = nwbfile.trials.to_dataframe()\n"
            "    \n"
            "    # Column summary\n"
            "    col_summary = []\n"
            "    for col in trials_df.columns:\n"
            "        col_data = trials_df[col]\n"
            "        col_summary.append({\n"
            "            'column': col,\n"
            "            'dtype': str(col_data.dtype),\n"
            "            'missing': int(col_data.isna().sum()),\n"
            "            'unique': col_data.nunique() if col_data.dtype != 'object' else min(col_data.nunique(), 20),\n"
            "            'sample': str(col_data.dropna().iloc[0])[:30] if len(col_data.dropna()) > 0 else 'N/A',\n"
            "        })\n"
            "    \n"
            "    print('Trial columns summary:')\n"
            "    display(pd.DataFrame(col_summary))\n"
            "else:\n"
            "    print('No trials table available for analysis.')\n"
        ),

        # Event histogram
        new_markdown_cell("## Basic Event Histogram"),
        new_code_cell(
            "if nwbfile.trials is not None:\n"
            "    trials_df = nwbfile.trials.to_dataframe()\n"
            "    \n"
            "    fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "    \n"
            "    # Trial duration histogram\n"
            "    if {'start_time', 'stop_time'}.issubset(trials_df.columns):\n"
            "        duration = trials_df['stop_time'] - trials_df['start_time']\n"
            "        axes[0].hist(duration, bins=30, edgecolor='black', alpha=0.7)\n"
            "        axes[0].set_xlabel('Duration (s)')\n"
            "        axes[0].set_ylabel('Count')\n"
            "        axes[0].set_title('Trial Duration Distribution')\n"
            "        axes[0].axvline(duration.median(), color='red', linestyle='--', label=f'Median: {duration.median():.2f}s')\n"
            "        axes[0].legend()\n"
            "    else:\n"
            "        axes[0].text(0.5, 0.5, 'No start_time/stop_time columns', ha='center', va='center')\n"
            "        axes[0].set_title('Trial Duration')\n"
            "    \n"
            "    # Trial start times\n"
            "    if 'start_time' in trials_df.columns:\n"
            "        axes[1].hist(trials_df['start_time'], bins=30, edgecolor='black', alpha=0.7)\n"
            "        axes[1].set_xlabel('Time (s)')\n"
            "        axes[1].set_ylabel('Count')\n"
            "        axes[1].set_title('Trial Start Times')\n"
            "    else:\n"
            "        axes[1].text(0.5, 0.5, 'No start_time column', ha='center', va='center')\n"
            "        axes[1].set_title('Trial Timing')\n"
            "    \n"
            "    plt.tight_layout()\n"
            "    plt.show()\n"
            "else:\n"
            "    print('No trials table found for histogram.')\n"
        ),
    ]

    template_cells = _build_template_cells(notebook_template)
    if template_cells:
        cells.extend([
            new_markdown_cell("---\n## Selected Notebook Template"),
            *_build_template_warning_cells(template_warnings or []),
            *template_cells,
        ])

    recipe_cells = _build_recipe_cells(recipes)
    if recipe_cells:
        cells.extend([
            new_markdown_cell("---\n## Selected Analysis Recipes"),
            *recipe_cells,
        ])

    cells.extend([

        # TODO section
        new_markdown_cell(
            "---\n"
            "## TODO: Analysis Roadmap\n"
            "\n"
            "Below are suggested next steps for analysis. Uncomment and customize as needed.\n"
        ),
        new_code_cell(
            "# =============================================================================\n"
            "# TODO 1: Event Alignment\n"
            "# =============================================================================\n"
            "# Align neural data to behavioral events (e.g., stimulus onset, reward, choice)\n"
            "#\n"
            "# def align_to_event(spike_times, event_times, window=(-0.5, 1.0)):\n"
            "#     \"\"\"Align spike times to event times within a window.\"\"\"\n"
            "#     aligned = []\n"
            "#     for event in event_times:\n"
            "#         mask = (spike_times >= event + window[0]) & (spike_times <= event + window[1])\n"
            "#         aligned.append(spike_times[mask] - event)\n"
            "#     return aligned\n"
            "#\n"
            "# Example:\n"
            "# event_times = trials_df['start_time'].values\n"
            "# aligned_spikes = align_to_event(unit_spike_times, event_times)\n"
        ),
        new_code_cell(
            "# =============================================================================\n"
            "# TODO 2: Neural Decoding\n"
            "# =============================================================================\n"
            "# Decode behavioral variables from neural activity\n"
            "#\n"
            "# from sklearn.model_selection import cross_val_score\n"
            "# from sklearn.linear_model import LogisticRegression\n"
            "#\n"
            "# def decode_choice(neural_features, choice_labels):\n"
            "#     \"\"\"Decode choice from neural features using logistic regression.\"\"\"\n"
            "#     clf = LogisticRegression(max_iter=1000)\n"
            "#     scores = cross_val_score(clf, neural_features, choice_labels, cv=5)\n"
            "#     return scores.mean(), scores.std()\n"
            "#\n"
            "# Example:\n"
            "# accuracy, std = decode_choice(firing_rates, trials_df['choice'].values)\n"
            "# print(f'Decoding accuracy: {accuracy:.2f} +/- {std:.2f}')\n"
        ),
        new_code_cell(
            "# =============================================================================\n"
            "# TODO 3: Behavioral Analysis\n"
            "# =============================================================================\n"
            "# Analyze behavioral patterns and performance\n"
            "#\n"
            "# def compute_psychometric(trials_df, stimulus_col, choice_col):\n"
            "#     \"\"\"Compute psychometric curve from trial data.\"\"\"\n"
            "#     grouped = trials_df.groupby(stimulus_col)[choice_col].mean()\n"
            "#     return grouped\n"
            "#\n"
            "# def reaction_time_analysis(trials_df):\n"
            "#     \"\"\"Analyze reaction times by trial type.\"\"\"\n"
            "#     if 'reaction_time' in trials_df.columns:\n"
            "#         return trials_df.groupby('trial_type')['reaction_time'].describe()\n"
            "#\n"
            "# Example:\n"
            "# rt_stats = reaction_time_analysis(trials_df)\n"
        ),
        new_code_cell(
            "# =============================================================================\n"
            "# TODO 4: QC Report Export\n"
            "# =============================================================================\n"
            "# Generate a quality control report for the dataset\n"
            "#\n"
            "# def generate_qc_report(nwbfile, output_path):\n"
            "#     \"\"\"Generate QC report for NWB file.\"\"\"\n"
            "#     report = {\n"
            "#         'identifier': nwbfile.identifier,\n"
            "#         'n_trials': len(nwbfile.trials.to_dataframe()) if nwbfile.trials else 0,\n"
            "#         'n_units': len(nwbfile.units.to_dataframe()) if nwbfile.units else 0,\n"
            "#         'acquisition_objects': list(nwbfile.acquisition.keys()),\n"
            "#         'processing_modules': list(nwbfile.processing.keys()),\n"
            "#         'missing_metadata': [],\n"
            "#     }\n"
            "#     \n"
            "#     # Check for missing metadata\n"
            "#     if not nwbfile.subject:\n"
            "#         report['missing_metadata'].append('subject')\n"
            "#     if not nwbfile.experimenter:\n"
            "#         report['missing_metadata'].append('experimenter')\n"
            "#     \n"
            "#     # Save report\n"
            "#     import json\n"
            "#     with open(output_path, 'w') as f:\n"
            "#         json.dump(report, f, indent=2, default=str)\n"
            "#     return report\n"
            "#\n"
            "# qc_report = generate_qc_report(nwbfile, 'qc_report.json')\n"
        ),
        _inspection_summary_cell(dataset_id, _get_value(notebook_template or {}, "id", None)),

        # Cleanup
        new_markdown_cell("---\n## Cleanup"),
        new_code_cell(
            "# Close the NWB file when done\n"
            "io.close()\n"
            "print('NWB file closed.')\n"
        ),
    ])

    # Build notebook
    notebook = new_notebook(
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
                "pygments_lexer": "ipython3",
            },
            "neural_search": {
                "dataset_id": str(dataset_id),
                "asset_id": str(asset_id),
                "source": source,
                "template_id": _get_value(notebook_template or {}, "id", "generic_nwb_inspection"),
                "template_title": _get_value(notebook_template or {}, "title", "Generic NWB inspection"),
                "template_warnings": list(template_warnings or []),
                "recipe_ids": [str(recipe.get("id", "")) for recipe in recipes or []],
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        },
        cells=cells,
    )

    # Write notebook
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
