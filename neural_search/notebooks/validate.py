"""Notebook validation CLI.

Validates generated Jupyter notebooks for structure, syntax, and Neural Search metadata.

Usage:
    python -m neural_search.notebooks.validate <notebook_path>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import nbformat
from nbformat import ValidationError as NBValidationError


def validate_notebook_structure(notebook_path: str | Path) -> dict[str, Any]:
    """Validate a Jupyter notebook's structure and content.

    Args:
        notebook_path: Path to the .ipynb file.

    Returns:
        Dict with validation results including:
        - valid: bool
        - errors: list of error messages
        - warnings: list of warning messages
        - metadata: extracted neural_search metadata
        - cell_count: number of cells
        - code_cells: number of code cells
        - markdown_cells: number of markdown cells
    """
    path = Path(notebook_path)
    result: dict[str, Any] = {
        "path": str(path),
        "valid": False,
        "errors": [],
        "warnings": [],
        "metadata": {},
        "cell_count": 0,
        "code_cells": 0,
        "markdown_cells": 0,
    }

    # Check file exists
    if not path.exists():
        result["errors"].append(f"File not found: {path}")
        return result

    # Check extension
    if path.suffix != ".ipynb":
        result["warnings"].append(f"File extension is '{path.suffix}', expected '.ipynb'")

    # Try to read and parse the notebook
    try:
        with path.open("r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)
    except json.JSONDecodeError as e:
        result["errors"].append(f"Invalid JSON: {e}")
        return result
    except Exception as e:
        result["errors"].append(f"Failed to read notebook: {e}")
        return result

    # Validate notebook format
    try:
        nbformat.validate(notebook)
    except NBValidationError as e:
        result["errors"].append(f"Notebook format validation failed: {e.message}")
        return result

    # Count cells
    cells = notebook.get("cells", [])
    result["cell_count"] = len(cells)
    result["code_cells"] = sum(1 for c in cells if c.get("cell_type") == "code")
    result["markdown_cells"] = sum(1 for c in cells if c.get("cell_type") == "markdown")

    # Check for empty notebook
    if result["cell_count"] == 0:
        result["warnings"].append("Notebook has no cells")

    # Extract neural_search metadata
    nb_metadata = notebook.get("metadata", {})
    neural_search_meta = nb_metadata.get("neural_search", {})
    result["metadata"] = neural_search_meta

    if not neural_search_meta:
        result["warnings"].append("No neural_search metadata found in notebook")
    else:
        if not neural_search_meta.get("dataset_id"):
            result["warnings"].append("Missing dataset_id in neural_search metadata")
        if not neural_search_meta.get("asset_id"):
            result["warnings"].append("Missing asset_id in neural_search metadata")

    # Check for required sections (by looking for markdown headers)
    required_sections = [
        "Setup",
        "Load NWB",
        "Session Metadata",
        "Acquisition",
        "Processing",
        "Units",
        "Trials",
    ]

    found_sections = set()
    for cell in cells:
        if cell.get("cell_type") == "markdown":
            source = cell.get("source", "")
            for section in required_sections:
                if section.lower() in source.lower():
                    found_sections.add(section)

    missing_sections = set(required_sections) - found_sections
    if missing_sections:
        result["warnings"].append(f"Missing recommended sections: {sorted(missing_sections)}")

    # Check for TODO section
    has_todo = any(
        "todo" in cell.get("source", "").lower()
        for cell in cells
        if cell.get("cell_type") in ("markdown", "code")
    )
    if not has_todo:
        result["warnings"].append("No TODO section found for analysis roadmap")

    # Check code cells for syntax errors (basic check)
    for i, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = cell.get("source", "")
            # Filter out IPython magic commands before syntax check
            filtered_lines = []
            for line in source.split("\n"):
                stripped = line.strip()
                if stripped.startswith("%") or stripped.startswith("!"):
                    continue  # Skip magic commands
                filtered_lines.append(line)
            filtered_source = "\n".join(filtered_lines)

            if not filtered_source.strip():
                continue  # Skip empty cells

            try:
                compile(filtered_source, f"<cell_{i}>", "exec")
            except SyntaxError as e:
                result["errors"].append(f"Syntax error in code cell {i}: {e.msg} (line {e.lineno})")

    # Set valid flag
    result["valid"] = len(result["errors"]) == 0

    return result


def format_validation_result(result: dict[str, Any], verbose: bool = False) -> str:
    """Format validation result for display."""
    lines = []

    status = "VALID" if result["valid"] else "INVALID"
    lines.append(f"Notebook: {result['path']}")
    lines.append(f"Status: {status}")
    lines.append("")

    lines.append(f"Cells: {result['cell_count']} total ({result['code_cells']} code, {result['markdown_cells']} markdown)")

    if result["metadata"]:
        lines.append("")
        lines.append("Neural Search Metadata:")
        for key, value in result["metadata"].items():
            lines.append(f"  {key}: {value}")

    if result["errors"]:
        lines.append("")
        lines.append("Errors:")
        for error in result["errors"]:
            lines.append(f"  - {error}")

    if result["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for warning in result["warnings"]:
            lines.append(f"  - {warning}")

    if result["valid"] and not result["warnings"]:
        lines.append("")
        lines.append("No issues found.")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for notebook validation."""
    parser = argparse.ArgumentParser(
        prog="python -m neural_search.notebooks.validate",
        description="Validate a generated NWB starter notebook.",
    )
    parser.add_argument(
        "notebook_path",
        type=Path,
        help="Path to the .ipynb file to validate.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (exit with error code if warnings present).",
    )
    args = parser.parse_args(argv)

    result = validate_notebook_structure(args.notebook_path)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_validation_result(result))

    # Exit code
    if not result["valid"]:
        return 1
    if args.strict and result["warnings"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
