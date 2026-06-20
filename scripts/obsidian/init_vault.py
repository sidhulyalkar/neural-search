#!/usr/bin/env python3
"""Scaffold the Obsidian vault folder structure.

Usage:
    python scripts/obsidian/init_vault.py --vault obsidian_vault
"""
from __future__ import annotations

import argparse
from pathlib import Path

VAULT_FOLDERS = [
    "00_Project",
    "01_Rubrics",
    "02_Ontology",
    "03_Datasets",
    "04_Queries",
    "05_Annotations/Human Audits",
    "06_Evaluations",
    "07_Whitepaper",
    "08_Dashboards",
    "99_Templates",
]


def init_vault(vault: Path) -> None:
    vault.mkdir(parents=True, exist_ok=True)
    for folder in VAULT_FOLDERS:
        folder_path = vault / folder
        folder_path.mkdir(parents=True, exist_ok=True)
        gitkeep = folder_path / ".gitkeep"
        if not gitkeep.exists():
            gitkeep.touch()
    print(f"Vault initialised at {vault} ({len(VAULT_FOLDERS)} folders)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vault", required=True, type=Path)
    args = parser.parse_args()
    init_vault(args.vault)


if __name__ == "__main__":
    main()
