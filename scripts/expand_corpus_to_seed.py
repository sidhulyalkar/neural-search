#!/usr/bin/env python3
"""Transform corpus YAML to seed YAML format.

This script reads the expanded corpus definitions and generates
properly formatted seed YAML files for demo_datasets.yaml and
demo_papers.yaml.

Usage:
    python scripts/expand_corpus_to_seed.py
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

CORPUS_PATH = Path("data/corpus/demo_neural_datasets.yaml")
SEED_DIR = Path("data/seed")
DATASETS_OUT = SEED_DIR / "demo_datasets.yaml"
PAPERS_OUT = SEED_DIR / "demo_papers.yaml"

# Default license for demo datasets
DEFAULT_LICENSE = "CC-BY-4.0"


def slugify(text: str) -> str:
    """Convert text to lowercase slug."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def corpus_to_seed_dataset(corpus_ds: dict) -> dict:
    """Transform corpus dataset format to seed dataset format."""
    dataset_id = corpus_ds["dataset_id"]
    source_name = corpus_ds.get("source_name", "demo")

    # Map behavioral_events to behaviors
    behaviors = corpus_ds.get("behavioral_events", [])

    # Determine if dataset has behavior and trials from content
    tasks = corpus_ds.get("tasks", [])
    has_behavior = len(behaviors) > 0 or any(
        t in tasks for t in [
            "go_nogo", "reversal_learning", "delay_discounting", "reaching",
            "visual_decision_making", "motor_imagery", "spatial_navigation",
            "working_memory", "foraging", "operant_conditioning"
        ]
    )
    has_trials = any(
        ev in behaviors for ev in [
            "trial_start", "trial_end", "trial_outcome", "choice",
            "stimulus_onset", "response", "reward", "error"
        ]
    ) or has_behavior  # Most behavioral tasks have trials

    # Generate URL from dataset_id
    url_slug = slugify(dataset_id.replace("DEMO_", ""))
    url = f"https://example.org/demo/{url_slug}"

    # Paper ID
    paper_id = f"PAPER_{dataset_id.replace('DEMO_', '')}"

    return {
        "source": source_name,
        "source_id": dataset_id,
        "title": corpus_ds["title"],
        "description": corpus_ds["description"],
        "url": url,
        "species": corpus_ds.get("species", []),
        "modalities": corpus_ds.get("modalities", []),
        "brain_regions": corpus_ds.get("brain_regions", []),
        "tasks": tasks,
        "behaviors": behaviors,
        "data_standards": corpus_ds.get("data_standards", []),
        "has_behavior": has_behavior,
        "has_trials": has_trials,
        "license": DEFAULT_LICENSE,
        "linked_paper_ids": [paper_id],
    }


def generate_paper_for_dataset(corpus_ds: dict) -> dict:
    """Generate a demo paper for a corpus dataset."""
    dataset_id = corpus_ds["dataset_id"]
    paper_id = f"PAPER_{dataset_id.replace('DEMO_', '')}"
    slug = slugify(dataset_id.replace("DEMO_", ""))

    # Generate abstract from description and analysis goals
    description = corpus_ds["description"]
    analysis_goals = corpus_ds.get("analysis_goals", [])
    keywords = corpus_ds.get("keywords", [])

    abstract_parts = [f"A demo paper describing {description.lower()}"]
    if analysis_goals:
        abstract_parts.append(f"Supports analyses including {', '.join(analysis_goals[:3])}.")

    abstract = " ".join(abstract_parts)

    # Concepts from tasks, modalities, and keywords
    concepts = (
        corpus_ds.get("tasks", [])[:2]
        + corpus_ds.get("modalities", [])[:2]
        + keywords[:2]
    )
    concepts = [c.replace(" ", "_").lower() for c in concepts]

    return {
        "id": paper_id,
        "openalex_id": f"W-{slug.upper()[:20]}",
        "doi": f"10.0000/demo.{slug[:30]}",
        "title": f"Demo paper: {corpus_ds['title']}",
        "abstract": abstract,
        "publication_year": 2024,
        "authors_json": [{"name": "Demo Author"}],
        "url": f"https://example.org/papers/{slug}",
        "concepts": concepts[:5],
    }


def main() -> int:
    """Transform corpus YAML and write seed files."""
    if not CORPUS_PATH.exists():
        print(f"Error: Corpus file not found: {CORPUS_PATH}")
        return 1

    with open(CORPUS_PATH, encoding="utf-8") as f:
        corpus = yaml.safe_load(f)

    if not corpus or "datasets" not in corpus:
        print("Error: Corpus file missing 'datasets' key")
        return 1

    # Transform datasets
    seed_datasets = []
    seed_papers = []

    for corpus_ds in corpus["datasets"]:
        seed_ds = corpus_to_seed_dataset(corpus_ds)
        seed_datasets.append(seed_ds)

        paper = generate_paper_for_dataset(corpus_ds)
        seed_papers.append(paper)

    # Write seed files
    SEED_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATASETS_OUT, "w", encoding="utf-8") as f:
        yaml.dump(
            {"datasets": seed_datasets},
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

    with open(PAPERS_OUT, "w", encoding="utf-8") as f:
        yaml.dump(
            {"papers": seed_papers},
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=120,
        )

    print(f"Generated {len(seed_datasets)} datasets -> {DATASETS_OUT}")
    print(f"Generated {len(seed_papers)} papers -> {PAPERS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
