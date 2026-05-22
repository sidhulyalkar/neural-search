#!/usr/bin/env python3
"""
Neural Search Demo Script

Demonstrates the core functionality:
1. Load the behavioral task ontology
2. Ingest sample dataset records
3. Generate dataset cards
4. Run example searches
5. Generate a starter notebook
"""

import json
import tempfile
from pathlib import Path

from neural_search.cards import generate_dataset_card_json
from neural_search.ingestion.demo_seed import build_demo_seed
from neural_search.notebooks import generate_nwb_starter_notebook
from neural_search.ontology import get_all_tasks, load_ontology, match_tasks
from neural_search.search import search_datasets


def main() -> None:
    print("=" * 60)
    print("Neural Search Demo")
    print("=" * 60)
    print()

    # Step 1: Load ontology
    print("1. Loading behavioral task ontology...")
    ontology = load_ontology()
    tasks = get_all_tasks()
    print(f"   Loaded {len(tasks)} tasks")
    print(f"   Categories: {sorted(set(t.category for t in tasks))}")
    print()

    # Step 2: Load demo datasets
    print("2. Loading demo dataset records...")
    demo_data = build_demo_seed()
    print(f"   Loaded {len(demo_data)} datasets:")
    for record in demo_data:
        ds = record["dataset"]
        print(f"   - {ds['source_id']}: {ds['title'][:50]}...")
    print()

    # Step 3: Generate dataset cards
    print("3. Generating dataset cards...")
    for record in demo_data:
        ds = record["dataset"]
        card = record.get("card")
        if card:
            print(f"   - {ds['source_id']}: readiness={card.analysis_readiness.score}/100")
    print()

    # Step 4: Test ontology matching
    print("4. Testing ontology matching...")
    test_queries = [
        "reversal learning",
        "go/no-go task",
        "two-alternative forced choice",
        "neuropixels",
    ]
    for query in test_queries:
        matches = match_tasks(query)
        if matches:
            print(f"   '{query}' -> {matches[0].label} (confidence: {matches[0].confidence:.2f})")
        else:
            print(f"   '{query}' -> No match")
    print()

    # Step 5: Run example searches
    print("5. Running example searches...")
    search_queries = [
        "Find reversal learning datasets with reward omission",
        "Go/NoGo task with calcium imaging",
        "Decision making with neuropixels",
    ]
    for query in search_queries:
        results = search_datasets(query, {}, demo_data, limit=3)
        print(f"\n   Query: '{query}'")
        for i, result in enumerate(results.results[:3], 1):
            print(f"      {i}. {result.dataset_id} (score: {result.score:.1f})")
            if result.why_matched:
                print(f"         Why: {result.why_matched[0]}")
    print()

    # Step 6: Generate a starter notebook
    print("6. Generating starter notebook...")
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "demo_notebook.ipynb"
        ds = demo_data[0]["dataset"]
        asset = {"id": "demo", "path": "sub-01/session.nwb"}
        response = generate_nwb_starter_notebook(ds, asset, output_path)
        print(f"   Generated: {output_path.name}")
        print(f"   Valid: {response.valid}")
        if response.warnings:
            print(f"   Warnings: {response.warnings}")
    print()

    print("=" * 60)
    print("Demo complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  - Run the API: make api")
    print("  - Run the frontend: make web")
    print("  - Open http://localhost:3000")


if __name__ == "__main__":
    main()
