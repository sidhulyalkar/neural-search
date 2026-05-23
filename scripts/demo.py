#!/usr/bin/env python3
"""
Neural Search Demo Script

A polished demo showcasing the complete Neural Search MVP workflow:
1. Load the behavioral task ontology
2. Load fixture datasets
3. Generate dataset cards
4. Index search records
5. Run benchmark queries
6. Generate one NWB starter notebook
7. Print the local frontend URL
8. Print example queries to try

Usage:
    python scripts/demo.py
    make demo
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from neural_search.cards import generate_dataset_card_json
from neural_search.evaluation.run_benchmark import run_full_benchmark, generate_markdown_report
from neural_search.ingestion.demo_seed import build_demo_seed, seed_demo_database
from neural_search.notebooks.generator import generate_nwb_starter_notebook
from neural_search.ontology import get_all_tasks, load_ontology, match_tasks
from neural_search.reports.dataset_compilation import (
    compile_dataset_report,
    generate_markdown_report as generate_compilation_markdown,
)
from neural_search.search import search_datasets


# Terminal colors
class Colors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


def print_header(text: str) -> None:
    """Print a styled header."""
    print()
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}  {text}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print()


def print_step(step: int, text: str) -> None:
    """Print a numbered step."""
    print(f"{Colors.GREEN}{Colors.BOLD}[{step}]{Colors.RESET} {Colors.BOLD}{text}{Colors.RESET}")


def print_substep(text: str) -> None:
    """Print a substep."""
    print(f"    {Colors.DIM}{text}{Colors.RESET}")


def print_success(text: str) -> None:
    """Print a success message."""
    print(f"    {Colors.GREEN}✓{Colors.RESET} {text}")


def print_warning(text: str) -> None:
    """Print a warning message."""
    print(f"    {Colors.YELLOW}!{Colors.RESET} {text}")


def print_info(text: str) -> None:
    """Print an info message."""
    print(f"    {Colors.CYAN}→{Colors.RESET} {text}")


def run_demo() -> int:
    """Run the full Neural Search demo."""
    start_time = time.time()

    print_header("NEURAL SEARCH DEMO")
    print(f"    {Colors.DIM}Experiment-aware neural data discovery{Colors.RESET}")
    print(f"    {Colors.DIM}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")

    # Step 1: Load ontology
    print()
    print_step(1, "Loading behavioral task ontology...")
    ontology = load_ontology()
    tasks = get_all_tasks()
    categories = sorted(set(t.category for t in tasks))
    print_success(f"Loaded {len(tasks)} tasks across {len(categories)} categories")
    print_substep(f"Categories: {', '.join(categories)}")

    # Step 2: Load fixture datasets
    print()
    print_step(2, "Loading fixture datasets...")
    demo_data = build_demo_seed()
    print_success(f"Loaded {len(demo_data)} datasets")
    for record in demo_data:
        ds = record["dataset"]
        modalities = ds.get("modalities", [])
        print_substep(f"{ds['source_id']}: {ds['title'][:45]}... [{', '.join(modalities[:2])}]")

    # Step 3: Generate dataset cards
    print()
    print_step(3, "Generating dataset cards...")
    cards_generated = 0
    for record in demo_data:
        ds = record["dataset"]
        extraction = record.get("extraction")
        papers = record.get("papers", [])
        if extraction:
            card = generate_dataset_card_json(ds, extraction, papers)
            record["card"] = card
            cards_generated += 1
            readiness = card.analysis_readiness.score
            print_substep(f"{ds['source_id']}: readiness={readiness}/100")
    print_success(f"Generated {cards_generated} dataset cards")

    # Step 4: Seed database
    print()
    print_step(4, "Seeding demo database...")
    db_result = seed_demo_database()
    print_success(f"Database seeded: {db_result['database_url']}")
    print_substep(f"Datasets: {db_result['datasets']}, Papers: {db_result['papers']}, Cards: {db_result['cards']}")

    # Step 5: Run benchmark queries
    print()
    print_step(5, "Running benchmark evaluation...")
    try:
        report = run_full_benchmark(datasets=demo_data)
        passed = sum(1 for q in report.queries if q.label_recall_at_10 >= 0.5)
        total = len(report.queries)
        print_success(f"Benchmark complete: {passed}/{total} queries passed")
        print_substep(f"Mean Precision@5: {report.mean_precision_at_5:.1%}")
        print_substep(f"Mean Label Recall@10: {report.mean_label_recall_at_10:.1%}")

        # Save report
        reports_dir = PROJECT_ROOT / "data" / "eval" / "results"
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / "latest_eval_report.md"
        md_path.write_text(generate_markdown_report(report), encoding="utf-8")
        print_substep(f"Report saved: {md_path.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        print_warning(f"Benchmark skipped: {e}")

    # Step 6: Generate dataset compilation report
    print()
    print_step(6, "Generating compilation report...")
    try:
        comp_report = compile_dataset_report()
        reports_dir = PROJECT_ROOT / "data" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        md_path = reports_dir / "dataset_compilation_report.md"
        md_path.write_text(generate_compilation_markdown(comp_report), encoding="utf-8")
        json_path = reports_dir / "dataset_compilation_report.json"
        json_path.write_text(json.dumps(comp_report, indent=2, default=str), encoding="utf-8")
        print_success(f"Compilation report generated")
        print_substep(f"Total datasets: {comp_report['summary']['total_datasets']}")
        print_substep(f"Reports saved to: {reports_dir.relative_to(PROJECT_ROOT)}/")
    except Exception as e:
        print_warning(f"Compilation report skipped: {e}")

    # Step 7: Generate NWB starter notebook
    print()
    print_step(7, "Generating NWB starter notebook...")
    notebooks_dir = PROJECT_ROOT / "data" / "notebooks" / "generated"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    # Pick first dataset
    ds = demo_data[0]["dataset"]
    asset = demo_data[0].get("assets", [{"id": "demo", "path": "sub-01/session.nwb"}])[0]
    card = demo_data[0].get("card")

    output_path = notebooks_dir / f"{ds['source_id']}.ipynb"
    response = generate_nwb_starter_notebook(ds, asset, output_path, card=card)

    if response.valid:
        print_success(f"Generated notebook: {output_path.relative_to(PROJECT_ROOT)}")
        # Count cells in the generated notebook
        import nbformat
        with open(output_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            total_cells = len(nb.cells)
            code_cells = sum(1 for c in nb.cells if c.cell_type == 'code')
            md_cells = sum(1 for c in nb.cells if c.cell_type == 'markdown')
        print_substep(f"Cells: {total_cells} ({code_cells} code, {md_cells} markdown)")
    else:
        print_warning(f"Notebook generated with warnings: {response.warnings}")

    # Step 8: Run demo search
    print()
    print_step(8, "Running demo search...")
    demo_query = "Find reversal learning datasets with reward omission"
    results = search_datasets(demo_query, {}, demo_data, limit=5)
    print_success(f"Search: \"{demo_query}\"")
    for i, result in enumerate(results.results[:3], 1):
        why = result.why_matched[0] if result.why_matched else "keyword match"
        print_substep(f"{i}. {result.dataset_id} (score: {result.score:.1f}) - {why}")

    # Print final summary
    elapsed = time.time() - start_time
    print_header("DEMO COMPLETE")

    print(f"  {Colors.GREEN}✓{Colors.RESET} Ontology loaded: {len(tasks)} tasks")
    print(f"  {Colors.GREEN}✓{Colors.RESET} Datasets indexed: {len(demo_data)}")
    print(f"  {Colors.GREEN}✓{Colors.RESET} Cards generated: {cards_generated}")
    print(f"  {Colors.GREEN}✓{Colors.RESET} Notebook created: {output_path.name}")
    print(f"  {Colors.GREEN}✓{Colors.RESET} Benchmark passed: {passed}/{total} queries")
    print()
    print(f"  {Colors.DIM}Completed in {elapsed:.1f}s{Colors.RESET}")

    # Frontend URL
    print()
    print(f"{Colors.BOLD}LOCAL FRONTEND{Colors.RESET}")
    print(f"  {Colors.CYAN}http://localhost:5173{Colors.RESET}  (run: make web)")
    print(f"  {Colors.CYAN}http://localhost:8000{Colors.RESET}  (run: make api)")

    # Example queries
    print()
    print(f"{Colors.BOLD}EXAMPLE QUERIES TO TRY{Colors.RESET}")
    example_queries = [
        "Find reversal learning datasets with reward omission",
        "Go/NoGo task with calcium imaging in mPFC",
        "Visual decision making with Neuropixels",
        "Delay discounting fiber photometry rat",
        "Human ECoG reaching and grasping BCI",
    ]
    for i, query in enumerate(example_queries, 1):
        print(f"  {i}. {Colors.DIM}{query}{Colors.RESET}")

    # Next steps
    print()
    print(f"{Colors.BOLD}DEMO WALKTHROUGH{Colors.RESET}")
    print("  1. Search: \"Find reversal learning datasets with reward omission\"")
    print("  2. Open top dataset card to see why it matched")
    print("  3. Review missing metadata warnings")
    print("  4. Generate starter notebook from dataset card")
    print("  5. Run benchmark dashboard to see evaluation metrics")
    print("  6. View dataset compilation report for coverage stats")

    print()
    print(f"{Colors.BOLD}QUICK START{Colors.RESET}")
    print("  make api   # Start API server on :8000")
    print("  make web   # Start frontend on :5173")
    print()

    return 0


def main() -> int:
    """Entry point."""
    try:
        return run_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted.")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
