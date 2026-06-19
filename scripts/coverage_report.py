#!/usr/bin/env python3
"""Print a human-readable coverage report from the DuckDB ledger."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural_search.coverage.duckdb_store import CoverageStore

store = CoverageStore()
summary = store.coverage_summary()
print(f"\n{'='*60}")
print("Neural Search Coverage Report")
print(f"{'='*60}")
print(f"Total datasets:  {summary['total_datasets']:,}")
print(f"Total entries:   {summary['total_entries']:,}")
print()
print(f"{'Dimension':<20} {'Datasets':>10} {'Coverage':>10}")
print("-" * 45)
for dim, stats in sorted(summary["dimension_coverage"].items()):
    print(f"{dim:<20} {stats['datasets']:>10,} {stats['pct']:>9.1f}%")

# source_coverage_rates returns DuckDBPyRelation: (source, n_total, regions_covered,
# regions_pct, modalities_covered, modalities_pct, species_covered, species_pct,
# tasks_covered, tasks_pct)
rates = store.source_coverage_rates().fetchall()
print(f"\n{'Source':<20} {'Datasets':>10} {'Regions':>10} {'Modalities':>12}")
print("-" * 55)
for r in sorted(rates, key=lambda x: -x[1])[:10]:
    print(f"{r[0]:<20} {r[1]:>10,} {r[3]:>9.1f}% {r[5]:>11.1f}%")

# dark_pairs returns DuckDBPyRelation: (dim_a_value, dim_b_value, n_observed,
# a_marginal, b_marginal, opportunity_score)
dark = store.dark_pairs(top_n=5).fetchall()
print("\nTop dark pairs (unexplored region×modality combos):")
for p in dark:
    print(f"  {p[0]} × {p[1]}: opportunity={p[5]:.2f}")

store.close()
