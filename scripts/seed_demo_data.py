#!/usr/bin/env python3
"""
Seed demo data into the database.

Usage:
    python scripts/seed_demo_data.py [--database-url URL]
"""

import argparse
import os

from neural_search.ingestion.demo_seed import seed_demo_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed demo data into database")
    parser.add_argument(
        "--database-url",
        default=os.getenv("DATABASE_URL", "sqlite:///./neural_search.db"),
        help="Database URL (default: sqlite:///./neural_search.db)",
    )
    args = parser.parse_args()

    print(f"Seeding demo data to: {args.database_url}")
    summary = seed_demo_database(args.database_url)

    print("\nSeed complete!")
    print(f"  Datasets: {summary['datasets']}")
    print(f"  Papers: {summary['papers']}")
    print(f"  Cards: {summary.get('cards', 'N/A')}")
    print(f"  Embeddings: {summary.get('embeddings', 'N/A')}")


if __name__ == "__main__":
    main()
