#!/usr/bin/env python3
"""Run the full vault export pipeline in sequence.

Usage:
    python scripts/obsidian/compile_vault.py \
        --corpus data/corpus/normalized/combined_corpus.jsonl \
        --queries artifacts/benchmark_queries.jsonl \
        --audit-queue artifacts/eval/audit_queue.jsonl \
        --vault obsidian_vault
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--audit-queue", required=True)
    parser.add_argument("--vault", required=True)
    args = parser.parse_args()

    py = sys.executable
    run([py, "scripts/obsidian/init_vault.py", "--vault", args.vault])
    run([py, "scripts/obsidian/export_dataset_cards.py",
         "--corpus", args.corpus, "--vault", args.vault])
    run([py, "scripts/obsidian/export_query_cards.py",
         "--queries", args.queries, "--vault", args.vault])
    run([py, "scripts/obsidian/export_audit_queue.py",
         "--audit-queue", args.audit_queue, "--vault", args.vault])
    print("Vault compilation complete.")


if __name__ == "__main__":
    main()
