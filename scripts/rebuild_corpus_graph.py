#!/usr/bin/env python3
"""Rebuild the KnowledgeGraph from all normalized corpus files.

Reads the real_*.jsonl corpus files, builds the graph via
build_real_corpus_graph.py, and reports node/edge counts.

Output: data/graph/neural_search_graph.real_corpus.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus" / "normalized"
GRAPH_OUT = PROJECT_ROOT / "data" / "graph" / "neural_search_graph.real_corpus.json"

# Files consumed by build_real_corpus_graph.py (same list as that script)
CORPUS_FILES = [
    "real_dandi.jsonl",
    "real_openneuro.jsonl",
    "real_allen.jsonl",
    "real_nemo.jsonl",
    "real_papers.jsonl",
]


def main() -> int:
    print("Corpus files:")
    for name in CORPUS_FILES:
        fp = CORPUS_DIR / name
        if fp.exists():
            lines = [ln for ln in fp.read_text().splitlines() if ln.strip()]
            print(f"  {name}: {len(lines)} records")
        else:
            print(f"  {name}: NOT FOUND (will be skipped by builder)")

    print("\nRunning build_real_corpus_graph.py ...")
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "build_real_corpus_graph.py")],
        capture_output=False,   # stream output live
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        print(f"Graph builder exited with code {result.returncode}")
        return 1

    # Verify output and print summary
    if GRAPH_OUT.exists():
        g = json.loads(GRAPH_OUT.read_text())
        n_nodes = len(g.get("nodes", {}))
        n_edges = len(g.get("edges", {}))
        print(f"\nGraph rebuilt: {n_nodes} nodes, {n_edges} edges")
        print(f"Written to: {GRAPH_OUT}")
    else:
        # Fallback: search common alternate paths
        for alt in [
            PROJECT_ROOT / "data" / "graphs" / "real_corpus_graph.json",
            PROJECT_ROOT / "data" / "graphs" / "knowledge_graph.json",
            PROJECT_ROOT / "data" / "knowledge_graph.json",
        ]:
            if alt.exists():
                g = json.loads(alt.read_text())
                n_nodes = len(g.get("nodes", {}))
                n_edges = len(g.get("edges", {}))
                print(f"\nGraph rebuilt: {n_nodes} nodes, {n_edges} edges")
                print(f"Written to: {alt}")
                break
        else:
            print("WARNING: Could not locate graph output file after build.")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
