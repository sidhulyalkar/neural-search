"""Build and merge knowledge-graph subgraphs from computed spectral bundles.

Reads a JSONL file of ``SpectralFeatureBundle`` records (as produced by
``scripts/reanalysis/run_aperiodic_batch.py``), builds one subgraph per
bundle via ``neural_search.spectral.kg.build_spectral_subgraph``, merges
them together, and writes the combined graph as JSON.

Usage:
    python scripts/kg/ingest_spectral_features.py \
        --bundles artifacts/spectral/bundles.jsonl \
        --out artifacts/graph/spectral_subgraph.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from neural_search.graph.builder import merge_graphs  # noqa: E402
from neural_search.graph.schema import write_graph_json  # noqa: E402
from neural_search.spectral.kg import build_spectral_subgraph  # noqa: E402
from neural_search.spectral.schemas import SpectralFeatureBundle  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lines = [
        line
        for line in args.bundles.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    subgraphs = []
    for line in lines:
        bundle = SpectralFeatureBundle.model_validate_json(line)
        subgraphs.append(build_spectral_subgraph(bundle))

    if not subgraphs:
        print("No bundles found; nothing to ingest.", file=sys.stderr)
        return

    combined = merge_graphs(subgraphs)
    write_graph_json(combined, args.out)
    print(
        f"Wrote {len(combined.nodes)} nodes / {len(combined.edges)} edges "
        f"from {len(subgraphs)} bundles to {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
