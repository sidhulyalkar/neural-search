#!/usr/bin/env python
"""Download Allen Mouse Connectivity Atlas data via the Allen SDK.

Run from the repo root:
    python scripts/ingestion/download_allen_connectivity.py

Outputs (into data/allen/connectivity/):
    manifest.json             — Allen SDK cache manifest
    experiments_df.csv        — 469+ injection experiments with source → projection metadata
    structure_tree.csv        — Allen CCF structure hierarchy with acronyms

Requires:
    pip install allensdk pandas
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "allen" / "connectivity"
DATA_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = DATA_DIR / "manifest.json"
EXPERIMENTS_PATH = DATA_DIR / "experiments_df.csv"
STRUCTURE_PATH = DATA_DIR / "structure_tree.csv"


def download():
    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
        import pandas as pd
    except ImportError:
        log.error("allensdk not installed. Run: pip install allensdk pandas")
        sys.exit(1)

    log.info("Initialising MouseConnectivityCache (manifest: %s)…", MANIFEST_PATH)
    mcc = MouseConnectivityCache(manifest_file=str(MANIFEST_PATH), resolution=100)

    # ── Experiment list ────────────────────────────────────────────────────
    if EXPERIMENTS_PATH.exists():
        log.info("Experiments CSV already exists at %s — skipping.", EXPERIMENTS_PATH)
    else:
        log.info("Fetching experiment list from Allen API (≈2 MB)…")
        experiments = mcc.get_experiments(dataframe=True)
        log.info("Fetched %d experiments.", len(experiments))
        experiments.to_csv(EXPERIMENTS_PATH, index=False)
        log.info("Saved → %s", EXPERIMENTS_PATH)

    # ── Structure tree ────────────────────────────────────────────────────
    if STRUCTURE_PATH.exists():
        log.info("Structure tree already exists at %s — skipping.", STRUCTURE_PATH)
    else:
        log.info("Fetching Allen CCF structure tree…")
        st = mcc.get_structure_tree()
        nodes = st.nodes()
        import pandas as pd
        df = pd.DataFrame([
            {
                "id": n["id"],
                "acronym": n["acronym"],
                "name": n["name"],
                "parent_structure_id": n.get("parent_structure_id"),
                "depth": n.get("depth", 0),
                "color_hex_triplet": n.get("color_hex_triplet", ""),
            }
            for n in nodes
        ])
        df.to_csv(STRUCTURE_PATH, index=False)
        log.info("Saved structure tree (%d nodes) → %s", len(df), STRUCTURE_PATH)

    log.info("Done. Allen connectivity data ready in %s/", DATA_DIR)
    log.info(
        "To fetch full projection density volumes (large!), "
        "call mcc.get_projection_matrix() with specific structure IDs."
    )


if __name__ == "__main__":
    download()
