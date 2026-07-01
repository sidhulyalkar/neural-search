#!/usr/bin/env python
"""Download Allen Mouse Connectivity Atlas data via the Allen Brain Atlas REST API.

Run from the repo root:
    python scripts/ingestion/download_allen_connectivity.py

Outputs (into data/allen/connectivity/):
    experiments_df.csv   — Injection experiments with source → projection metadata
    structure_tree.csv   — Allen CCF structure hierarchy with acronyms

No additional dependencies required (uses httpx, which is already installed).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import httpx
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "allen" / "connectivity"
DATA_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS_PATH = DATA_DIR / "experiments_df.csv"
STRUCTURE_PATH = DATA_DIR / "structure_tree.csv"

_BASE = "https://api.brain-map.org/api/v2/data"


def _fetch_all(url: str, *, timeout: float = 60.0) -> list[dict]:
    """Fetch a paginated Allen REST endpoint; return all msg records."""
    resp = httpx.get(url, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("success"):
        raise RuntimeError(f"Allen API error: {body.get('msg')}")
    return body.get("msg", [])


def download() -> None:
    # ── Experiment list ────────────────────────────────────────────────────
    if EXPERIMENTS_PATH.exists():
        log.info("Experiments CSV already exists at %s — skipping.", EXPERIMENTS_PATH)
    else:
        log.info("Fetching mouse connectivity experiments from Allen API…")
        records = _fetch_all(
            f"{_BASE}/SectionDataSet/query.json"
            "?criteria=[failed$eqfalse],products[abbreviation$eqMouse%20Connectivity]"
            "&num_rows=5000&count=false"
        )
        log.info("Fetched %d experiments.", len(records))
        pd.DataFrame(records).to_csv(EXPERIMENTS_PATH, index=False)
        log.info("Saved → %s", EXPERIMENTS_PATH)

    # ── Structure tree ────────────────────────────────────────────────────
    if STRUCTURE_PATH.exists():
        log.info("Structure tree already exists at %s — skipping.", STRUCTURE_PATH)
    else:
        log.info("Fetching Allen CCF structure tree…")
        nodes = _fetch_all(
            f"{_BASE}/Structure/query.json"
            "?criteria=[graph_id$eq1]&num_rows=5000&count=false"
        )
        df = pd.DataFrame([
            {
                "id": n.get("id"),
                "acronym": n.get("acronym"),
                "name": n.get("name"),
                "parent_structure_id": n.get("parent_structure_id"),
                "depth": n.get("depth", 0),
                "color_hex_triplet": n.get("color_hex_triplet", ""),
            }
            for n in nodes
        ])
        df.to_csv(STRUCTURE_PATH, index=False)
        log.info("Saved structure tree (%d nodes) → %s", len(df), STRUCTURE_PATH)

    log.info("Done. Allen connectivity data ready in %s/", DATA_DIR)


if __name__ == "__main__":
    try:
        download()
    except Exception as exc:
        log.error("Download failed: %s", exc)
        sys.exit(1)
