#!/usr/bin/env python
"""Download NeuroSynth v7 database files via NiMARE.

This script only downloads the raw files — no conversion step needed.
neurosynth_builder.py reads the TSV/NPZ files directly.

Run from the repo root:
    python scripts/ingestion/download_neurosynth.py

Downloads to: data/neurosynth/neurosynth/
  - data-neurosynth_version-7_coordinates.tsv.gz
  - data-neurosynth_version-7_metadata.tsv.gz
  - data-neurosynth_version-7_vocab-terms_source-abstract_type-tfidf_features.npz
  - data-neurosynth_version-7_vocab-terms_vocabulary.txt
  (plus LDA50/100/200/400 vocabulary files)

After this completes, run:
    python -m neural_search.ingestion.neurosynth_builder

Requires: pip install "statsmodels>=0.14.0" nimare
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "neurosynth"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def download() -> None:
    try:
        from nimare.extract import fetch_neurosynth
    except ImportError:
        log.error("nimare not installed. Run: pip install 'statsmodels>=0.14.0' nimare nibabel nilearn")
        sys.exit(1)

    log.info("Fetching NeuroSynth v7 via NiMARE (skips files that already exist)…")
    raw = fetch_neurosynth(data_dir=str(DATA_DIR), version="7", overwrite=False)

    # Verify raw TSV/NPZ files are in place
    nested = DATA_DIR / "neurosynth"
    coords = list(nested.glob("*_coordinates.tsv.gz")) if nested.exists() else []
    features = list(nested.glob("*vocab-terms*_features.npz")) if nested.exists() else []

    if coords and features:
        size_mb = coords[0].stat().st_size / (1024 ** 2)
        feat_mb = features[0].stat().st_size / (1024 ** 2)
        log.info("Coordinates file: %s (%.1f MB)", coords[0].name, size_mb)
        log.info("Features file:    %s (%.1f MB)", features[0].name, feat_mb)
        log.info("All files in place. Run:")
        log.info("  python -m neural_search.ingestion.neurosynth_builder")
    else:
        log.warning("Expected files not found under %s", nested)
        log.warning("raw return: %r", raw)


if __name__ == "__main__":
    download()
