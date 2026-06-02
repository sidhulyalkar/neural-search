"""Tests for 5-layer deduplication pipeline."""
import subprocess
import sys


def test_dry_run():
    r = subprocess.run(
        [sys.executable, "scripts/dedup_corpus.py", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "DRY RUN" in r.stdout


def test_syntax():
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/dedup_corpus.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


def test_dedup_layer1_exact_doi():
    import sys
    sys.path.insert(0, ".")
    from scripts.dedup_corpus import layer1_exact_ids

    records = [
        {"dataset_id": "ds1", "doi": "10.0001/same", "title": "Dataset A"},
        {"dataset_id": "ds2", "doi": "10.0001/same", "title": "Dataset B (copy)"},
        {"dataset_id": "ds3", "doi": "10.0002/different", "title": "Dataset C"},
    ]
    dupes = layer1_exact_ids(records)
    dupe_ids = {pair[0] for pair in dupes} | {pair[1] for pair in dupes}
    assert "ds1" in dupe_ids or "ds2" in dupe_ids
    assert "ds3" not in dupe_ids
