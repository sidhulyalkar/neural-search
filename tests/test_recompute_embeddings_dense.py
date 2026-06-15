"""Test that recompute_embeddings.py accepts --provider dense."""
import subprocess
import sys
from pathlib import Path

import pytest

_HAS_CORPUS = bool(list(Path("data/corpus/normalized").glob("real_*.jsonl"))) if Path("data/corpus/normalized").exists() else False


@pytest.mark.skipif(not _HAS_CORPUS, reason="real_*.jsonl corpus files not present in this environment")
def test_dry_run_default_provider():
    r = subprocess.run(
        [sys.executable, "scripts/recompute_embeddings.py", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


@pytest.mark.skipif(not _HAS_CORPUS, reason="real_*.jsonl corpus files not present in this environment")
def test_dry_run_dense_provider():
    r = subprocess.run(
        [sys.executable, "scripts/recompute_embeddings.py",
         "--provider", "dense", "--dry-run"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert "dense" in r.stdout.lower() or "bge" in r.stdout.lower()


def test_syntax():
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/recompute_embeddings.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
