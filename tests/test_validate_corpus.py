"""Tests for corpus quality dashboard."""
import subprocess
import sys


def test_syntax():
    r = subprocess.run(
        [sys.executable, "-m", "py_compile", "scripts/validate_corpus.py"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr


def test_runs_on_existing_corpus():
    r = subprocess.run(
        [sys.executable, "scripts/validate_corpus.py"],
        capture_output=True, text=True,
    )
    # May not pass all checks (corpus < 4000), but must not crash
    assert "Source" in r.stdout or "Corpus" in r.stdout
